import config
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.dataset.uv_face import mean_shape_map_np, uv_kpt_ind
from skimage import io, transform
from torchvision import transforms
from src.dataset.uv_face import face_mask_np

face_mask = torch.from_numpy(face_mask_np).float()
    

def kpt2TformBatchWeighted(kpt_src, kpt_dst, W):
    sum_W = torch.sum(W, dim=(1, 2), keepdim=True)
    sum_W = sum_W + torch.rand(sum_W.shape).to(DEVICE) * 1e-8 
    centroid_src = torch.sum(W.matmul(kpt_src), dim=1, keepdim=True) / sum_W
    centroid_dst = torch.sum(W.matmul(kpt_dst), dim=1, keepdim=True) / sum_W

    sum_dist1 = torch.sum(torch.norm(kpt_src - centroid_src, dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    sum_dist2 = torch.sum(torch.norm(kpt_dst - centroid_dst, dim=2), dim=1).unsqueeze(-1).unsqueeze(-1)
    sum_dist1 = sum_dist1 + torch.rand(sum_dist1.shape).to(DEVICE) * 1e-8 
    sum_dist2 = sum_dist2 + torch.rand(sum_dist2.shape).to(DEVICE) * 1e-8 
    
    S = sum_dist2 / sum_dist1

    A = kpt_src * S
    B = kpt_dst
    mu_A = A.mean(dim=1, keepdim=True)
    mu_B = B.mean(dim=1, keepdim=True)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.permute(0, 2, 1).matmul(W).matmul(BB)
    H = H + (torch.rand(H.shape).to(DEVICE)-0.5)*2 * 1e-8 
    U, _S, V = torch.svd(H)
    R = V.matmul(U.permute(0, 2, 1))
    t = torch.mean(B - A.matmul(R.permute(0, 2, 1)), dim=1)
    return R * sum_dist2 / sum_dist1, t, R


class VisibilityRebuildModule(nn.Module):
    def __init__(self, select_n=200, only_foreface=True):
        super(VisibilityRebuildModule, self).__init__()
        self.mean_posmap_tensor = nn.Parameter(torch.from_numpy(mean_shape_map_np.transpose((2, 0, 1))))
        self.mean_posmap_tensor.requires_grad = False
        self.select_n = select_n
        self.only_foreface = only_foreface
        self.face_mask = face_mask.clone()

    def forward(self, Offset, Posmap_kpt, confidence=None, add_mean_face=True, mask_from_dst=True):
        B, C, W, H = Posmap_kpt.shape
        if add_mean_face:
            Offset = Offset + self.mean_posmap_tensor
        offsetmap = Offset.permute(0, 2, 3, 1)
        kptmap = Posmap_kpt.permute(0, 2, 3, 1)
        if confidence is not None:
            confidence = confidence[:,0,:,:]
        
        if mask_from_dst:
            mask = (torch.sum(kptmap,dim=-1)!=0)*1.0
        else:
            mask = (torch.sum(offsetmap,dim=-1)!=0)*1.0
            
        if self.only_foreface:
            if self.face_mask.device != kptmap.device:
                self.face_mask = self.face_mask.to(kptmap.device)
            mask = mask * self.face_mask
        kpt_dst = torch.zeros((B,self.select_n,3), device=Posmap_kpt.device)
        kpt_src = torch.zeros((B,self.select_n,3), device=Posmap_kpt.device)
        Weight = torch.ones((B, self.select_n), device=Posmap_kpt.device)
        for b in range(B):
            mask_idx = torch.nonzero(mask[b])
            if len(mask_idx) > 0: 
                mask_idx_select = torch.randint(len(mask_idx),(self.select_n,))
            else: 
                continue
            
            mask_idx_x = mask_idx[mask_idx_select,0]
            mask_idx_y = mask_idx[mask_idx_select,1]
            kpt_dst[b] = kptmap[b, mask_idx_x, mask_idx_y]
            kpt_src[b] = offsetmap[b, mask_idx_x, mask_idx_y]
            if confidence is not None:
                Weight[b] = confidence[b, mask_idx_x, mask_idx_y]

        R, T, R_rot = kpt2TformBatchWeighted(kpt_src, kpt_dst, torch.diag_embed(Weight))
        
        outpos = offsetmap.matmul(R.permute(0, 2, 1).unsqueeze(1)) + T.unsqueeze(1).unsqueeze(1)
        outpos = outpos.reshape((B, W, H, C))
        outpos = outpos.permute(0, 3, 1, 2)
        shape_map = offsetmap.matmul(R.permute(0, 2, 1).unsqueeze(1))
        shape_map = shape_map.reshape((B, W, H, C))
        shape_map = shape_map.permute(0, 3, 1, 2)
        return outpos, shape_map, R_rot


def get_R_rot_from_Tm(Tm): # input Tm (bs,12)
    bs = Tm.shape[0]
    Tm = Tm.reshape((bs, 3, 4))
    Tm = Tm[:,:,:3]
    ones = torch.ones((bs, 1), device=Tm.device)
    H = ones.repeat(1,3)
    H = torch.diag_embed(H).to(Tm.device)
    H[:,1,1] = -1.
    R_rot = torch.bmm(H, Tm)
    return R_rot
    
    
def transform_by_Tm(offset_uvm, Tm):
    bs = offset_uvm.shape[0]
    Tm = Tm.reshape((bs,3,4))
    R_rot = get_R_rot_from_Tm(Tm)
    mean_shape_map_torch = torch.from_numpy(mean_shape_map_np[None,...].transpose(0,3,1,2)).to(offset_uvm.device)
    offset_uvm = (offset_uvm + mean_shape_map_torch)/config.OFFSET_FIX_RATE
    ones = torch.ones((offset_uvm.shape[0], 1, config.UV_MAP_SIZE, config.UV_MAP_SIZE)).to(offset_uvm.device)
    uvm4d = torch.cat([offset_uvm, ones], dim=1)
    face_uvm = uvm4d.permute(0,2,3,1).matmul(Tm.unsqueeze(1).permute(0,1,3,2))
    face_uvm = face_uvm[...,:3].permute(0,3,1,2)
    face_uvm = face_uvm / POSMAP_FIX_RATE
    face_uvm_nose = face_uvm[:,:, uv_kpt_ind[30, 0], uv_kpt_ind[30, 1]]
    face_uvm_onface_depth = face_uvm[:, [2]] - face_uvm_nose[:, [2]].unsqueeze(2).unsqueeze(3)
    face_uvm = torch.cat([face_uvm[:, :2], face_uvm_onface_depth], dim=1)
    return face_uvm, R_rot

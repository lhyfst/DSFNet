import sys
import os
sys.path.append(os.getcwd())

import config
from src.model.modules import *
from src.model.loss import *
from src.dataset.uv_face import mean_shape_map_np

import torch
import time
from src.model.hrnet import get_hrnet
from src.model.unet import UNet
from src.model.pointnet import PointNet
from src.util.morphablemodel import MorphableModel as BFM_UV


class DSFNet(nn.Module):
    def __init__(self):
        super(DSFNet, self).__init__()
        
        self.hrnet = get_hrnet()
        self.unet = UNet(in_channels=12, n_classes=3, feature_scale=4)
        self.pointnet = PointNet(num_pts=2048)
        self.rebuilder = VisibilityRebuildModule(select_n=2048)
        
        self.mean_shape_map = torch.from_numpy(mean_shape_map_np.transpose((2, 0, 1))).to(config.DEVICE)
        meshgrid_u, meshgrid_v = torch.meshgrid(torch.arange(256), torch.arange(256))
        self.meshgrid_uv = torch.stack([meshgrid_v, meshgrid_u], dim=-1).type(torch.float)
        self.bfm_uv = BFM_UV(model_path='./data/Out/BFM.mat', model_type='BFM_UV', device=config.DEVICE)

    def forward(self, inpt, targets, mode='predict'):
        img = inpt['img']
        bs = img.shape[0]
        out = self.hrnet(img)
        
        grid = out['grid']
        seg = out['seg']
        conf_is = out['image_space_conf']
        depth = out['depth']
        shape_para_ms = out['shape_para']
        exp_para_ms = out['exp_para']
        Tm_ms = out['Tm']
            
        seg_threhold = 0.5
        seg_mask = seg>seg_threhold

        seg_mask_valid = torch.sum(seg_mask, dim=(1,2,3))
        seg_mask[seg_mask_valid==0] = True
        
        grid = grid*seg_mask
        grid_int = (grid+1)/2
        grid_int = (grid_int*256).long()
        grid_int = torch.clamp(grid_int, min=0, max=255)
        
        grid_uv = torch.zeros((bs,256,256,2))
        for idx in range(bs):
            grid_uv[idx,grid_int[idx,1,:,:],grid_int[idx,0,:,:]] = self.meshgrid_uv
        grid_uv = grid_uv.to(config.DEVICE)
        grid_uv = grid_uv/128 -1
        
        # depth kpt
        meshgrid_y, meshgrid_x = torch.meshgrid(torch.arange(256), torch.arange(256))
        meshgrid_x = meshgrid_x[None,None,...].repeat(bs,1,1,1).to(config.DEVICE)/280
        meshgrid_y = meshgrid_y[None,None,...].repeat(bs,1,1,1).to(config.DEVICE)/280
        dkpt_img = torch.cat([meshgrid_x, meshgrid_y, depth], dim=1) # pred
        dkpt_img = dkpt_img*seg_mask
        dkpt_uv = F.grid_sample(dkpt_img, grid_uv, mode='nearest', align_corners=False)
        conf_is_uv = F.grid_sample(conf_is, grid_uv, mode='nearest', align_corners=False)
        mask_uv = F.grid_sample(seg_mask*1.0, grid_uv, mode='nearest', align_corners=False).squeeze(1)
        
        batch_mean_face = self.mean_shape_map[None,...].repeat(bs,1,1,1)
        dkpt_uv_initpose, _shape_uvm, _R_rot = self.rebuilder(dkpt_uv,batch_mean_face,confidence=conf_is_uv,add_mean_face=False,mask_from_dst=False)
        _Tm_is, shape_para_is, exp_para_is, feature_is = self.pointnet(torch.cat([dkpt_uv_initpose,conf_is_uv],dim=1), mask_uv=mask_uv)
        offset_uvm_is = self.bfm_uv.compute_offset_uvm(shape_para_is, exp_para_is)
        face_uvm_is, shape_uvm_is, R_rot_is = self.rebuilder(offset_uvm_is, dkpt_uv, confidence=conf_is_uv)
    
        offset_uvm_ms = self.bfm_uv.compute_offset_uvm(shape_para_ms, exp_para_ms)
        face_uvm_ms, R_rot_ms = transform_by_Tm(offset_uvm_ms, Tm_ms)

        offset_uvm_f, kpt_uvm_f = self.unet(torch.cat([face_uvm_is, face_uvm_ms, offset_uvm_is, offset_uvm_ms],dim=1),
                                         feature=out['model_space_feature'])
        face_uvm_f, shape_uvm_f, R_rot_f = self.rebuilder(offset_uvm_f, kpt_uvm_f, confidence=conf_is_uv)

        out = dict()
        out['face_uvm'] = face_uvm_f
        out['R_rot'] = R_rot_f
        return out


def get_model():
    model = DSFNet()
    return model


import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import sys
import os
sys.path.append(os.getcwd())

from src.dataset.uv_face import face_mask_np
face_mask = torch.from_numpy(face_mask_np).float()


class PointNet(nn.Module):
    def __init__(self, num_pts=2048):
        super(PointNet,self).__init__()
        
        self.num_pts = num_pts
        self.face_mask = face_mask.clone()
        self.pts_dim = 6
        
        self.conv1 = torch.nn.Conv1d(self.pts_dim,64,1)
        self.conv2 = torch.nn.Conv1d(64,64,1)
        self.conv3 = torch.nn.Conv1d(64,64,1)
        self.conv4 = torch.nn.Conv1d(64,128,1)
        self.conv5 = torch.nn.Conv1d(128,1024,1)
        self.conv6_Tm = nn.Conv1d(1024, 12, 1)
        self.conv6_shape = nn.Conv1d(1024, 199, 1)
        self.conv6_exp = nn.Conv1d(1024, 29, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6_shape = nn.BatchNorm1d(199)
        self.bn6_exp = nn.BatchNorm1d(29)
        self.max_pool = nn.MaxPool1d(num_pts)
        
        
    def preprocess_kptmap(self, x, mask_uv):
        B, C, W, H = x.shape
        kptmap = x.permute(0, 2, 3, 1)
        mask = mask_uv

        if self.face_mask.device != x.device:
            self.face_mask = self.face_mask.to(x.device)
        mask = mask * self.face_mask

        pts = torch.zeros((B,self.num_pts,self.pts_dim), device=x.device)

        for b in range(B):
            mask_idx = torch.nonzero(mask[b])
            if len(mask_idx) > 0:
                mask_idx_select = torch.randint(len(mask_idx),(self.num_pts,))
            else: 
                continue

            mask_idx_x = mask_idx[mask_idx_select,0]
            mask_idx_y = mask_idx[mask_idx_select,1]
            pts[b,:,:C] = kptmap[b, mask_idx_x, mask_idx_y]
            uvpos = torch.cat([mask_idx_x[...,None], mask_idx_y[...,None]],dim=1)
            uvpos = uvpos / 128 - 1 
            pts[b,:,C:C+2] = uvpos

        x = pts.permute(0,2,1)
        return x

    def forward(self,x, mask_uv=None):
        x = self.preprocess_kptmap(x, mask_uv)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        global_features = self.max_pool(out)

        out_Tm = self.conv6_Tm(global_features).squeeze(2)
        out_shape = self.bn6_shape(self.conv6_shape(global_features)).squeeze(2)
        out_exp = self.bn6_exp(self.conv6_exp(global_features)).squeeze(2)

        return out_Tm, out_shape, out_exp, global_features.squeeze(2)
    

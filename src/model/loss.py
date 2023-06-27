import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import *
import skimage.io as io
from src.dataset.uv_face import face_mask_np, face_mask_fix_rate, foreface_ind, uv_kpt_ind

face_mask = torch.from_numpy(face_mask_np).float()

class NME(nn.Module):
    def __init__(self, rate=1.0):
        super(NME, self).__init__()
        self.rate = rate
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        self.face_mask = self.face_mask.to(y_true.device)
        pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist) * self.rate * 100


class NME2D(nn.Module):
    def __init__(self):
        super(NME2D, self).__init__()
        self.face_mask = face_mask.clone()

    def forward(self, y_true, y_pred):
        if self.face_mask.device != y_true.device:
            self.face_mask = self.face_mask.to(y_true.device)
        pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred[:, :2] - gt[:, :2], dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist) * 100


class KptNME(nn.Module):
    def __init__(self):
        super(KptNME, self).__init__()

    def forward(self, y_true, y_pred):
        gt = y_true[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        pred = y_pred[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist) * 100


class KptNME2D(nn.Module):
    def __init__(self):
        super(KptNME2D, self).__init__()

    def forward(self, y_true, y_pred):
        gt = y_true[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        pred = y_pred[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]]
        for i in range(y_true.shape[0]):
            pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
            gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
        dist = torch.mean(torch.norm(pred[:, :2] - gt[:, :2], dim=1), dim=1)
        left = torch.min(gt[:, 0, :], dim=1)[0]
        right = torch.max(gt[:, 0, :], dim=1)[0]
        top = torch.min(gt[:, 1, :], dim=1)[0]
        bottom = torch.max(gt[:, 1, :], dim=1)[0]
        bbox_size = torch.sqrt((right - left) * (bottom - top))
        dist = dist / bbox_size
        return torch.mean(dist) * 100


class FastAlignment:
    def __init__(self):
        super(FastAlignment, self).__init__()

    def __call__(self, uvm_src, uvm_dst):
        B, C, W, H = uvm_src.shape
        pts_dst = uvm_dst[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]].permute(0, 2, 1)
        pts_src = uvm_src[:, :, uv_kpt_ind[:, 0], uv_kpt_ind[:, 1]].permute(0, 2, 1)
        R, T = self.get_tform_batch(pts_src, pts_dst)
        output_uvm = uvm_src.permute(0, 2, 3, 1).reshape(B, W * H, C).matmul(R.permute(0, 2, 1)) + T.unsqueeze(1)
        output_uvm = output_uvm.reshape(B, W, H, C).permute(0, 3, 1, 2)
        return output_uvm

    def get_tform_batch(self, pts_src, pts_dst):
        sum_dist1 = torch.sum(torch.norm(pts_src - pts_src.mean(dim=1, keepdim=True), dim=2), dim=1).unsqueeze(
            -1).unsqueeze(-1)
        sum_dist2 = torch.sum(torch.norm(pts_dst - pts_dst.mean(dim=1, keepdim=True), dim=2), dim=1).unsqueeze(
            -1).unsqueeze(-1)
        A = pts_src * sum_dist2 / sum_dist1
        B = pts_dst
        mu_A = A.mean(dim=1, keepdim=True)
        mu_B = B.mean(dim=1, keepdim=True)
        AA = A - mu_A
        BB = B - mu_B
        H = AA.permute(0, 2, 1).matmul(BB)
        H = H + (torch.rand(H.shape).to(DEVICE)-0.5)*2 * 1e-8 
        U, S, V = torch.svd(H)
        R = V.matmul(U.permute(0, 2, 1))
        t = torch.mean(B - A.matmul(R.permute(0, 2, 1)), dim=1)
        return R * sum_dist2 / sum_dist1, t


def cp(kpt_src, kpt_dst):
    sum_dist1 = np.sum(np.linalg.norm(kpt_src - kpt_src[0], axis=1))
    sum_dist2 = np.sum(np.linalg.norm(kpt_dst - kpt_dst[0], axis=1))
    A = kpt_src * sum_dist2 / sum_dist1
    B = kpt_dst
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B
    H = AA.T.dot(BB)
    H = H + (np.random.rand(*H.shape)-0.5)*2 * 1e-8 
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # if np.linalg.det(R) < 0:
    #     print('singular R')
    #     Vt[2, :] *= -1
    #     R = Vt.T.dot(U.T)
    t = mu_B - mu_A.dot(R.T)
    R = R * sum_dist2 / sum_dist1
    tform = np.zeros((4, 4))
    tform[0:3, 0:3] = R
    tform[0:3, 3] = t
    tform[3, 3] = 1
    return tform


class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.ICP = FastAlignment()

    def forward_torch(self, y_true, y_pred):
        aligned_pred = self.ICP(y_pred, y_true)
        outer_interocular_vec = y_true[:, :, uv_kpt_ind[36, 0], uv_kpt_ind[36, 1]] - y_true[:, :, uv_kpt_ind[45, 0], uv_kpt_ind[45, 1]]
        outer_interocular_dist = torch.norm(outer_interocular_vec, dim=1, keepdim=True)
        pred = aligned_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
        dist = torch.mean(torch.norm(pred - gt, p=2,dim=1), dim=1)
        dist = dist / outer_interocular_dist
        return torch.mean(dist)

    def forward(self, y_true, y_pred):
        y_true = y_true[0].cpu().permute(1, 2, 0).numpy()
        y_pred = y_pred[0].cpu().permute(1, 2, 0).numpy()

        y_pred_vertices = y_pred[face_mask_np > 0]
        y_true_vertices = y_true[face_mask_np > 0]

        Tform = cp(y_pred_vertices, y_true_vertices)

        y_fit_vertices = y_pred_vertices.dot(Tform[0:3, 0:3].T) + Tform[0:3, 3]
        dist = np.linalg.norm(y_fit_vertices - y_true_vertices, axis=1)

        outer_interocular_dist = y_true[uv_kpt_ind[36, 0], uv_kpt_ind[36, 1]] - y_true[
            uv_kpt_ind[45, 0], uv_kpt_ind[45, 1]]
        bbox_size = np.linalg.norm(outer_interocular_dist[0:3])

        dist = torch.from_numpy(dist)
        loss = torch.mean(dist / bbox_size)
        return loss


def complete_skip_data(data, skip_indices): 
    # (1969,3) -> (2000,3)
    ret = np.zeros((2000,3))
    idx = 0
    for i in range(2000):
        if i in skip_indices:
            continue
        else:
            ret[i] = data[idx]
            idx += 1
    return ret

def benchmark_FOE(pred_pose, valid_idx=np.arange(2000)):
    """
    pred_angles: (2000,3) # [pitch-yaw-roll] 
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99] (following FSA-Net https://github.com/shamangary/FSA-Net)
    """
    exclude_aflw2000 = '../SynergyNet/aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = '../SynergyNet/aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'
    gt_pose_ = np.load(exclude_aflw2000) 
    skip_indices_ = np.load(skip_aflw2000)

    valid_mask = np.full((2000,1),False)
    valid_mask[valid_idx] = True
    valid_mask[skip_indices_] = False

    gt_pose = complete_skip_data(gt_pose_, skip_indices_)

    pose_analyis = np.sum(np.abs(pred_pose-gt_pose)*valid_mask,axis=0)/np.sum(valid_mask)
    MAE = np.mean(pose_analyis)
    yaw = pose_analyis[1]
    pitch = pose_analyis[0]
    roll = pose_analyis[2]
    msg = 'Mean MAE = %3.3f (in deg), [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll)
    print('\nFace orientation estimation: ')
    print(msg)
    return {'MAE': MAE, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
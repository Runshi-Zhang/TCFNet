import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
import MPED
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
from earth_movers_distance.emd import EarthMoverDistance
MPEDloss = MPED.MPED_VALUE
EMD = EarthMoverDistance()
def pariwise_l2_norm2_batch(x, y):
    nump_x = x.shape[1]
    nump_y = y.shape[1]

    xx = torch.unsqueeze(x, -1)
    xx = torch.tile(xx, (1,1,1,nump_y))

    yy = torch.unsqueeze(y, -1)
    yy = torch.tile(yy, (1,1,1,nump_x))

    yy = yy.permute(0, 3, 2, 1)

    diff = torch.subtract(xx, yy)
    square_diff = torch.square(diff)
    square_dist = torch.sum(square_diff,dim=2)
    return square_dist

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)
def emd(pcs1, pcs2):
    dists = EMD(pcs1, pcs2)
    return torch.sum(dists)

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2
class CDLoss(torch.nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self):
        super(CDLoss, self).__init__()
    def forward(self, y_pred, y_true):
        d1, d2, _, _ = chamfer_dist(y_pred, y_true)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2


class GeometricLoss(torch.nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, nnk=16, densityWeight=1.0):
        super(GeometricLoss, self).__init__()
        self.nnk = nnk
        self.densityWeight = densityWeight

    def forward(self, y_pred, y_true):
        #calculate shape loss
        square_dist = pariwise_l2_norm2_batch(y_true, y_pred)
        dist = torch.sqrt(square_dist)
        minrow,index = torch.min(dist, dim=2)

        mincol,index = torch.min(dist, dim=1)

        shapeLoss = (torch.mean(minrow) + torch.mean(mincol)) / 2

        #calculate  density loss
        square_dist2 = pariwise_l2_norm2_batch(y_true, y_true)
        dist2 = torch.sqrt(square_dist2)
        knndis, index = torch.topk(torch.negative(dist), k=self.nnk)
        knndis2, index = torch.topk(torch.negative(dist2), k=self.nnk)
        densityLoss = torch.mean(torch.abs(knndis - knndis2))

        dataloss = shapeLoss + densityLoss * self.densityWeight
        return dataloss, shapeLoss, densityLoss


class RegularizingLoss(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(RegularizingLoss, self).__init__()

    def forward(self, x_true, y_true, x_pred, y_pred):

        displacements_A = torch.cat((x_true, y_pred), dim=2)
        displacements_B = torch.cat((y_true, x_pred), dim=2)

        square_dist = pariwise_l2_norm2_batch(displacements_A, displacements_B)
        dist = torch.sqrt(square_dist)

        minrow,index = torch.min(dist, dim=2)
        mincol,index = torch.min(dist, dim=1)
        regularLoss = (torch.mean(minrow) + torch.mean(mincol)) / 2

        return regularLoss

class DCDLoss(torch.nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self):
        super(DCDLoss, self).__init__()
    def calc_cd(self, output, gt, return_raw=False, normalize=False, separate=False):
        dist1, dist2, idx1, idx2 = chamfer_dist(gt, output)
        cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
        cd_t = (dist1.mean(1) + dist2.mean(1))

        if separate:
            res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
                   torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
        else:
            res = [cd_p, cd_t]

        if return_raw:
            res.extend([dist1, dist2, idx1, idx2])
        return res

    def forward(self, x, gt, alpha=50, n_lambda=0.5, return_raw=False, non_reg=False):
        x = x.float()
        gt = gt.float()
        batch_size, n_x, _ = x.shape
        batch_size, n_gt, _ = gt.shape
        assert x.shape[0] == gt.shape[0]

        if non_reg:
            frac_12 = max(1, n_x / n_gt)
            frac_21 = max(1, n_gt / n_x)
        else:
            frac_12 = n_x / n_gt
            frac_21 = n_gt / n_x

        cd_p, cd_t, dist1, dist2, idx1, idx2 = self.calc_cd(x, gt, return_raw=True)
        # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
        # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
        # dist2 and idx2: vice versa
        exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

        loss1 = []
        loss2 = []
        for b in range(batch_size):
            count1 = torch.bincount(idx1[b])
            weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
            weight1 = (weight1 + 1e-6) ** (-1) * frac_21
            loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

            count2 = torch.bincount(idx2[b])
            weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
            weight2 = (weight2 + 1e-6) ** (-1) * frac_12
            loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

        loss1 = torch.stack(loss1)
        loss2 = torch.stack(loss2)
        loss = (loss1 + loss2) / 2

        #res = [loss, cd_p, cd_t]
        #if return_raw:
            #res.extend([dist1, dist2, idx1, idx2])

        return loss.mean()


class AuxiLoss(torch.nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self):
        super(AuxiLoss, self).__init__()


    def forward(self, A, B, predicted_A, predicted_B, path_local, names, emd_used = False, mped_used=False):
        dev = A.device
        file = np.loadtxt(path_local + str(int(names.item())) + '.txt')
        file = file / 300.0
        index_bone_bi = torch.ones([20480, 3], dtype=torch.int8, device=dev)
        index_bone_bi[A[0, :, 0] < file[0, 0]] = 0
        index_bone_bi[A[0, :, 0] > file[0, 3]] = 0
        index_bone_bi[A[0, :, 2] > file[0, 2]] = 0
        index_bone_bi[A[0, :, 2] < file[0, 5]] = 0
        # b1 = torch.sum(index_bone_bi == 1)

        index_bone_zui = torch.ones([20480, 3], dtype=torch.int8, device=dev)
        index_bone_zui[A[0, :, 0] < file[1, 0]] = 0
        index_bone_zui[A[0, :, 0] > file[1, 3]] = 0
        index_bone_zui[A[0, :, 1] > min(file[1, 1], file[1, 4])] = 0
        index_bone_zui[A[0, :, 2] > file[1, 2]] = 0
        index_bone_zui[A[0, :, 2] < file[1, 5]] = 0
        # b2 = torch.sum(index_bone_zui == 1)

        index_face_bi = torch.ones([20480, 3], dtype=torch.int8, device=dev)
        index_face_bi[B[0, :, 0] < file[2, 0]] = 0
        index_face_bi[B[0, :, 0] > file[2, 3]] = 0
        index_face_bi[B[0, :, 2] > file[2, 2]] = 0
        index_face_bi[B[0, :, 2] < file[2, 5]] = 0
        # b3 = torch.sum(index_face_bi == 1)

        index_face_zui = torch.ones([20480, 3], dtype=torch.int8, device=dev)
        index_face_zui[B[0, :, 0] < file[1, 0]] = 0
        index_face_zui[B[0, :, 0] > file[3, 3]] = 0
        index_face_zui[B[0, :, 1] > min(file[3, 1], file[3, 4])] = 0
        index_face_zui[B[0, :, 2] > file[3, 2]] = 0
        index_face_zui[B[0, :, 2] < file[3, 5]] = 0
        # b4 = torch.sum(index_face_zui == 1)
        index = torch.cat(
            (index_bone_bi[:, None], index_bone_zui[:, None], index_face_bi[:, None], index_face_zui[:, None]),
            dim=1)
        index = torch.transpose(index, 1, 2)
        index = torch.transpose(index, 0, 1)
        A_ground_bi = A[0, index[0, :, 0] == 1, :]
        A_ground_zui = A[0, index[0, :, 1] == 1, :]
        A_predict_bi = predicted_A[0, index[0, :, 2] == 1, :]
        A_predict_zui = predicted_A[0, index[0, :, 3] == 1, :]

        B_ground_bi = B[0, index[0, :, 2] == 1, :]
        B_ground_zui = B[0, index[0, :, 3] == 1, :]
        B_predict_bi = predicted_B[0, index[0, :, 0] == 1, :]
        B_predict_zui = predicted_B[0, index[0, :, 1] == 1, :]


        A_ground_zui = A_ground_zui[None,...]
        A_predict_bi = A_predict_bi[None,...]
        A_ground_bi = A_ground_bi[None,...]
        A_predict_zui = A_predict_zui[None,...]

        B_ground_zui = B_ground_zui[None,...]
        B_predict_bi = B_predict_bi[None,...]
        B_ground_bi = B_ground_bi[None,...]
        B_predict_zui = B_predict_zui[None,...]
        cd1_local_bi_bone = chamfer_sqrt(A_predict_bi, A_ground_bi)
        cd1_local_zui_bone = chamfer_sqrt(A_predict_zui, A_ground_zui)
        cd1_local_bi_face = chamfer_sqrt(B_predict_bi, B_ground_bi)
        cd1_local_zui_face = chamfer_sqrt(B_predict_zui, B_ground_zui)
        loss = cd1_local_bi_bone + cd1_local_zui_bone + cd1_local_bi_face + cd1_local_zui_face


        if(emd_used):
            emd_local_bi_bone = emd(A_predict_bi, A_ground_bi)
            emd_local_zui_bone = emd(A_predict_zui, A_ground_zui)
            emd_local_bi_face = emd(B_predict_bi, B_ground_bi)
            emd_local_zui_face = emd(B_predict_zui, B_ground_zui)
            loss = loss + 0.01 * (emd_local_bi_bone + emd_local_bi_face + emd_local_zui_face + emd_local_zui_bone)
        if(mped_used):
            mped_local_bi_bone = MPEDloss(A_predict_bi, A_ground_bi)
            mped_local_zui_bone = MPEDloss(A_predict_zui, A_ground_zui)
            mped_local_bi_face = MPEDloss(B_predict_bi, B_ground_bi)
            mped_local_zui_face = MPEDloss(B_predict_zui, B_ground_zui)
            loss = loss + 0.01 * (mped_local_bi_bone + mped_local_bi_face + mped_local_zui_face + mped_local_zui_bone)
        return loss



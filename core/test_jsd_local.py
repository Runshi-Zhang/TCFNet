import logging
import os

import numpy as np
import torch

import MPED

from jsd import jsd_between_point_cloud_sets

from models.model_newp2p_20480 import P2PNet as Model
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from datasets import datasets
import glob
from torch.utils.data import DataLoader
from earth_movers_distance.emd import EarthMoverDistance

EMD = EarthMoverDistance()
from scipy.spatial.distance import directed_hausdorff


def hausdorff_distance(setA, setB):
    dist_A_B = directed_hausdorff(setA, setB)[0]
    dist_B_A = directed_hausdorff(setB, setA)[0]
    return max(dist_A_B, dist_B_A)


def emd(pcs1, pcs2):
    dists = EMD(pcs1, pcs2)
    return torch.sum(dists)


chamfer_dist = chamfer_3DDist()


def random_subsample(pcd, n_points=4096):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    device = pcd.device
    batch_idx = torch.arange(b, dtype=torch.long, device=device).reshape((-1, 1)).repeat(1, n_points)
    idx = torch.cat([torch.randperm(n, dtype=torch.long, device=device)[:n_points].reshape((1, -1)) for i in range(b)],
                    0)
    return pcd[batch_idx, idx, :]


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def lr_lambda(epoch):
    if 0 <= epoch <= 100:
        return 1
    elif 100 < epoch <= 150:
        return 0.5
    elif 150 < epoch <= 250:
        return 0.1
    else:
        return 0.5


def test_net():
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    MPEDloss = MPED.MPED_VALUE
    torch.backends.cudnn.benchmark = True
    val_dir = r'/home/zrs/pyproject/pointcloudd/data/test/'
    load_chepoint_path = r'/home/zrs/pyproject/TwoStage/exp/output/checkpoints/ours/data2-16/ckpt-epoch-060.pth'
    save_path = r'/home/zrs/result/data20480/data_2/'

    val_set = datasets.FaceBoneDenseInferDataset(sorted(glob.glob(val_dir + '*h5'), key=lambda name: int(name[42:-3])))

    val_data_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    path_local = '/home/zrs/pyproject/local/'
    model = Model()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(load_chepoint_path)
    model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()
    with torch.no_grad():
        id = 0
        cd_all = []
        cd1_all = []
        cd2_all = []
        mped_all = []
        mped1_all = []
        mped2_all = []
        emd_all = []
        emd1_all = []
        emd2_all = []
        emd_zong = []
        emd1_zong = []
        emd2_zong = []
        cd_zong = []
        cd1_zong = []
        cd2_zong = []

        jsd_all = []
        jsd1_all = []
        jsd2_all = []
        hd_all = []
        hd1_all = []
        hd2_all = []

        cd_local = []
        cd1_local = []
        cd2_local = []
        mped_local = []
        mped1_local = []
        mped2_local = []
        emd_local = []
        emd1_local = []
        emd2_local = []

        hd_local = []
        hd1_local = []
        hd2_local = []

        print('============================ TEST RESULTS ============================')
        for data in val_data_loader:
            model.eval()
            data = [t.cuda() for t in data]
            A = data[0]
            B = data[1]
            names = data[2]
            # print(names)
            predicted_B_step, predicted_B, predicted_A_step, predicted_A = model(A, B)
            # print(emd(A, A))

            # local
            file = np.loadtxt(path_local + str(int(names[0, 0, 0].item())) + '.txt')
            file = file / 300.0
            index_bone_bi = torch.ones([20480, 3], dtype=torch.int8)
            index_bone_bi[A[0, :, 0] < file[0, 0]] = 0
            index_bone_bi[A[0, :, 0] > file[0, 3]] = 0
            index_bone_bi[A[0, :, 2] > file[0, 2]] = 0
            index_bone_bi[A[0, :, 2] < file[0, 5]] = 0
            #b1 = torch.sum(index_bone_bi == 1)

            index_bone_zui = torch.ones([20480, 3], dtype=torch.int8)
            index_bone_zui[A[0, :, 0] < file[1, 0]] = 0
            index_bone_zui[A[0, :, 0] > file[1, 3]] = 0
            index_bone_zui[A[0, :, 1] > min(file[1, 1], file[1, 4])] = 0
            index_bone_zui[A[0, :, 2] > file[1, 2]] = 0
            index_bone_zui[A[0, :, 2] < file[1, 5]] = 0
            #b2 = torch.sum(index_bone_zui == 1)

            index_face_bi = torch.ones([20480, 3], dtype=torch.int8)
            index_face_bi[B[0, :, 0] < file[2, 0]] = 0
            index_face_bi[B[0, :, 0] > file[2, 3]] = 0
            index_face_bi[B[0, :, 2] > file[2, 2]] = 0
            index_face_bi[B[0, :, 2] < file[2, 5]] = 0
            #b3 = torch.sum(index_face_bi == 1)

            index_face_zui = torch.ones([20480, 3], dtype=torch.int8)
            index_face_zui[B[0, :, 0] < file[1, 0]] = 0
            index_face_zui[B[0, :, 0] > file[3, 3]] = 0
            index_face_zui[B[0, :, 1] > min(file[3, 1], file[3, 4])] = 0
            index_face_zui[B[0, :, 2] > file[3, 2]] = 0
            index_face_zui[B[0, :, 2] < file[3, 5]] = 0
            #b4 = torch.sum(index_face_zui == 1)
            index = torch.cat(
                (index_bone_bi[:, None], index_bone_zui[:, None], index_face_bi[:, None], index_face_zui[:, None]),
                dim=1)
            index = torch.transpose(index, 1, 2)
            index = torch.transpose(index, 0, 1)
            A_ground_bi = A[0, index[0, :, 0] == 1, :][None, ...]
            A_ground_zui = A[0, index[0, :, 1] == 1, :][None, ...]
            A_predict_bi = predicted_A[0, index[0, :, 2] == 1, :][None, ...]
            A_predict_zui = predicted_A[0, index[0, :, 3] == 1, :][None, ...]

            B_ground_bi = B[0, index[0, :, 2] == 1, :][None, ...]
            B_ground_zui = B[0, index[0, :, 3] == 1, :][None, ...]
            B_predict_bi = predicted_B[0, index[0, :, 0] == 1, :][None, ...]
            B_predict_zui = predicted_B[0, index[0, :, 1] == 1, :][None, ...]

            cd1_local_bi_bone = chamfer_sqrt(A_predict_bi, A_ground_bi).item() * names[0, 0, 1]
            cd1_local_zui_bone = chamfer_sqrt(A_predict_zui, A_ground_zui).item() * names[0, 0, 1]
            cd1_local_bi_face = chamfer_sqrt(B_predict_bi, B_ground_bi).item() * names[0, 0, 2]
            cd1_local_zui_face = chamfer_sqrt(B_predict_zui, B_ground_zui).item() * names[0, 0, 2]
            cd1_local.append(cd1_local_bi_bone.item())
            cd1_local.append(cd1_local_zui_bone.item())
            cd2_local.append(cd1_local_bi_face.item())
            cd2_local.append(cd1_local_zui_face.item())
            cd_local.append(cd1_local_bi_bone.item() + cd1_local_bi_face.item())
            cd_local.append(cd1_local_zui_bone.item() + cd1_local_zui_face.item())

            emd_local_bi_bone = emd(A_predict_bi, A_ground_bi).item() * names[0, 0, 1] / A_predict_bi.shape[1]
            emd_local_zui_bone = emd(A_predict_zui, A_ground_zui).item() * names[0, 0, 1] / A_predict_zui.shape[1]
            emd_local_bi_face = emd(B_predict_bi, B_ground_bi).item() * names[0, 0, 2] / B_predict_bi.shape[1]
            emd_local_zui_face = emd(B_predict_zui, B_ground_zui).item() * names[0, 0, 2] / B_predict_zui.shape[1]
            emd1_local.append(emd_local_bi_bone.item())
            emd1_local.append(emd_local_zui_bone.item())
            emd2_local.append(emd_local_bi_face.item())
            emd2_local.append(emd_local_zui_face.item())
            emd_local.append(emd_local_bi_bone.item() + emd_local_bi_face.item())
            emd_local.append(emd_local_zui_bone.item() + emd_local_zui_face.item())

            mped_local_bi_bone = MPEDloss(A_predict_bi, A_ground_bi).item() * names[0, 0, 1]
            mped_local_zui_bone = MPEDloss(A_predict_zui, A_ground_zui).item() * names[0, 0, 1]
            mped_local_bi_face = MPEDloss(B_predict_bi, B_ground_bi).item() * names[0, 0, 2]
            mped_local_zui_face = MPEDloss(B_predict_zui, B_ground_zui).item() * names[0, 0, 2]
            mped1_local.append(mped_local_bi_bone.item())
            mped1_local.append(mped_local_zui_bone.item())
            mped2_local.append(mped_local_bi_face.item())
            mped2_local.append(mped_local_zui_face.item())
            mped_local.append(mped_local_bi_bone.item() + mped_local_bi_face.item())
            mped_local.append(mped_local_zui_bone.item() + mped_local_zui_face.item())

            hd_local_bi_bone = hausdorff_distance(A_predict_bi.cpu().numpy()[0, ...],
                                                  A_ground_bi.cpu().numpy()[0, ...]) * names[0, 0, 1]
            hd_local_zui_bone = hausdorff_distance(A_predict_zui.cpu().numpy()[0, ...],
                                                   A_ground_zui.cpu().numpy()[0, ...]) * names[0, 0, 1]
            hd_local_bi_face = hausdorff_distance(B_predict_bi.cpu().numpy()[0, ...],
                                                  B_ground_bi.cpu().numpy()[0, ...]) * names[0, 0, 2]
            hd_local_zui_face = hausdorff_distance(B_predict_zui.cpu().numpy()[0, ...],
                                                   B_ground_zui.cpu().numpy()[0, ...]) * names[0, 0, 2]
            hd1_local.append(hd_local_bi_bone.item())
            hd1_local.append(hd_local_zui_bone.item())
            hd2_local.append(hd_local_bi_face.item())
            hd2_local.append(hd_local_zui_face.item())
            hd_local.append(hd_local_bi_bone.item() + hd_local_bi_face.item())
            hd_local.append(hd_local_zui_bone.item() + hd_local_zui_face.item())

            jsd_local_bi_bone = jsd_between_point_cloud_sets(A_predict_bi.cpu().numpy(), A_ground_bi.cpu().numpy())
            jsd_local_zui_bone = jsd_between_point_cloud_sets(A_predict_zui.cpu().numpy(), A_ground_zui.cpu().numpy())
            jsd_local_bi_face = jsd_between_point_cloud_sets(B_predict_bi.cpu().numpy(), B_ground_bi.cpu().numpy())
            jsd_local_zui_face = jsd_between_point_cloud_sets(B_predict_zui.cpu().numpy(), B_ground_zui.cpu().numpy())


            cd1_one = chamfer_sqrt(predicted_A_step, A).item() * names[0, 0, 1]
            cd2_one = chamfer_sqrt(predicted_B_step, B).item() * names[0, 0, 2]
            emd1_one = emd(predicted_A_step, A).item() * names[0, 0, 1] / 20480.0
            emd2_one = emd(predicted_B_step, B).item() * names[0, 0, 2] / 20480.0

            hd1 = hausdorff_distance(predicted_A.cpu().numpy()[0, ...], A.cpu().numpy()[0, ...]) * names[0, 0, 1]
            hd2 = hausdorff_distance(predicted_B.cpu().numpy()[0, ...], B.cpu().numpy()[0, ...]) * names[0, 0, 2]

            hd1_all.append(hd1.item())
            hd2_all.append(hd2.item())
            hd_all.append(hd1.item() + hd2.item())

            jsd1 = jsd_between_point_cloud_sets(predicted_A.cpu().numpy(), A.cpu().numpy())
            jsd2 = jsd_between_point_cloud_sets(predicted_B.cpu().numpy(), B.cpu().numpy())
            jsd1_all.append(jsd1)
            jsd2_all.append(jsd2)
            jsd_all.append(jsd2 + jsd1)

            cd1_zong.append(cd1_one.item())
            cd2_zong.append(cd2_one.item())
            cd_zong.append(cd1_one.item() + cd2_one.item())

            emd1_zong.append(emd1_one.item())
            emd2_zong.append(emd2_one.item())
            emd_zong.append(emd1_one.item() + emd2_one.item())
            # A = A[...,0:3]
            # B = B[...,0:3]
            id = id + 1
            np.savetxt(save_path + str(int(names[0, 0, 0].item())) + 'bone.txt', A.cpu().numpy()[0] * 300)
            np.savetxt(save_path + str(int(names[0, 0, 0].item())) + 'bone_predicted.txt',
                       predicted_A.cpu().numpy()[0] * 300)
            np.savetxt(save_path + str(int(names[0, 0, 0].item())) + 'face.txt', B.cpu().numpy()[0] * 300)
            np.savetxt(save_path + str(int(names[0, 0, 0].item())) + 'face_predicted.txt',
                       predicted_B.cpu().numpy()[0] * 300)

            cd1 = chamfer_sqrt(predicted_A, A).item() * names[0, 0, 1]
            cd2 = chamfer_sqrt(predicted_B, B).item() * names[0, 0, 2]

            mped1 = MPEDloss(predicted_A, A).item() * names[0, 0, 1]
            mped2 = MPEDloss(predicted_B, B).item() * names[0, 0, 2]

            emd1 = emd(predicted_A, A).item() * names[0, 0, 1] / (20480.0)
            emd2 = emd(predicted_B, B).item() * names[0, 0, 2] / (20480.0)
            emd_all.append(emd1.item() + emd2.item())
            emd1_all.append(emd1.item())
            emd2_all.append(emd2.item())
            cd = cd1 + cd2
            cd_all.append(cd.item())
            cd1_all.append(cd1.item())
            cd2_all.append(cd2.item())

            mped = mped1 + mped2
            mped_all.append(mped.item())
            mped1_all.append(mped1.item())
            mped2_all.append(mped2.item())
            print(
                'name {:.1f}, spacingA: {:.1f}, spacingB: {:.1f}, mped: {:.3f}, mpedA: {:.3f}, mpedB: {:.3f}, cd: {:.3f}, cdA: {:.3f}, cdB: {:.3f}, emd: {:.3f}, emdA: {:.3f}, emdB: {:.3f}, hd: {:.3f}, hdA: {:.3f}, hdB: {:.3f},, jsd: {:.3f}, jsdA: {:.3f}, jsdB: {:.3f}'.format(
                    names[0, 0, 0].item(),
                    names[0, 0, 1].item(),
                    names[0, 0, 2].item(),
                    mped.item(),
                    mped1.item(),
                    mped2.item(),
                    cd.item(),
                    cd1.item(),
                    cd2.item(),
                    (emd1 + emd2).item(),
                    emd1.item(),
                    emd2.item(),
                    hd1 + hd2,
                    hd1,
                    hd2,
                    jsd1 + jsd2,
                    jsd1,
                    jsd2
                ))

    # Validate the current model
    # cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)

    print(
        'mped: {:.3f}+{:.3f}, mpedA: {:.3f}+{:.3f}, mpedB: {:.3f}+{:.3f},cd: {:.3f}+{:.3f}, cdA: {:.3f}+{:.3f}, cdB: {:.3f}+{:.3f}, emd: {:.3f}+{:.3f}, emdA: {:.3f}+{:.3f}, emdB: {:.3f}+{:.3f}, jsd: {:.3f}+{:.3f}, jsdA: {:.3f}+{:.3f}, jsdB: {:.3f}+{:.3f}, hd: {:.3f}+{:.3f}, hdA: {:.3f}+{:.3f}, hdB: {:.3f}+{:.3f}'.format(
            np.mean(mped_all),
            np.std(mped_all),
            np.mean(mped1_all),
            np.std(mped1_all),
            np.mean(mped2_all),
            np.std(mped2_all),
            np.mean(cd_all),
            np.std(cd_all),
            np.mean(cd1_all),
            np.std(cd1_all),
            np.mean(cd2_all),
            np.std(cd2_all),
            np.mean(emd_all),
            np.std(emd_all),
            np.mean(emd1_all),
            np.std(emd1_all),
            np.mean(emd2_all),
            np.std(emd2_all),
            np.mean(jsd_all),
            np.std(jsd_all),
            np.mean(jsd1_all),
            np.std(jsd1_all),
            np.mean(jsd2_all),
            np.std(jsd2_all),
            np.mean(hd_all),
            np.std(hd_all),
            np.mean(hd1_all),
            np.std(hd1_all),
            np.mean(hd2_all),
            np.std(hd2_all)
        ))
    print(
        'cd_mean: {:.6f}, cdA_mean: {:.6f}, cdB_mean: {:.6f},cd_std: {:.6f}, cdA_std: {:.6f}, cdB_std: {:.6f}, emd_mean: {:.6f}, emdA_mean: {:.6f}, emdB_mean: {:.6f}, emd_std: {:.6f}, emdA_std: {:.6f}, emdB_std: {:.6f}'.format(
            np.mean(cd_zong),
            np.mean(cd1_zong),
            np.mean(cd2_zong),
            np.std(cd_zong),
            np.std(cd1_zong),
            np.std(cd2_zong),
            np.mean(emd_zong),
            np.mean(emd1_zong),
            np.mean(emd2_zong),
            np.std(emd_zong),
            np.std(emd1_zong),
            np.std(emd2_zong)

        ))
    print(
        'mped: {:.3f}+{:.3f}, mpedA: {:.3f}+{:.3f}, mpedB: {:.3f}+{:.3f},cd: {:.3f}+{:.3f}, cdA: {:.3f}+{:.3f}, cdB: {:.3f}+{:.3f}, emd: {:.3f}+{:.3f}, emdA: {:.3f}+{:.3f}, emdB: {:.3f}+{:.3f}, hd: {:.3f}+{:.3f}, hdA: {:.3f}+{:.3f}, hdB: {:.3f}+{:.3f}'.format(
            np.mean(mped_local),
            np.std(mped_local),
            np.mean(mped1_local),
            np.std(mped1_local),
            np.mean(mped2_local),
            np.std(mped2_local),
            np.mean(cd_local),
            np.std(cd_local),
            np.mean(cd1_local),
            np.std(cd1_local),
            np.mean(cd2_local),
            np.std(cd2_local),
            np.mean(emd_local),
            np.std(emd_local),
            np.mean(emd1_local),
            np.std(emd1_local),
            np.mean(emd2_local),
            np.std(emd2_local),

            np.mean(hd_local),
            np.std(hd_local),
            np.mean(hd1_local),
            np.std(hd1_local),
            np.mean(hd2_local),
            np.std(hd2_local)
        ))


if __name__ == '__main__':
    # Check python version
    seed = 1
    test_net()
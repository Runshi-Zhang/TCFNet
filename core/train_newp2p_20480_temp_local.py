# -*- coding: utf-8 -*-
# @Author: XP
import random

import numpy as np
from pytorch3d.ops import sample_farthest_points
import MPED
import losses
import sys
import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter

from utils.average_meter import AverageMeter
from models.model_newp2p_20480 import P2PNet as Model
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from datasets import datasets
import glob
from torch.utils.data import DataLoader
from earth_movers_distance.emd import EarthMoverDistance
EMD = EarthMoverDistance()
def emd(pcs1, pcs2):
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)

chamfer_dist = chamfer_3DDist()

def random_subsample(pcd, n_points=20480):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    device = pcd.device
    batch_idx = torch.arange(b, dtype=torch.long, device=device).reshape((-1, 1)).repeat(1, n_points)
    idx = torch.cat([torch.randperm(n, dtype=torch.long, device=device)[:n_points].reshape((1, -1)) for i in range(b)], 0)
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
    if 0 <= epoch <= 30:
        return 1
    elif 30 < epoch <= 45:
        return 0.1
    elif 45 < epoch <= 60:
        return 0.01
    else:
        return 0.5

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"/logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train_net():
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    train_dir = r'/home/zrs/pyproject/pointcloudd/data/train/'
    val_dir = r'/home/zrs/pyproject/pointcloudd/data/test/'
    batch_size = 2
    max_epoch = 60
    train_set = datasets.FaceBoneDenseDataset(glob.glob(train_dir + '*.h5'))
    val_set = datasets.FaceBoneDenseInferDataset(sorted(glob.glob(val_dir + '*h5'), key=lambda name: int(name[42:-3])))

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader= DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join('./exp/output', '%s', datetime.datetime.now().isoformat())
    checkpoints_path = output_dir % 'checkpoints'
    logs_path = output_dir % 'logs'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logs_path, 'test'))
    sys.stdout = Logger(os.path.join(logs_path, 'train'))

    model = Model()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)


    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lr_lambda)


    init_epoch = 0
    best_metrics = float('inf')

    cdloss = losses.CDLoss()
    localloss = losses.AuxiLoss()


    bestMetrics = 100
    bestMetrics_A = 100
    bestMetrics_B = 100


    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, max_epoch + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        total_cd1 = 0
        total_cd2 = 0
        total_cd3 = 0
        total_pmd = 0
        id = 0
        batch_end_time = time()
        n_batches = len(train_data_loader) * 1
        for ii in range(1):
            for data in train_data_loader:
                model.train()
                data_time.update(time() - batch_end_time)
                data = [t.cuda() for t in data]
                A = data[0]  # (batch_size, 20480, 3)
                B = data[1]
                names = data[2]

                predicted_B_step, predicted_B, predicted_A_step, predicted_A = model(A, B)


                dataloss_A = cdloss(predicted_A_step, A) + cdloss(predicted_B_step, B) + 0.3 * cdloss(
                    torch.cat((B, predicted_A_step), dim=1),
                    torch.cat((A, predicted_B_step), dim=1))  # + emdA# + 0.2 * cdloss(predicted_A, predicted_A_step)
                dataloss_B = cdloss(predicted_B, B) + cdloss(predicted_A, A) + 0.3 * cdloss(
                    torch.cat((predicted_A, B), dim=1),
                    torch.cat((predicted_B, A), dim=1))  # + emdB# + 0.2 * cdloss(predicted_B_step, predicted_B)

                from pytorch3d.ops.knn import knn_points, knn_gather
                nnk = 16
                knn_1 = knn_points(A, predicted_A, K=nnk)
                knn_2 = knn_points(A, A, K=nnk)
                densityLoss = torch.mean(torch.abs(torch.sqrt(knn_1.dists) - torch.sqrt(knn_2.dists)))
                emd_loss = emd(torch.flatten(knn_gather(predicted_A, knn_1.idx), start_dim=0, end_dim=1),torch.flatten(knn_gather(A, knn_2.idx), start_dim=0, end_dim=1))

                knn_1 = knn_points(B, predicted_B, K=nnk)
                knn_2 = knn_points(B, B, K=nnk)
                emd_loss = emd(torch.flatten(knn_gather(predicted_B, knn_1.idx), start_dim=0, end_dim=1),torch.flatten(knn_gather(B, knn_2.idx), start_dim=0, end_dim=1)) + emd_loss
                regularloss = (torch.mean(
                    torch.abs(torch.sqrt(knn_1.dists) - torch.sqrt(knn_2.dists))) + densityLoss) + 0.1 * emd_loss# + cdlocal * 0.05


                regularloss = regularloss + 0.25 * localloss(A, B, predicted_A, predicted_B, '/home/zrs/pyproject/local/', names[0, 0, 0])


                dataloss = dataloss_A + dataloss_B


                loss = dataloss + regularloss



                optimizer.zero_grad()
                loss.backward()
                # torch.cuda.amp.GradScaler().scale(loss).backward()

                optimizer.step()
                cd1_item = dataloss.item()
                total_cd1 += cd1_item
                cd2_item = dataloss_A.item()
                total_cd2 += cd2_item
                cd3_item = dataloss_B.item()
                total_cd3 += cd3_item
                pmd_item = regularloss.item()
                total_pmd += pmd_item

                n_itr = (epoch_idx - 1) * n_batches + id
                id = id + 1
                train_writer.add_scalar('Loss/Batch/cd1', cd1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd2', cd2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd3', cd3_item, n_itr)
                train_writer.add_scalar('Loss/Batch/pmd', pmd_item, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()

                print(
                    'epoch: {}, Iter {} of {} loss {:.4f}, loss_A_B: {:.6f}, Reg: {:.6f}, shapeA: {:.6f}, shapeB: {:.6f}, cd: {:.6f}, cdA: {:.6f}, cdB: {:.6f}'.format(
                        epoch_idx, id, n_batches,
                        loss.item(),
                        dataloss.item(),
                        regularloss.item(),
                        dataloss_A.item(),
                        dataloss_B.item(),
                        bestMetrics,
                        bestMetrics_A,
                        bestMetrics_B
                    ))
                if(id % 190 == 0):
                   with torch.no_grad():
                       #id = 0
                       cd_all = 0
                       cd1_all = 0
                       cd2_all = 0
                       emd_all = 0
                       emd1_all = 0
                       emd2_all = 0

                       print('============================ TEST RESULTS ============================')
                       for data in val_data_loader:
                           model.eval()
                           data = [t.cuda() for t in data]
                           A = data[0]
                           B = data[1]
                           names = data[2]
                           # print(names)
                           predicted_B_step, predicted_B, predicted_A_step, predicted_A = model(A, B)

                           cd1 = chamfer_sqrt(predicted_A, A).item() * names[0, 0, 1]
                           cd2 = chamfer_sqrt(predicted_B, B).item() * names[0, 0, 2]

                           emd1 = cd1
                           emd2 = cd2
                           emd_all = emd1 + emd2 + emd_all
                           emd1_all = emd1_all + emd1
                           emd2_all = emd2_all + emd2

                           cd = cd1 + cd2
                           cd_all = cd_all + cd
                           cd1_all = cd1 + cd1_all
                           cd2_all = cd2 + cd2_all
                           print(
                               'name {:.4f}, spacingA: {:.6f}, spacingB: {:.6f}, cd: {:.6f}, cdA: {:.6f}, cdB: {:.6f}, emd: {:.6f}, emdA: {:.6f}, emdB: {:.6f}'.format(
                                   names[0, 0, 0].item(),
                                   names[0, 0, 1].item(),
                                   names[0, 0, 2].item(),
                                   cd.item(),
                                   cd1.item(),
                                   cd2.item(),
                                   (emd1 + emd2).item(),
                                   emd1.item(),
                                   emd2.item()
                               ))

                   # Validate the current model

                   cd_eval = cd_all / 25.0
                   cd1 = cd1_all / 25.0
                   cd2 = cd2_all / 25.0

                   emd_e = emd_all / 25.0
                   emd1 = emd1_all / 25.0
                   emd2 = emd2_all / 25.0

                   if (cd_eval < bestMetrics):
                       bestMetrics = cd_eval
                       bestMetrics_A = cd1
                       bestMetrics_B = cd2
                   print(
                       'cd: {:.6f}, cdA: {:.6f}, cdB: {:.6f},emd: {:.6f}, emdA: {:.6f}, emdB: {:.6f}, bestcd: {:.6f}, bestcdA: {:.6f}, bestcdB: {:.6f}'.format(
                           cd_eval.item(),
                           cd1.item(),
                           cd2.item(),
                           emd_e.item(),
                           emd1.item(),
                           emd2.item(),
                           bestMetrics,
                           bestMetrics_A,
                           bestMetrics_B
                       ))



                   # Save checkpoints
                   if epoch_idx % 5 == 0 or cd_eval < best_metrics:
                       file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                       output_path = os.path.join(checkpoints_path, file_name)
                       torch.save({
                           'epoch_index': epoch_idx,
                           'best_metrics': best_metrics,
                           'model': model.state_dict()
                       }, output_path)

                       logging.info('Saved checkpoint to %s ...' % output_path)
                       if cd_eval < best_metrics:
                           best_metrics = cd_eval

        time_end = time()
        alltime = (time_end - epoch_start_time) * (max_epoch - epoch_idx)

        timeresult = str(datetime.timedelta(seconds=alltime))
        print('training time:' + timeresult)
        avg_cd1 = total_cd1 / n_batches
        avg_cd2 = total_cd2 / n_batches
        avg_cd3 = total_cd3 / n_batches
        avg_pmd = total_pmd / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd3', avg_cd3, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/pmd', avg_pmd, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, max_epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd1, avg_cd2, avg_cd3, avg_pmd]]))
        '''
                Validation
        '''




    train_writer.close()
    val_writer.close()

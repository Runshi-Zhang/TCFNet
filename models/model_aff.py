import numpy as np
import torch
import torch.nn as nn

class GeoConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, norm='bn'):
        super(GeoConv, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_channels) for _ in range(6)])
        self.norm1 = nn.BatchNorm1d(hidden_channels)
        self.acti1 = nn.ReLU(inplace=True)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.acti2 = nn.ReLU(inplace=True)

    def forward(self, x, B, n, sid_euc, tid_euc, bid, p_cos, p_d):
        # x[B*N, C] p[B*N, 3] sid/tid[B*n*k], r = 0.15
        k = int(len(sid_euc) / B / n)
        dev = x.device

        euc_i, euc_j = x[tid_euc], x[sid_euc]  # [B*n*k, C]
        edge = euc_j - euc_i

        edge = torch.stack([lin(edge) for lin in self.lins])  # [bases, B*n*k, C]
        edge = torch.stack([edge[bid[:, i], range(B * n * k)] for i in range(3)])  # [3, B*n*k, C]
        edge = edge.view(3, B, n, k, -1)
        edge = edge * p_cos  # [3, B, n, k, C]
        edge = edge.sum(dim=0)  # [B, n, k, C]

        edge = edge * p_d / p_d.sum(dim=2, keepdim=True)  # [B, n, k, C]
        y = edge.sum(dim=2).transpose(1, -1)  # [B, C, n]
        y = self.acti1(self.norm1(y)).transpose(1, -1)  # [B, n, C]
        x = self.lin1(x[tid_euc[::k]]).view(B, n, -1)  # [B, n, C]
        y = x + self.lin2(y)  # [B, n, C]
        y = y.transpose(1, -1)  # [B, C, n]
        y = self.acti2(self.norm2(y))

        return y

class PointPlus(nn.Module):  # PointNet++
    def __init__(self, in_channels, out_channels, first_layer=False):
        super(PointPlus, self).__init__()
        self.first_layer = first_layer
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x, B, n, sid_euc, tid_euc):
        # x[B*N, C] sid/tid[B*n*k]
        #sid_euc, tid_euc = id_euc
        k = int(sid_euc.size(0) / B / n)

        if self.first_layer:
            x, norm = x[:, :3], x[:, 3:]
            x_i, x_j = x[tid_euc], x[sid_euc] # [B*n*k, C]
            norm_j = norm[sid_euc] # [B*n*k, C]
            edge = torch.cat([x_j - x_i, norm_j], dim=-1) # [B*n*k, C]
        else:
            x_i, x_j = x[sid_euc], x[tid_euc] # [B*n*k, C]
            edge = x_j - x_i
        edge = edge.view(B, n, k, -1) # [B, n, k, C]
        edge = self.fc1(edge.transpose(1,-1)) # [B, n, k, C]
        y  = edge.max(-2)[0].transpose(1,2) # [B, n, C]
        y = y.contiguous().view(B*n, -1) # [B*n, C]

        return y


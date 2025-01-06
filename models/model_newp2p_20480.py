
import torch
import torch.nn as nn

from models.utils import Conv1d

from models.model_v3 import PointTransformerV3

from models.model_aff import GeoConv,  PointPlus



def knn_gnn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)

    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices
    B, _, N = x.size()
    n = N
    dev = x.device
    #x = x.to(torch.float16)
    if not torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, sid = KNN(k=k, transpose_mode=True)(ref, query)
        idx = sid.clone()

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        sid = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        idx = sid.clone()
    sid += torch.arange(B, device=dev).view(B, 1, 1) * N
    sid = sid.reshape(-1) # [B*n*k]
    tid = torch.arange(B * N, device=dev) # [B*n]
    tid = tid.view(-1, 1).repeat(1, k).view(-1) # [B*n*k]
    return idx, sid, tid, pairwise_distance # [B*n*k]

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError




def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points

class Unit(nn.Module):
    def __init__(self, in_channel=256):
        super(Unit, self).__init__()

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

    def forward(self, cur_x, prev_s):
        """
        Args:
            cur_x: Tensor, (B, in_channel, N)
            prev_s: Tensor, (B, in_channel, N)

        Returns:
            h: Tensor, (B, in_channel, N)
            h: Tensor, (B, in_channel, N)
        """

        z = self.conv_z(torch.cat([cur_x, prev_s], 1))
        r = self.conv_r(torch.cat([cur_x, prev_s], 1))
        h_hat = self.conv_h(torch.cat([cur_x, r * prev_s], 1))
        h = (1 - z) * cur_x + z * h_hat
        return h
class PPNet_GNN_brach_noise_unit(nn.Module):
    def __init__(self, if_noise=True, noise_dim=3, noise_stdv=1e-2):
        super(PPNet_GNN_brach_noise_unit, self).__init__()
        self.feat_conv = nn.Sequential(
            nn.Conv1d(640, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 64, 1),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(80, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 3, 1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(15, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.GNN1 = GeoConv(in_channels=64, hidden_channels=64, out_channels=128)
        self.GNN2 = GeoConv(in_channels=128, hidden_channels=64, out_channels=256)
        self.unit = Unit(in_channel=64)
        self.pointplus1 = PointPlus(in_channels=64, out_channels=128)
        self.pointplus2 = PointPlus(in_channels=128, out_channels=256)

    def forward(self, point_cloud, feat):
        """
        Args:
            point_cloud: Tensor, (B, 20480, 3)
        """
        # b, npoint, _ = point_cloud.shape
        device = point_cloud.device
        b, n, _ = point_cloud.shape
        point_cloud = point_cloud.transpose(1, 2)

        knn_idx, sid, tid, distance = knn_gnn(point_cloud, k=8)#8
        knn_x = index_points(point_cloud.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        noise_points = torch.normal(mean=0, std=torch.ones((b, 3, n),
                                                           device=device) * 1e-2)

        l0_points_raw = torch.cat([point_cloud, covariances, noise_points], dim=1)  # (B, 12, N)
        l0_points_raw = self.conv1(l0_points_raw)  # (B, 64, N)
        l0_points_raw = torch.flatten(l0_points_raw.transpose(1, 2), start_dim=0, end_dim=1)
        l0_points1 = self.pointplus1(l0_points_raw, b, n, sid, tid)
        l0_points1 = self.pointplus2(l0_points1, b, n, sid, tid)
        l0_points1 = l0_points1.view(b, n, -1).transpose(1,2)

        max_valid_neighbors = 8#8
        dis, sid = distance.topk(max_valid_neighbors, largest=False)  # [B, n, max_valid_neighbors]#这里计算的是最远距离
        sid += torch.arange(b, device=device, dtype=sid.dtype).view(b, 1, 1) * n
        sid = sid.view(-1)
        fpsi = torch.arange(b * n, device=device)
        tid = fpsi.view(-1, 1).repeat(1, max_valid_neighbors).view(-1)
        p = torch.flatten(point_cloud.transpose(1, 2), start_dim=0, end_dim=1)
        p_diff = p[sid] - p[tid]  # [B*n*k, 3]
        p_dis = p_diff.norm(dim=-1, keepdim=True).clamp(min=1e-16)  # [B*n*k, 1]
        p_cos = (p_diff / p_dis).cos() ** 2  # [B*n*k, 3]
        p_cos = p_cos.transpose(0, 1).reshape(-1, b, n, max_valid_neighbors, 1)  # [3, B, n, k, 1]
        bid = (p_diff > 0).long()  # [B*n*k, 3]
        bid += torch.tensor([0, 2, 4], device=device, dtype=torch.long).view(1, 3)
        p_dis = p_dis.view(b, n, max_valid_neighbors, 1)
        p_r = p_dis.max(dim=2, keepdim=True)[0] * 1.1  # [B, n, 1, 1]
        p_d = (p_r - p_dis) ** 2  # [B, n, k, 1]

        l0_points3 = self.GNN1(l0_points_raw,
                                b, n, sid, tid, bid, p_cos, p_d)
        l0_points4 = self.conv2(l0_points3)
        l0_points4 = self.GNN2(torch.flatten(l0_points4.transpose(1, 2), start_dim=0, end_dim=1),
                                b, n, sid, tid, bid, p_cos, p_d)
        noise_points = torch.normal(mean=0, std=torch.ones((b, 16, n),
                                                           device=device) * 1e-2)
        l0_points = self.feat_conv(torch.cat((l0_points3, l0_points4, l0_points1), dim=1))
        feat2 = self.unit(l0_points, feat)
        l0_points = self.seg_head(torch.cat((feat2, noise_points), dim=1))
        return torch.add(l0_points, point_cloud), l0_points







class PPNet_after_cov(nn.Module):
    def __init__(self, if_noise=True, noise_dim=3, noise_stdv=1e-2):
        super(PPNet_after_cov, self).__init__()
        self.if_noise = if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv

        self.seg_head = (
            nn.Sequential(
                nn.Linear(80, 64),
                PointBatchNorm(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, 64),
                PointBatchNorm(64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, 3),
            )
        )

        self.displace = PointTransformerV3(in_channels=6)
        self.stepconv = PPNet_GNN_brach_noise_unit()


    def forward(self, point_cloud):
        """
        Args:
            point_cloud: Tensor, (B, 20480, 3)
        """

        device = point_cloud.device
        l0_xyz = point_cloud.clone()
        b, n,_ = point_cloud[:, :, 0:3].shape
        point_cloud = point_cloud.transpose(1, 2)

        noise_points = torch.normal(mean=0, std=torch.ones((b, n, (self.noise_dim if self.if_noise else 0)), device=device) * self.noise_stdv)
        l0_points = torch.cat([point_cloud.transpose(1,2), noise_points], 2)
        l0_points = torch.flatten(l0_points, start_dim=0, end_dim=1)
        l0_xyz = torch.flatten(l0_xyz, start_dim=0, end_dim=1)

        if (b == 2):
            data_dict = {'coord': l0_xyz,
                         #'offset': torch.tensor([20480, 20480 * 2, 20480 * 3, 20480 * 4], device=device), "grid_size": 0.01,
                         'offset': torch.tensor([20480, 20480 * 2], device=device), "grid_size": 0.01,
                         'feat': l0_points}
            # feat = self.displace(data_dict)

        if (b == 1):
            data_dict = {'coord': l0_xyz,
                         'offset': torch.tensor([20480], device=device), "grid_size": 0.01,
                         'feat': l0_points}
            # feat = self.displace(data_dict)
        data_dict = self.displace(data_dict)['feat']
        noise_points = torch.normal(mean=0, std=torch.ones((b * n, 16),
                                                           device=device) * 1e-2)
        feat = self.seg_head(torch.cat([data_dict, noise_points], -1))
        feat = feat.view(-1, 20480, 3)
        displacement = feat.permute(0, 2, 1).contiguous()
        point_cloud1 = torch.add(displacement, point_cloud[:, 0:3, :])
        point_cloud, displacement1 = self.stepconv(point_cloud1.permute(0, 2, 1).contiguous(),
                                                   data_dict.view(-1, 20480, 64).permute(0, 2, 1).contiguous())

        return point_cloud1, point_cloud

class P2PNet(nn.Module):
    def __init__(self):
        super(P2PNet, self).__init__()
        self.pp_moduleA = PPNet_after_cov()
        self.pp_moduleB = PPNet_after_cov()


    def forward(self, point_cloud_A, point_cloud_B):
        point_cloud_A_step, point_cloud_A = self.pp_moduleA(point_cloud_A)
        point_cloud_B_step, point_cloud_B = self.pp_moduleB(point_cloud_B)
        return point_cloud_A_step.permute(0, 2, 1).contiguous(), point_cloud_A.permute(0, 2, 1).contiguous(), point_cloud_B_step.permute(0, 2, 1).contiguous(), point_cloud_B.permute(0, 2, 1).contiguous()
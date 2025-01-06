'''
import h5py

# 替换为你的HDF5文件路径
import numpy as np

file_path = r'/home/zrs/pointcloud/face_transform_bone/test.h5'
train_path = r'/home/zrs/pointcloud/face_transform_bone/data/test/'

hdfFile = h5py.File(file_path, 'r')
data = hdfFile['skeleton'][:]
data1 = hdfFile['surface'][:]
data2 = hdfFile['names'][:]
for i in range(data.shape[0]):
    with h5py.File(train_path + str(i+1) + '.h5', 'w') as f:
        f.create_dataset('skeleton', data=data[i,...])
        f.create_dataset('surface', data=data1[i,...])
        f.create_dataset('names', data=data2[20*i,...])
        f.close()




print(np.max(data[:,:,0]))
print(np.min(data[:,:,0]))

print(np.max(data[:,:,1]))
print(np.min(data[:,:,1]))

print(np.max(data[:,:,2]))
print(np.min(data[:,:,2]))


print(np.max(data[:,:,:]))

data = np.loadtxt('/home/zrs/result_con.txt')
print(np.std(np.mean(data,axis=1)))
'''
import os

import h5py

# 替换为你的HDF5文件路径
import numpy as np
import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    #xyz = xyz.transpose(0, 2, 1)
    B, N, C = xyz.shape

    centroids = np.zeros((B, npoint))  # 采样点矩阵（B, npoint）
    distance = np.ones((B, N)) * 1e10  # 采样点到所有点距离（B, N）

    batch_indices = np.arange(B)  # batch_size 数组

    barycenter = np.sum((xyz), 1)  # 计算重心坐标 及 距离重心最远的点
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, C)  # numpy中的reshape相当于torch中的view

    dist = np.sum((xyz - barycenter) ** 2, -1)
    farthest = np.argmax(dist, 1)  # 将距离重心最远的点作为第一个点，这里跟torch.max不一样

    for i in range(npoint):
        #print("-------------------------------------------------------")
        #print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, C)  # 取出这个最远点的xyz坐标
        dist = np.sum((xyz - centroid) ** 2, -1)  # 计算点集中的所有点到这个最远点的欧式距离，-1消掉了xyz那个维度
        #print("dist    : ", dist)
        mask = dist < distance
        #print("mask %i : %s" % (i, mask))
        distance[mask] = dist[mask]  # 更新distance，记录样本中每个点距离所有已出现的采样点（已采样集合中的点）的最小距离
        #print("distance: ", distance)

        farthest = np.argmax(distance, -1)  # 返回最远点索引

    return centroids
from pytorch3d.ops import sample_farthest_points
train_dir = r'/home/zrs/result/train_after_25000/'
pathlist = os.listdir(train_dir)
for i in pathlist:
    path = train_dir + i
    hdfFile = h5py.File(path, 'r')
    x = hdfFile['skeleton'][:][:, 0:3][None,...]
    y = hdfFile['surface'][:][:, 0:3]#[None,...]
    name = hdfFile['names'][:]
    #np.savetxt('bone.txt', x[0,...] * 300)
    #np.savetxt('face.txt', y[0, ...] * 300)
    x = torch.from_numpy(x).cuda()
    #y = torch.from_numpy(y).cuda()
    x_temp = sample_farthest_points(x, K=20480)
    #y_temp = sample_farthest_points(y, K=20480)
    x_temp = x_temp[0].cpu().numpy()[0]
    #y_temp = y_temp[0].cpu().numpy()[0]
    with h5py.File('/home/zrs/result/train_after_25000_sample/' + i, 'w') as f:
        f.create_dataset('skeleton', data=x_temp)
        f.create_dataset('surface', data=y)
        f.create_dataset('names', data=name)
        f.close()
    '''
    for t in range(5):
        #x_temp = farthest_point_sample(x, 20480)
        #y_temp = farthest_point_sample(y, 20480)
        x_temp = sample_farthest_points(x, K=20480)
        y_temp = sample_farthest_points(y, K=20480)
        x_temp = x_temp[0].cpu().numpy()[0] * 300
        y_temp = y_temp[0].cpu().numpy()[0] * 300
        np.savetxt('bone_temp.txt', x_temp)
        np.savetxt('face_temp.txt', y_temp)


        with h5py.File('/home/zrs/pyproject/pointcloudd/data/train_5/' + str(t) + '_' + i, 'w') as f:
            f.create_dataset('skeleton', data=x_temp)
            f.create_dataset('surface', data=y_temp)
            f.create_dataset('names', data=name)
            f.close()
            '''




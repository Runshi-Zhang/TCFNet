import os, glob
import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import numpy as np
import h5py
import datasets.data_transformer

def transformer(x, y):
    #x, y = datasets.data_transformer.centershift(x, y)
    #if random.random() < 0.2:
        #x, y = datasets.data_transformer.random_point_dropout(x, y, 0.9)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate(x, y, [-1 / 64, 1 / 64], 'z')
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate(x, y, [-1 / 64, 1 / 64], 'x')
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate(x, y, [-1 / 64, 1 / 64], 'y')
    if random.random() < 0.2:
        x, y = datasets.data_transformer.random_scale_point_cloud(x, y, 0.99, 1.11)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip(x, y, 0)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip(x, y, 1)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip(x, y, 2)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.jitter_point_cloud(x, y, 0.0005, 0.002)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.shuffle_points(x, y)
    #x, y = datasets.data_transformer.centershift(x, y)
    x = datasets.data_transformer.normalize_data(x)
    y = datasets.data_transformer.normalize_data(y)
    return x, y

def transformer_normal(x, y):
    if random.random() < 0.2:
        x[:, 0:3], y[:, 0:3] = datasets.data_transformer.random_point_dropout(x[:, 0:3], y[:, 0:3], 0.9)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate_normal(x, y, [-1 / 64, 1 / 64], 'z')
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate_normal(x, y, [-1 / 64, 1 / 64], 'x')
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomrotate_normal(x, y, [-1 / 64, 1 / 64], 'y')
    if random.random() < 0.2:
        x[:, 0:3], y[:, 0:3] = datasets.data_transformer.random_scale_point_cloud(x[:, 0:3], y[:, 0:3], 0.9, 1.1)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip_normal(x, y, 0)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip_normal(x, y, 1)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.randomflip_normal(x, y, 2)
    if random.random() < 0.2:
        x[:, 0:3], y[:, 0:3] = datasets.data_transformer.jitter_point_cloud(x[:, 0:3], y[:, 0:3], 0.005, 0.02)
    if random.random() < 0.2:
        x, y = datasets.data_transformer.shuffle_points(x, y)
    #x, y = datasets.data_transformer.centershift(x, y)
    x[:, 0:3] = datasets.data_transformer.normalize_data(x[:, 0:3])
    y[:, 0:3] = datasets.data_transformer.normalize_data(y[:, 0:3])
    return x, y





class FaceBoneDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:]#(4096,3)
        y = hdfFile['surface'][:]#(4096,3)
        data2 = hdfFile['names'][:]#(3)
        '''
        pathint = int(path[37:-3])
        pathint_random = random.randint(0, 99)
        pathint_y = (pathint - 1) // 100 * 100 + pathint_random
        if (pathint_y == 0):
            pathint_y = random.randint(1, 99)
        path_y = path.replace(str(pathint)+'.h5', str(pathint_y)+'.h5')
        path_y = path_y[:-3] + '.h5'
        hdfFile = h5py.File(path_y, 'r')
        y = hdfFile['surface'][:]  # (4096,3)
        data2 = hdfFile['names'][:]
        '''

        #x, y = transformer(x, y)

        #x = np.concatenate((x, x*data2[1]), axis=1)
        #y = np.concatenate((y, y*data2[2]), axis=1)
        x = np.ascontiguousarray(x.astype(np.float32))  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y.astype(np.float32))

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class FaceBoneInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:]  # (4096,3)
        y = hdfFile['surface'][:]  # (4096,3)
        z = hdfFile['names'][:]  # (3)

        #x = np.concatenate((x, x * z[1]), axis=1)
        #y = np.concatenate((y, y * z[2]), axis=1)

        #x, y = x[None, ...], y[None, ...]

        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(z)

        x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        return x, y, z

    def __len__(self):
        return len(self.paths)

class FaceBoneDenseDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:][:, 0:3]#(4096,3)
        y = hdfFile['surface'][:][:, 0:3]#(4096,3)
        data2 = hdfFile['names'][:]#(3)

        '''
        path_index = path.find('_')
        path_y = path[0:path_index]
        path_y = path_y +  '_' + str(random.randint(1, 20)) + '.h5'
        hdfFile_y = h5py.File(path_y, 'r')
        y = hdfFile_y['surface'][:][:, 0:3]
        '''


        '''
        pathint = int(path[40:-3])
        pathint_random = random.randint(0, 4)
        pathint_y = (pathint - 1) // 5 * 5 + pathint_random
        if (pathint_y == 0):
            pathint_y = random.randint(1, 4)
        path_y = path.replace(str(pathint) + '.h5', str(pathint_y) + '.h5')
        hdfFile = h5py.File(path_y, 'r')
        y = hdfFile['surface'][:][:, 0:3] # (4096,3)
        data2 = hdfFile['names'][:]

        x, y = datasets.data_transformer.shuffle_points(x, y)
        '''
        #x, x = datasets.data_transformer.shuffle_points(x, x)
        #y, y = datasets.data_transformer.shuffle_points(y, y)

        x = np.ascontiguousarray(x.astype(np.float32))  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y.astype(np.float32))
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        points_perm = torch.randperm(x.shape[0])
        x = x[points_perm, :]
        points_perm = torch.randperm(y.shape[0])
        y = y[points_perm, :]
        '''
        l1 = x.shape[0]
        x_temp = torch.zeros([25000, 3], dtype = torch.float32)
        x_temp[:l1,:] = x[:,:]

        l2 = y.shape[0]
        y_temp = torch.zeros([25000, 3], dtype = torch.float32)
        y_temp[:l2,:] = y[:,:]
        '''
        #l1 = l2 = 20480
        #x_temp = x[:l1, :]
        #y_temp = y[:l2, :]

        return x, y, data2#x_temp, y_temp, l1, l2

        #return x, y

    def __len__(self):
        return len(self.paths)


class FaceBoneDenseInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:][:, 0:3]  # (4096,3)
        y = hdfFile['surface'][:][:, 0:3]  # (4096,3)
        z = hdfFile['names'][:]  # (3)

        #x, y = x[None, ...], y[None, ...]

        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(z)

        x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        return x, y, z

    def __len__(self):
        return len(self.paths)

class FaceBoneNormalDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:]#(4096,3)
        y = hdfFile['surface'][:]#(4096,3)
        data1 = hdfFile['names'][:]#(3)

        '''
        pathint = int(path[40:-3])
        pathint_random = random.randint(0,4)
        pathint_y = (pathint - 1) // 5 * 5 + pathint_random
        if(pathint_y == 0):
            pathint_y = random.randint(1,4)
        path_y = path.replace(str(pathint) + '.h5', str(pathint_y) + '.h5')

        hdfFile = h5py.File(path_y, 'r')
        y = hdfFile['surface'][:]  # (4096,3)
        data2 = hdfFile['names'][:]
        '''


        #x, y = transformer(x, y)

        #x, y, data2 = pc_normalize(np.concatenate((x,y), axis=0))

        #x, y = x[None, ...], y[None, ...]

        #x, x_seg = self.transforms([x, x_seg])
        #y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x.astype(np.float32))  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y.astype(np.float32))

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

class FaceBoneNormalInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        #self.transforms = transforms


    def __getitem__(self, index):
        path = self.paths[index]
        hdfFile = h5py.File(path, 'r')
        x = hdfFile['skeleton'][:]  # (4096,3)
        y = hdfFile['surface'][:]  # (4096,3)
        z = hdfFile['names'][:]  # (3)

       # x, y, data2 = pc_normalize(np.concatenate((x, y), axis=0))

        #z = z * data2
        #x, y = x[None, ...], y[None, ...]

        # x, x_seg = self.transforms([x, x_seg])
        # y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(z)

        x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        return x, y, z

    def __len__(self):
        return len(self.paths)

import os
import random
import h5py
train_dir = r'/home/zrs/result/train_after_25000_sample/'
pathlist = os.listdir(train_dir)
revome_list = []
for i in pathlist:
    revome_list.append(i[:i.find("_")])
revome_set_list = list(set(revome_list))
random.shuffle(revome_set_list)

for i in range(len(revome_set_list)):
    if(i < len(revome_set_list) * 0.8):
        path_number = revome_set_list[i]
        for j in range(20):
            hdfFile = h5py.File(train_dir + path_number + '_' + str(j + 1) + '.h5', 'r')
            x = hdfFile['skeleton'][:][:, 0:3]
            y = hdfFile['surface'][:][:, 0:3]
            name = hdfFile['names'][:]
            with h5py.File('/media/zrs/000BD980000DDCC6/zrs/bone_face_data/20480/data_2/train/' + path_number + '_' + str(j + 1) + '.h5', 'w') as f:
                f.create_dataset('skeleton', data=x)
                f.create_dataset('surface', data=y)
                f.create_dataset('names', data=name)
                f.close()
    else:
        path_number = revome_set_list[i]
        hdfFile = h5py.File(train_dir + path_number  + '_1.h5', 'r')
        x = hdfFile['skeleton'][:][:, 0:3]
        y = hdfFile['surface'][:][:, 0:3]
        name = hdfFile['names'][:]
        with h5py.File('/media/zrs/000BD980000DDCC6/zrs/bone_face_data/20480/data_2/test/' + path_number + '_1.h5', 'w') as f:
            f.create_dataset('skeleton', data=x)
            f.create_dataset('surface', data=y)
            f.create_dataset('names', data=name)
            f.close()



a = 1
import pickle
import os
from os.path import exists, join, isdir
import numpy as np
from utils.ply import read_ply, write_ply
from utils.gamma_transform import gamma_transform
from utils.log_transform import log_transform
train_points_path = '/media/zt/D/3060/KPConv-2-master/Data/DFC2019/DFC2019_DATA/train-track4/Track4'
train_points_labels = '/media/zt/D/3060/KPConv-2-master/Data/DFC2019/DFC2019_DATA/train-track4-truth/Track4-Truth'
points = np.sort([f for f in os.listdir(train_points_path) if f[-4:] == '.txt'])
labels = np.sort([f for f in os.listdir(train_points_labels) if f[-4:] == '.txt'])

for each_file in points:
    print('\n Loading...', os.path.join(train_points_path, each_file))
    data_points = np.loadtxt(os.path.join(train_points_path, each_file), delimiter=',')
    data_labels = np.loadtxt(os.path.join(train_points_labels, each_file))
    x = data_points[:, 0]
    y = data_points[:, 1]
    z = data_points[:, 2]
    red = gamma_transform(data_points[:, 3], 0.5)
    green = gamma_transform(data_points[:, 3], 1)
    blue = gamma_transform(data_points[:, 3], 1.5)
    # red = gamma_transform(data_points[:, 3], 0.5)
    # green = log_transform(data_points[:, 3],1.5)
    # blue = data_points[:, 3]
    # reshape
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    z = z.reshape(len(z), 1)
    red = red.reshape(len(red), 1)
    green = green.reshape(len(green), 1)
    blue = blue.reshape(len(blue), 1)
    # astype
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    red = red.astype(np.uint8)
    green = green.astype(np.uint8)
    blue = blue.astype(np.uint8)

    data_labels = data_labels.reshape(len(data_labels), 1)
    data_labels = data_labels.astype(np.int32)
    new_data_label = np.zeros((data_labels.shape[0], 1))
    for i in range(data_labels.shape[0]):
        if int(data_labels[i]) == 2:
            new_data_label[i, :] = 0
        elif int(data_labels[i]) == 5:
            new_data_label[i, :] = 1
        elif int(data_labels[i]) == 6:
            new_data_label[i, :] = 2
        elif int(data_labels[i]) == 9:
            new_data_label[i, :] = 3
        elif int(data_labels[i]) == 17:
            new_data_label[i, :] = 4
        else:
            new_data_label[i, :] = -1


    data_fusion = np.hstack((x, y, z, red, green, blue, new_data_label))
    data_fusion=data_fusion[new_data_label[:,0]!=-1,:]
    # np.savetxt(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\DFC2019\\' + each_file[:-4]+'.txt',data_fusion)
    write_ply('/media/zt/D/PycharmProjects/KPConv-PyTorch-master/Data/DFC2019_PLY/original_ply/' + each_file[:-4] + '.ply',
              [data_fusion],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
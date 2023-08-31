
# import open3d as o3d
import numpy as np
import glob
import os
import posixpath
import time
from os import listdir, makedirs

from os import listdir
from os.path import exists, join, isdir
from utils.ply import read_ply,write_ply

from plyfile import PlyData
def gamma_transform(img,gamma):
    """
    gamma transform 2d grayscale image, convert uint image to float
    param: input img: input grayscale image
    param: input c: scale of the transform
    param: input gamma: gamma value of the transoform
    """
    img = img.astype(float)  # 先要把图像转换成为float，不然结果点不太相同
    img=img/(img.max())
    epsilon = 1e-5  # 非常小的值以防出现除0的情况

    img_dst = np.power(img + epsilon, gamma)*255
    img_dst=img_dst.astype(np.uint8)

    return img_dst

cloud_names = ['section_1',
                'section_2',
                'section_3',
                'section_4'
                #'Vaihingen3D_train',
                #'Vaihingen3D_test'
            ]
# t_dir='D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\LASDU_RAW\lasdu_1'
# original_name = np.sort([f for f in os.listdir(t_dir) if f[-4:]=='.txt'])
root_dir='/media/zt/D/PycharmProjects/KPConv-PyTorch-master/Data/LASDU'#'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D'
train_path='original_ply_1.5'

t0 = time.time()
for cloud_name in cloud_names:
    ply_path = join(root_dir, train_path)
    if not exists(ply_path):
        makedirs(ply_path)
    cloud_file=join(ply_path,cloud_name+'.ply')
    cloud_floder = join(root_dir, cloud_name+'.ply')
    original_ply=read_ply(cloud_floder)

    cloud_x=original_ply['x']
    cloud_y=original_ply['y']
    cloud_z=original_ply['z']
    #ADD FEATURE
    red=gamma_transform(original_ply['intensity'],1.5)
    green=gamma_transform(original_ply['intensity'],1.5)
    blue=gamma_transform(original_ply['intensity'],1.5)
    # red = original_ply['intensity']
    # green = original_ply['intensity']
    # blue = original_ply['intensity']

    label=original_ply['label']

    label=label.astype(np.int32)
    cloud_points=np.vstack((cloud_x,cloud_y,cloud_z)).T
    color=np.vstack((red,green,blue)).T
    color=color.astype(np.uint8)
    # print(cloud_points[:,3])
    write_ply(cloud_file,[cloud_points,color,label],
              ['x','y','z','red','green','blue','class'])

print('Done in {:.1f}s'.format(time.time() - t0))

# b=o3d.io.read_point_cloud(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D\original_ply\Vaihingen3D_test.ply')
# o3d.visualization.draw_geometries([b])
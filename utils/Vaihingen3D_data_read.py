from utils.ply import *
import open3d as o3d
import numpy as np
# OS functions
import glob
import os
import posixpath
import time
from os import listdir, makedirs

from os import listdir
from os.path import exists, join, isdir



from utils.ply import *


from plyfile import PlyData
cloud_names = ['Vaihingen3D_train',
                'Vaihingen3D_test',]
root_dir='D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D'
train_path='original_ply'
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
    red=original_ply['red']
    green=cloud_z.max()-cloud_z
    blue=(cloud_z/cloud_z.max())*100

    label=original_ply['scalar_Scalar_field']

    cloud_x=cloud_x-(cloud_x.min())
    cloud_y=cloud_y-(cloud_y.min())
    cloud_z=cloud_z-(cloud_z.min())
    cloud_x=cloud_x.reshape(len(cloud_x),1)
    cloud_y=cloud_y.reshape(len(cloud_y),1)
    cloud_z=cloud_z.reshape(len(cloud_z),1)
    label=label.reshape(len(label),1)
    cloud_x=cloud_x.astype(np.float32)
    cloud_y=cloud_y.astype(np.float32)
    cloud_z=cloud_z.astype(np.float32)

    # red=red-(red.min())
    # green=green-(green.min())
    # blue=blue-(blue.min())
    red=red.reshape(len(red),1)
    green=green.reshape(len(red),1)
    blue=blue.reshape(len(red),1)
    red=red.astype(np.uint8)
    green=green.astype(np.uint8)
    blue=blue.astype(np.uint8)



    label=label.astype(np.int32)
    cloud_points=np.hstack((cloud_x,cloud_y,cloud_z,red,green,blue))
    # print(cloud_points[:,3])
    write_ply(cloud_file,[cloud_points,label],
              ['x','y','z','red','green','blue','class'])

print('Done in {:.1f}s'.format(time.time() - t0))

b=o3d.io.read_point_cloud(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D\original_ply\Vaihingen3D_test.ply')
o3d.visualization.draw_geometries([b])
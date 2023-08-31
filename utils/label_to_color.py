
import numpy as np
from utils.ply import *
import colorsys, random, os, sys

def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


original_ply=read_ply(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D\original_ply\Vaihingen3D_test.ply')
cloud_x=original_ply['x']
cloud_y=original_ply['y']
cloud_z=original_ply['z']
#ADD FEATURE

label=original_ply['class']

cloud_x=cloud_x.reshape(len(cloud_x),1)
cloud_y=cloud_y.reshape(len(cloud_y),1)
cloud_z=cloud_z.reshape(len(cloud_z),1)
label=label.reshape(len(label),1)
cloud_x=cloud_x.astype(np.float32)
cloud_y=cloud_y.astype(np.float32)
cloud_z=cloud_z.astype(np.float32)

label=label.astype(np.int32)
cloud_points=np.hstack((cloud_x,cloud_y,cloud_z,label))

data1 = cloud_points[:,:3]
label1 = cloud_points[:,3]

# rgb_codes = [[200, 90, 0],
#             [255, 0, 0],
#             [255, 0, 255],
#             [0, 220, 0],
#             [0, 200, 255]]

# rgb_codes =[[0, 0, 255],
#             [0, 255, 0],
#             [192, 192, 192],
#             [255, 97, 3],
#             [255, 0, 255],
#             [255, 0, 0],
#             [255, 255, 0],
#             [189,252,201],
#             [46,139,87]]
rgb_codes = random_colors(6)
color = np.zeros((label1.shape[0], 3))


for i in range(label1.shape[0]):

    color[i,:] = [code for code in rgb_codes[int(label1[i])]]
    # color = color.astype(np.uint8)
# color = color.astype(np.uint8)
print(color.shape)
write_ply(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\Vaihingen3D\original_ply\Vaihingen3D_test_Colored.ply',[data1,color,label1],
          ['x','y','z','red','green','blue','class'])


#Error Maping
# for i in range(label1.shape[0]):
#     if int(label1[i])==0:
#         color[i,:] = [0, 0, 255]
#     else:
#         color[i, :] = [255, 0, 0]
# print(color.shape)
# write_ply(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\5_Davos_16_preds.ply', [data1, color, label1],
#           ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.blocks import closest_pool
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

def log_transform(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output

trian_data=np.loadtxt('/home/zt/TAO/KPConv-2-master/Data/Vaihingen_pts/Vaihingen3D_Traininig.pts')
test_data=np.loadtxt('/home/zt/TAO/KPConv-2-master/Data/Vaihingen_pts/Vaihingen3D_Testing.pts')
#add futures

x_train=trian_data[:,0]
y_train=trian_data[:,1]
z_train=trian_data[:,2]

# red_train=gamma_transform(trian_data[:,3],1)
# green_train=gamma_transform(trian_data[:,3],1)
# blue_train=gamma_transform(trian_data[:,3],1)
red_train=trian_data[:,3]
green_train=trian_data[:,3]
blue_train=trian_data[:,3]

data_train=np.vstack((x_train,y_train,z_train)).T
color_train=np.vstack((red_train,green_train,blue_train)).T
color_train=color_train.astype(np.uint8)
label_train=trian_data[:,6]


x_test=test_data[:,0]
y_test=test_data[:,1]
z_test=test_data[:,2]
red_test_1=test_data[:,3]
# red_test=gamma_transform(test_data[:,3],1)
# green_test=gamma_transform(test_data[:,3],1)
# blue_test=gamma_transform(test_data[:,3],1)
red_test=test_data[:,3]
green_test=test_data[:,3]
blue_test=test_data[:,3]
data_test=np.vstack((x_test,y_test,z_test)).T
color_test=np.vstack((red_test,green_test,blue_test)).T
color_test=color_test.astype(np.uint8)
label_test=test_data[:,6]




write_ply('/home/zt/TAO/KPConv-2-master/Data/Vaihingen_gama01/Vaihingen3D_train.ply',[data_train,color_train,label_train],
          ['x','y','z','red','green','blue','class'])

write_ply('/home/zt/TAO/KPConv-2-master/Data/Vaihingen_gama01/Vaihingen3D_test.ply',[data_test,color_test,label_test],
          ['x','y','z','red','green','blue','class'])
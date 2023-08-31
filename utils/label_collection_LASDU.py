import os
import re
from os.path import isdir,join
import collections
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import datetime
import time
import urllib.request
import requests
import re
import hashlib
from utils.ply import read_ply
np.set_printoptions(suppress=True)
plt.style.use('ieee')
plt.style.context('ieee')




mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负


path = '/media/zt/D/PycharmProjects/KPConv-PyTorch-master/Data/LASDU/original_ply'
original_name = np.sort([f for f in os.listdir(path) if f[-4:]=='.ply'])

# ulabels_0=[]
# Ground_2=[]
# Vegetation_5=[]
# Building_6=[]
# Water_9=[]
# Bridge_17=[]
labels_0=[]#Ground
labels_1=[]#Building
labels_2=[]#Tree
labels_3=[]#Low_evg
labels_4=[]#Artifact


for i in original_name:
    # truth=np.loadtxt(join(path,i))
    truth1 = read_ply(join(path, i))
    truth=truth1['class']
    counts=collections.Counter(truth)
    labels_0.append(counts[0])

    print('{}'.format(i), counts)
    labels_1.append(counts[1])
    labels_2.append(counts[2])
    labels_3.append(counts[3])
    labels_4.append(counts[4])



labels_00=sum(labels_0)
labels_01=sum(labels_1)
labels_02=sum(labels_2)
labels_03=sum(labels_3)
labels_04=sum(labels_4)
total=labels_00+labels_01+labels_02+labels_03+labels_04
print('labels_00:',labels_00,'labels_01:', labels_01,'labels_02',labels_02,'labels_03:',labels_03,'labels_04:',labels_04)
print('Ground:',(labels_00/total)*100,'Building:', (labels_01/total)*100,'Tree',(labels_02/total)*100,'Low_evg:',(labels_03/total)*100,'Artifact:',(labels_04/total)*100)

# r = [[Ground/6148920,       1],  # Ground
#      [Vegetation/1321488,       1],  # Vegetation
#      [Building/782455,      1],  # Building
#      [Water/19266,      1],  # Water
#      [Bridge/25034,       1],  # Bridge
#      ]
# print(r)
# # 这是柱图x轴标签
# #ysr = ['Grd' ,'Veg' ,'Bui' ,'Wat' ,'Bri']
# ysr = ['C00' ,'C01' ,'C02' ,'C03' ,'C04']
#
# def DrawGeoDtaabse(rcount, y):
#     # 第一行 第一列图形   2,1 代表2行1列
#     ax1 = plt.subplot(1 ,1 ,1)
#     # 第二行 第一列图形
#     # ax3 = plt.subplot(2 ,1 ,2)
#     # 默认时间格式
#     plt.sca(ax1)
#     # plt.xlabel("" ,color = 'black')  # X轴标签
#     plt.ylabel("Percentage(%)" ,color = 'black')  # Y轴标签
#     # plt.grid(True)   #显示格网
#     # plt.gcf().autofmt_xdate() #显示时间
#     plt.legend() # 显示图例
#     plt.title("Point Distribution Across Each Category")  # 标题
#
#     x1 = [1 ,5 ,9 ,13 ,17 ] # x轴点效率位置
#     x2 = [i + 1 for i in x1]    # x轴线效率位置
#     y1 = [i[0] for i in rcount] # y轴点效率位置
#     y2 = [i[1] for i in rcount] # y轴线效率位置
#     ##占位以免 数据源标签丢失
#     y0 = ["", "", "", "", ""]
#     plt.bar(x1, y1, alpha=0.7, width=1, color='orange', label="test", tick_label=y0)
#     plt.bar(x2, y2, alpha=0.7, width=1, color='blue', label="test_paper", tick_label=y)
#
#
#
#
#     # plt.sca(ax3)
#     # plt.xlabel("数据源" ,color = 'r')  # X轴标签
#     # plt.ylabel("条/s" ,color = 'r')  # Y轴标签
#     # # plt.grid(True)
#     # plt.legend() # 显示图例
#     # plt.title("[写入]效率")  # 图标题
#     #
#     #
#     # y1 = [i[0] for i in wcount]
#     # y2 = [i[1] for i in wcount]
#     # y3 = [i[2] for i in wcount]
#     # y0 = ["" ,"" ,"" ,"" ,"" ,"" ,"" ,""]
#     # plt.bar(x1, y1, alpha=0.7, width=0.6, color='r' ,label="点", tick_label=y0)
#     # plt.bar(x3, y3, alpha=0.7, width=0.6, color='b' ,label="面", tick_label=y0)
#     # plt.bar(x2, y2, alpha=0.7, width=0.6, color='g' ,label="线", tick_label=y)
#
#     plt.legend()
#     plt.show()
#
# DrawGeoDtaabse(r ,ysr)

import os
import re

import numpy as np

path = r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\train-track4-truth\Track4-Truth\\'
original_name = os.listdir(path)

# for i in original_name:
#
#     name = i.split('_')[0] + '_' + i.split('_')[1] + '_' + 'PC3'
#     print(name)
file=open(r'D:\Python\KPConv-PyTorch-master\KPConv-PyTorch-master\Data\DFC2019\filenames.txt',mode='w')
for i ,data in enumerate(original_name):
    name = data.split('_')[0] + '_' + data.split('_')[1] + '_' + 'PC3'
    file.write('\n'+str(name))
file.close()
print("改名完成")

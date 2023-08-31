import os
import re
path = '/media/zt/D/3060/KPConv-2-master/Data/DFC2019/DFC2019_DATA/train-track4-truth/Track4-Truth'
original_name = os.listdir(path)

for i in original_name:

    name=i.split('_')[0]+'_'+i.split('_')[1]+'_'+'PC3'+'.txt'


    os.rename(os.path.join(path, i), os.path.join(path, name))
print("改名完成")

import numpy as np
from utils.ply import *
import matplotlib.pyplot as plt
np.random.seed(0)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science','no-latex'])
plt.style.use(['science','ieee'])

mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

# set test scores

# scoresT=np.loadtxt(r'D:\pytorch_202207\KPConv-PyTorch-master\Data\3DLabeling\Vaihingen3D_Traininig.pts')
# x = scoresT[:,3]
scoresT=read_ply('/media/zt/D/PycharmProjects/KPConv-PyTorch-master/Data/LASDU/original_ply_1/section_1.ply')
x=scoresT['red']
print(x.shape)
# plot histogram
bins = range(0,256,10)
with plt.style.context(['science', 'no-latex']):
    plt.figure()
    plt.hist(x,bins=bins,
             color="#377eb8",
             histtype="bar",
             rwidth=1.0,
             edgecolor="#000000")
    # set x,y-axis label
    plt.xlabel("Train_intensity ($\gamma $=$1$)")
    plt.ylabel("Frequency")

    plt.show()

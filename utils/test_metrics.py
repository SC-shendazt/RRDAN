import numpy as np
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from utils.ply import read_ply
from utils.metrics import smooth_metrics
data_true=read_ply('/media/zt/D/3060/KPConv-2-master/Data/Dales_ply/original_ply/5100_54490.ply')
data_test=read_ply('/media/zt/D/3060/KPConv-2-master/test/Log_2022-06-18_10-23-57/predictions/5100_54490.ply')
y_true=data_true['class']
y_preds=data_test['preds']
a=data_true['reflectance']
f1=f1_score(y_true,y_preds,average='macro')
accuracy=accuracy_score(y_true,y_preds)
C=confusion_matrix(y_true,y_preds,normalize='pred')
PRE, REC, F1, IoU, ACC = smooth_metrics(C[1:, 1:])
print(f1,
      accuracy,
      IoU.mean())

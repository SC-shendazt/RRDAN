import time
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from utils.metrics import smooth_metrics
# Custom libs

# Dataset
from plyfile import PlyData, PlyElement
labels = ['Unclassified','Gnd', 'Trees', 'Car', 'Truck', 'Wire', 'Fence', 'Poles' , 'Bldngs']
# labels=['unlabels', 'Powerline','Low vegetation','surface','Car','Fence', 'Roof','Facade', 'shurb', 'Tree']
ignored_labels = ['Unclassified']
final_labels = labels[:]
for each in ignored_labels: final_labels.remove(each)

# Given below is the path to test output and ground truth files.
# All files in test output file should be present in ground truth folder for successful execution of this file
# This Also works in windows
test_predictions_path = '/home/zt/TAO/KPConv-2-master/test/Log_2022-06-11_11-41-52/predictions/'
test_groundtruth_path = '/home/zt/TAO/KPConv-2-master/results/Log_2022-06-11_11-41-52/val_preds_40/'
files_pred = [f for f in os.listdir(test_predictions_path) if f[-4:] == '.ply']
files_ground = [f for f in os.listdir(test_groundtruth_path) if f[-4:] == '.ply']
if(all(each in files_ground for each in files_pred )):
    print("All files good")
else:
    print("Error some files at ",test_predictions_path, "not matching with files at",test_groundtruth_path)
    exit()

once = False
total_list_micro = list()
total_list_macro = list()
total_list_miou =list()
Cum = None
for each_file in files_pred:
    print('\n Loading.... ', os.path.join(test_predictions_path, each_file))
    data_pred = PlyData.read(os.path.join(test_predictions_path, each_file))
    data_grtr = PlyData.read(os.path.join(test_groundtruth_path, each_file))
    y_true = data_grtr.elements[0]['class']
    y_pred = data_pred.elements[0]['preds']
    """ # Uncomment these lines for saving confusion matrix in a color scale in pdf format
    C = confusion_matrix(y_true, y_pred, normalize='pred')
    for l_ind, label_value in enumerate(labels):
        if label_value in ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)
    if not once:
        Cum = C
    else:
        Cum += C
    plt.imshow(Cum)
    ticks = range(len(final_labels))
    plt.xticks(ticks=ticks,labels=final_labels)
    plt.yticks(ticks=ticks,labels=final_labels)
    if not once: plt.colorbar()
    once = True
    plt.title(" Confusion Matrix ")
    plt.savefig("results/"+each_file[:-4]+'.pdf')
    """
    F1_score_micro = f1_score(y_true, y_pred, average='micro')
    F1_score_macro = f1_score(y_true, y_pred, average='macro')
    C = confusion_matrix(y_true, y_pred, normalize='pred')
    for l_ind, label_value in enumerate(labels):
        if label_value in ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)
    if not once:
        Cum = C
    else:
        Cum += C
    PRE, REC, F1, IoU, ACC = smooth_metrics(Cum)
    print("micro F1: \t",F1_score_micro)
    print("macro F1: \t",F1_score_macro)
    print('MIOU: \t',IoU.mean())
    total_list_miou += [IoU.mean()]
    total_list_macro += [F1_score_macro]
    total_list_micro += [F1_score_micro]
avg_micro = sum(total_list_micro)/len(total_list_micro)
avg_macro = sum(total_list_macro)/len(total_list_macro)
MIOU=sum(total_list_miou)/len(total_list_miou)
print( " Final Avg micro : ", avg_micro, "|  Avg  macro : ", avg_macro,"|  Avg  miou : ",MIOU)
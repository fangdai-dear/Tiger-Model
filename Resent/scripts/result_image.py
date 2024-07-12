import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from scripts.multiAUC import Metric
import numpy
from random import sample
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scripts.plot import create_multi_bars

label = ['No Finding',
            'Enlarged Card..',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

df1 = pd.read_csv('/export/home/daifang/CXP/result/Male_age.txt', sep=',', names=label)
data1 = []
# for indexs in df1.index:
#         rowData = df1.loc[indexs].values[0:14]
#         rowData = rowData.tolist()
#         data1.append(rowData)
df2 = pd.read_csv('/export/home/daifang/CXP/result/Female_age.txt', sep=',', names=label)
for indexs in df1.index:
        rowData = df2.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        data1.append(rowData)
AUC1 = []
auc1 = pd.read_csv('/export/home/daifang/CXP/result/Male_Model_80-all/Male_Model_80-all_auc.txt', sep=',', names=label)
auc2 = pd.read_csv('/export/home/daifang/CXP/result/Female_Model_80-all/Female_Model_80-all_auc.txt', sep=',', names=label)
# for indexs in auc1.index:
#         rowData = auc1.loc[indexs].values[0:14]
#         rowData = rowData.tolist()
#         AUC1 .append(rowData)
for indexs in auc2.index:
        rowData = auc2.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        AUC1 .append(rowData)

# color1 = ['#FFD9Da','#FFC0BE','#FF82A9','#7F95D1']
# color1 = ['#CDB4DB','#FFC8DD','#FFAFFC','#BDE0FE']
color1 = ['#CDB4DB','#FFC0BE','#FF82A9','#A2D2FF']
text_label1 = [ 'Female:18-40 (9997)','Female:40-60 (17364)','Female:60-80 (21908)','Female:>80 (7840)']
base_auc1 = [0.875,0.668,0.855,0.727,0.771,0.837,0.734,0.740,0.697,0.873,0.880,0.795,0.750,0.878]
title1 = 'CXP datasets subgroup: Female 18-40, 40-60, 60-80, >80)'
save1 = "/export/home/daifang/CXP/figure/test_age(female).png"
create_multi_bars(label, data1, base_auc1, AUC1, color1, text_label1, title1, save1, bar_gap=0.03)


data2 = []
df11 = pd.read_csv('/export/home/daifang/CXP/result/Female_rate/Female_rate_auc.txt', sep=',', names=label)
for indexs in df11.index:
        rowData = df11.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        data2.append(rowData)
AUC2 = []
auc22 = pd.read_csv('/export/home/daifang/CXP/result/Female_rate/Female_rate_after.txt', sep=',', names=label)
for indexs in auc22.index:
        rowData = auc22.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        AUC2.append(rowData)


color2 = ['gold','dimgray','deepskyblue','orange']
text_label2 = ['Female:Asian (2398)','Female:Black (973)','Female:White (9499)','Female:Other (3692)']

base_auc2 = [0.875,0.698,0.851,0.747,0.76,0.829,0.754,0.777,0.727,0.873,0.860,0.785,0.760,0.868]

title2 = 'CXP datasets subgroup: Female: Asian, Black, White, Other' 

save2 = "/export/home/daifang/CXP/figure/test_rate(female).png"

create_multi_bars(label, data2, base_auc2, AUC2, color2, text_label2, title2, save2,bar_gap=0.03)
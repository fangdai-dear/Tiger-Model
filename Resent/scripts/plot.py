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
import numpy as np
from sklearn.manifold import TSNE


def plotimage(train, val, test, ylabel, modelname):
    y1 = np.array(torch.tensor(train, device='cpu'))
    y2 = np.array(torch.tensor(val, device='cpu'))
    y3 = np.array(torch.tensor(test, device='cpu'))
    plt.title("%s vs. Number of Training Epochs" % ylabel)
    plt.xlabel("Training Epochs")
    plt.ylabel(ylabel)
    if ylabel == 'Loss':
        plt.plot(range(1, len(train) + 1), y1, label="Train %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y2, label="Valid %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y3, label="Test  %s" % ylabel)
    if ylabel != 'Loss':
        plt.plot(range(0, len(train)), y1, label="Train %s" % ylabel)
        plt.plot(range(0, len(train)), y2, label="Valid %s" % ylabel)
        plt.plot(range(0, len(train)), y3, label="Test  %s" % ylabel)
        # plt.plot(range(0, len(train)), y3, label="Test Min %s" % ylabel)
        plt.ylim((0, 1.))

    plt.xticks(np.arange(1, 100, 10.0))
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(linestyle=":", color="r")  
    plt.savefig("./result/%s/%s_%s.png" % (modelname, modelname, ylabel))
    plt.cla()


def result_csv(train_auc, val_auc, testA_auc,modelname):

    y1 = np.array(torch.tensor(train_auc, device='cpu'))
    y2 = np.array(torch.tensor(val_auc, device='cpu'))
    y3 = np.array(torch.tensor(testA_auc, device='cpu'))
    # y4 = np.array(torch.tensor(testB_auc, device='cpu'))
    CSV0 = pd.DataFrame(y1, columns=['train_AUC'])
    CSV1 = pd.DataFrame(y2, columns=['valid_AUC'])
    CSV2 = pd.DataFrame(y3, columns=['test_AUC'])
    # CSV3 = pd.DataFrame(y4, columns=['test_min_AUC'])
    CSV = pd.concat([CSV0, CSV1, CSV2], axis=1)
    CSV.to_csv("./result/%s/%s.csv" % (modelname,modelname), encoding='gbk')


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics




def create_multi_bars(labels, datas1,base_auc,AUC,color,label,title, save, tick_step=1, group_gap=0.25, bar_gap=0):
  
    plt.grid(linestyle=":", color="g")
    plt.figure(dpi=2000, figsize=(7, 2.5))

    ticks = np.arange(len(labels)) * tick_step

    group_num = len(datas1)

    group_width = tick_step - group_gap

    bar_span = group_width / group_num
  
    bar_width = bar_span - bar_gap

    baseline_x = ticks - (group_width - bar_span) / 2
    
    plt.plot(0,base_auc[0],'*',color = 'red',markersize = 5.5,label = 'All')
    plt.bar(0.25, 0.6, bar_width, color = 'gold',label= 'AUC development')
    # plt.bar(0.25, 0.6, bar_width, color = '#F8ACFF',label= 'AUC development')
    for index, y1 in enumerate(AUC):
        # print(baseline_x+ 3.5*bar_span)
        plt.plot(baseline_x+ 1.5*bar_span,base_auc,'*',color = 'red',markersize = 5.5)
        plt.bar(baseline_x + index*bar_span, y1, bar_width, color = 'gold')
        # plt.bar(baseline_x + index*bar_span, y1, bar_width, color = '#F8ACFF')

    for index, y in enumerate(datas1):
        plt.bar(baseline_x + index*bar_span, y, bar_width,fc=color[index],label =label[index])
        # plt.plot(baseline_x+ 3.5*bar_span,base_auc,'*',color = 'r',markersize = 6)

    # for index, y1 in enumerate(AUC):
    #     plt.plot(baseline_x+ index*bar_span,y1,'.',color = 'gold',markersize = 3)
        # plt.bar(baseline_x + index*bar_span, y1, bar_width, color = 'gold',alpha = 0.5)

    plt.ylabel('AUC',fontsize=8)
    # plt.title('CXP datasets subgroup: Male/Female 18-40,40-60,60-80,>80)')
    plt.title(title,fontsize=8)
    plt.xticks(ticks, labels,fontsize=6,rotation=25)
    plt.yticks(np.arange(0.6, 0.97, 0.05), fontsize=6)
    plt.ylim(ymin=0.6)
    # plt.plot(0,0.885,'*',label = 'All',color='gray')
    plt.legend(fontsize=5, loc=2, bbox_to_anchor=(0.01, 0.98), borderaxespad=0.,ncol=3)
    plt.savefig(save,dpi=300, bbox_inches='tight')


def tnes():
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    X_embedded.shape

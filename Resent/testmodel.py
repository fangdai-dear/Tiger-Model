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
from pandas.core.frame import DataFrame
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def softmax(x):
    exp_x = np.exp(x)  # 为了数值稳定性，减去最大值
    return exp_x / np.sum(exp_x)

def bootstrap_auc(y, pred, classes, bootstraps=10, fold_size=500):
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



if __name__ == '__main__':
    batch_size = 128
    flag = torch.cuda.is_available()
    if flag:
        print("CUDA可使用")
    else:
        print("CUDA不可用")
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("驱动为：", device)
    net = models.resnet18(pretrained=False)
    # 导入预训练的模型
    features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(features, 2),
    )
    modelname = "Model_all"

    CSV, Mean_auc, Max_auc, Min_auc, Name_auc, Test_auc = [], [], [], [], [], []
    net.load_state_dict(torch.load("./modelsaved/A_goodA0.87_epoch15.pth",
                                   map_location=lambda storage, loc: storage), strict=False)

    file_name = "test"
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
    }

    print("%s Initializing Datasets and Dataloaders..." % file_name)

    test_datasets = datasets.ImageFolder("/export/home/daifang/Diffusion/Resnet/dataset/figure/PTC", transform=data_transforms['test'])
    num_workers = 20
    dataloaders_dict = {
        'test': DataLoader(test_datasets, batch_size=64, shuffle=True, num_workers=num_workers)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for phase in ['test']:
        net.eval()
        running_loss = 0.0
        running_corrects = 0
        prob_all, label_all, output_list = [], [], [[], []]
        outputs = []
        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            output_list[0].extend(outputs.tolist())

            output_list[1].extend(labels.tolist())
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            prob_all.extend(outputs[:, 1].cpu().detach().numpy())
            label_all.extend(labels.cpu().detach().numpy())
        data_auc = roc_auc_score(label_all, prob_all)
        epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
        print(prob_all)
        print(label_all)
        scores = np.array(prob_all)
        y = np.array(label_all)
        # print(scores)
        # print(y)

        prob_true, prob_pred = calibration_curve(y, softmax(scores), n_bins=10)
        print(prob_true)
        print(prob_pred)
        # 绘制校准曲线
        plt.figure(figsize=(10, 10))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Probability')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid()
        plt.savefig("calibration_curve.png", dpi=300, bbox_inches='tight')
        statistics = bootstrap_auc(y, scores, [0, 1, 2, 3, 4])

        mean_auc = np.mean(statistics, axis=1).max()
        max_auc = np.max(statistics, axis=1).max()
        min_auc = np.min(statistics, axis=1).max()

        Name_auc.append(file_name + phase)
        Test_auc.append(data_auc)
        Mean_auc.append(mean_auc)
        Max_auc.append(max_auc)
        Min_auc.append(min_auc)
        print('Testing: {}  Acc: {:.4f}   AUC: {:.4f} ({:.4f} - {:.4f}) '.format(phase, epoch_acc, data_auc,min_auc, max_auc))

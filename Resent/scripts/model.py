import torch
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
from tqdm import tqdm
from random import sample
from scripts.plot import bootstrap_auc,result_csv,plotimage
import pynvml
pynvml.nvmlInit()
from prettytable import PrettyTable


def train_model(model, dataloaders, criterion, optimizer,num_epochs, modelname, device):
    global VAL_auc,TEST_auc
    since = time.time()
    train_loss_history, valid_loss_history, test_loss_history= [], [], []
    test_maj_history, test_min_history = [], []
    train_auc_history, val_auc_history, test_auc_history = [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(num_epochs):
        start = time.time()

        print('{}  Epoch {}/{}  {}'.format('-' * 30, epoch, num_epochs - 1, '-' * 30))
        for phase in ['train','valid', 'test']:
            if phase == 'train' and epoch != 0:
                model.train()
            else:
                model.eval()
            running_loss,running_corrects,prob_all, label_all = [], [], [], []
            with tqdm(range(len(dataloaders[phase])),desc='%s' % phase, ncols=100) as t:
                if epoch == 0 :
                    t.set_postfix(L = 0.000, usedMemory = 0)

                for data in dataloaders[phase]:
                    inputs, labels, sub = data
                    print(labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True) 
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train' and epoch != 0:
                            loss.backward()
                            optimizer.step()
                    running_loss.append(loss.item())
                    running_corrects.append((preds.cpu().detach() == labels.cpu().detach()).numpy())

                    prob_all.extend(outputs[:, 1].cpu().detach().numpy())
                    label_all.extend(labels.cpu().detach().numpy())

                    """
                    B:batch 
                    L:Loss
                    maj: Maj group AUC
                    min: Min group AUC
                    n: NVIDIA Memory used
                    """   
                    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
                    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
                    usedMemory = usedMemory/meminfo              
                    t.set_postfix(loss = loss.data.item(), usedMemory = usedMemory)  # 
                    t.update()



            # num = len(label_all)
            # auc = roc_auc_score(label_all, prob_all)
            # epoch_loss = np.mean(running_loss)
            # label_all = np.array(label_all)
            # prob_all = np.array(prob_all)
            # statistics = bootstrap_auc(label_all, prob_all, [0,1,3,4,5])
            # max_auc = np.max(statistics, axis=1).max()
            # min_auc = np.min(statistics, axis=1).max()
            # print('{} --> Num: {} Loss: {:.4f}  AUROC: {:.4f} ({:.2f} ~ {:.2f})'.format(
            #     phase, num, epoch_loss, auc, min_auc, max_auc ))
            if modelname =="Thyroid_PF":
                try:
                    data_auc = roc_auc_score(Label,Output)
                    Data_auc_maj = roc_auc_score(Label_maj, Output_maj)
                    Data_auc_min = roc_auc_score(Label_min, Output_min)
                except:
                    data_auc = roc_auc_score(Label,Output)
                    Data_auc_maj = 0
                    Data_auc_min = 0
                epoch_loss = running_loss / Batch
                statistics = bootstrap_auc(Label, Output, [0,1,2,3,4])
                max_auc = np.max(statistics, axis=1).max()
                min_auc = np.min(statistics, axis=1).max()
                if G == [] and phase == "train":
                    G1.append(0)
                elif phase == "train":
                    G1.append(sum(G)/len(G))
                print('{} --> Num: {} Loss: {:.4f}  Gamma: {:.4f} AUROC: {:.4f} ({:.2f} ~ {:.2f}) (Maj {:.4f}, Min {:.4f})'.format(
                phase, len(outputs_out), epoch_loss, G1[-1], data_auc, min_auc, max_auc, Data_auc_maj, Data_auc_min))

            else:
                myMetic = Metric(Output,Label)
                data_auc,auc = myMetic.auROC() 
                epoch_loss = running_loss / Batch
                statistics = bootstrap_auc(Label, Output, [0,1,2,3,4])
                max_auc = np.max(statistics, axis=1).max()
                min_auc = np.min(statistics, axis=1).max()
                if G == [] and phase == "train":
                    G1.append(0)
                elif phase == "train":
                    G1.append(sum(G)/len(G))
                print('{} --> Num: {} Loss: {:.4f}  AUROC: {:.4f} ({:.2f} ~ {:.2f}) (Maj {:.4f}, Min {:.4f})'.format(
                        phase, len(outputs_out), epoch_loss, data_auc, min_auc, max_auc, data_auc_maj,data_auc_min))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_auc_history.append(auc)

            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                val_auc_history.append(auc)

            if phase == 'test':
                test_loss_history.append(epoch_loss)
                test_auc_history.append(auc)
      
            if phase == 'valid' and train_auc_history[-1] >= 0.9:
                if val_auc_history[-1] >= max(val_auc_history) or test_auc_history[-1] >= max(test_auc_history):
                    print("In epoch %d, better AUC(%.3f) and save model. " % (epoch, float(val_auc_history[-1])))
                    PATH = '/export/home/daifang/Diffusion/Resnet/modelsaved/%s/e%d_%s_V%.3fT%.3f.pth' % (modelname,epoch,modelname,val_auc_history[-1],test_auc_history[-1])
                    torch.save(model.state_dict(),PATH)
    
        print("learning rate = %.6f     time: %.1f sec" % (optimizer.param_groups[-1]['lr'], time.time() - start))
        if epoch != 0:
            scheduler.step()
        print()

        plotimage(train_auc_history, val_auc_history, test_auc_history,"AUC", modelname)        
        plotimage(train_loss_history, valid_loss_history, test_loss_history,"Loss", modelname)
        result_csv( train_auc_history, val_auc_history, test_auc_history, modelname)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
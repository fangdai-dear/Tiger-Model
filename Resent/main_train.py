import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.model import train_model
import scripts.dataset as DATA
import scripts.config as config
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args):
    modelname = args.modelname
    imagepath = args.imagepath
    label_num,subgroup_num = config.THYROID()
    Datasets = DATA.Thyroid_Datasets

    if args.architecture =='resnet':
        net = models.resnet18(pretrained=True)  
        features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(features, args.Class))


    if os.path.exists('./modelsaved/%s' % modelname) == False:  
        os.makedirs('./modelsaved/%s' % modelname)
    if os.path.exists('./result/%s' % modelname) == False:  
        os.makedirs('./result/%s' % modelname)


    data_transforms = config.Transforms(modelname)

    print("%s Initializing Datasets and Dataloaders..." % modelname)
    
    transformed_datasets = {}
    transformed_datasets['train'] = Datasets(
        path_to_images=imagepath,
        fold=args.train_data,
        PRED_LABEL=label_num,
        transform=data_transforms['train'])
    transformed_datasets['valid'] = Datasets(
        path_to_images=imagepath,
        fold=args.valid_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])
    transformed_datasets['test'] = Datasets(
        path_to_images=imagepath,
        fold=args.test_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=1,
        shuffle=False,
        num_workers=24)
    dataloaders['test'] = torch.utils.data.DataLoader(
        transformed_datasets['test'],
        batch_size=1,
        shuffle=False,
        num_workers=24)


    if args.modelload_path:
        net.load_state_dict(torch.load('%s' % args.modelload_path , map_location=lambda storage, loc: storage),strict=False)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = net.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)
   

    if args.Class > 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_model(net,dataloaders, criterion, optimizer, args.num_epochs, modelname, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="Thyroid")
    parser.add_argument("--architecture", type=str, choices= ["resnet","densnet","efficientnet"], default="resnet")
    parser.add_argument("--modelload_path", type=str,  default= None)
    parser.add_argument("--imagepath", type=str,  default="./dataset/")
    parser.add_argument("--train_data", type=str, default='thyroid_train')
    parser.add_argument("--valid_data", type=str, default='thyroid_valid')
    parser.add_argument("--test_data", type=str, default='thyroid_test')
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--Class", type=int, default=2)
    args = parser.parse_args()
    main(args)

   

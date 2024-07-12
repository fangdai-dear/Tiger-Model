import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

def Divide(outputs, labels, subggroup):
    Min_img, Min_lab, Maj_img, Maj_lab = torch.ones(0, len(labels[0])), torch.ones(0, len(labels[0])), \
                                        torch.ones(0, len(labels[0])), torch.ones(0, len(labels[0]))
    for x in range(len(subggroup)):
        if subggroup[x].tolist()[0] == 1 :
            Min_img = torch.cat((Min_img, outputs[x].detach().unsqueeze(0)), 0)
            Min_lab = torch.cat((Min_lab, labels[x].detach().unsqueeze(0)), 0)
        else:
            Maj_img = torch.cat((Maj_img, outputs[x].detach().unsqueeze(0)), 0)
            Maj_lab = torch.cat((Maj_lab, labels[x].detach().unsqueeze(0)), 0)
    return Maj_img, Maj_lab, Min_img, Min_lab

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, subggroup, outputs):
        Min_img, Maj_img, = torch.ones(0, 100), torch.ones(0, 100)
        for x in range(len(subggroup)):
            if subggroup[x].tolist()[0] == 1 :
                Min_img = torch.cat((Min_img, outputs[x].detach().unsqueeze(0)), 0)
            else:
                Maj_img = torch.cat((Maj_img, outputs[x].detach().unsqueeze(0)), 0)

        source = Min_img
        target = Maj_img

        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class BatchSpectralShrinkage(nn.Module):

    def __init__(self, k=1):
        super(BatchSpectralShrinkage, self).__init__()
        self.k = k

    def forward(self, feature):
        result = 0
        u, s, v = torch.svd(feature.t())
        num = s.size(0)
        for i in range(self.k):
            result += torch.pow(s[num-1-i], 2)
        return result


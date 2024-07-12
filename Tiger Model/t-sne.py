import torch
from sklearn.manifold import TSNE 
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# # setup_seed(1337)


# class FeatureExtractor(nn.Module):
#     def __init__(self, submodule, extracted_layers):
#         super(FeatureExtractor, self).__init__()
#         self.submodule = submodule
#         self.extracted_layers = extracted_layers
 
#     def forward(self, x):
#         outputs = []
#         print('---------',self.submodule._modules.items())
#         for name, module in self.submodule._modules.items():
#             if "fc" in name:
#                 x = x.view(x.size(0), -1)
#             print(module)
#             x = module(x)
#             print('name', name)
#             if name in self.extracted_layers:
#                 outputs.append(x)
#         return outputs


# data_transforms = {
#     'valid': transforms.Compose([
#         transforms.CenterCrop(768),
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
# }
# data_dir = '/export/home/daifang/Diffusion/diffusers/dataset/T-sne'
# val_datasets = datasets.ImageFolder(os.path.join(data_dir), transform=data_transforms['valid'])

# dataloaders_dict = {
#     'valid': DataLoader(val_datasets, batch_size=1000, shuffle=True, num_workers=20)}

# device = torch.device("cuda")

# model = models.resnet50(pretrained=True)
# model.load_state_dict(torch.load('/export/home/daifang/Diffusion/mmselfsup/work_dirs/export/npid_resnet50_8xb32-steplr-200e_in1k_daifang/epoch_200.pth', 
#                                 map_location=lambda storage, loc: storage),strict=False)

# # model.load_state_dict(torch.load('/export/home/daifang/Diffusion/mmselfsup/work_dirs/resnet50_linear-8xb32-coslr-100e_in1k_MAE_daifang/epoch_100.pth', 
# #                                 map_location=lambda storage, loc: storage),strict=False)
# model.fc = nn.Sequential(*[])
# model = model.to(device)


# with torch.no_grad():
#     i = 0
#     for image_batch, label_batch in dataloaders_dict['valid']:
#         image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#         label_batch = label_batch.long().squeeze()
#         inputs = image_batch
#         feature = model(inputs)

#         if i == 0:
#             feature_bank = feature
#             label_bank = label_batch
#         else:
#             feature_bank = torch.cat((feature_bank, feature))
#             label_bank = torch.cat((label_bank, label_batch))
#         i +=1

# feature_bank = feature_bank.cpu().numpy()
# label_bank = label_bank.cpu().numpy()
# # tsne = TSNE(n_components = 2, perplexity = 50, learning_rate = 200, early_exaggeration = 35)
# # output = tsne.fit_transform(feature_bank) 
# # colors = ['r', 'g', 'b', 'pink', 'yellow', 'c', 'orange',  'black','deeppink']
# # alpha=[0.3,0.1,0.3,0.3,0.3,0.3,0.3,0.9,0.3]
# # for i in range(9):	
# #     index = (label_bank==i)
# #     plt.scatter(output[index, 0], output[index, 1], s=9, c=colors[i], alpha=alpha[i])
# # plt.legend(["Benign", "PTC", "FTC", "MTC", "SD-noseed","SD-seed","DiT_FTC1","SD-try","DiT-PTC"])
# # plt.savefig('./t-sne_seed1.png')


# tsne = TSNE(n_components = 2, perplexity = 50, learning_rate = 1000, early_exaggeration = 220, n_iter = 1000, min_grad_norm = 1e-6)
# output = tsne.fit_transform(feature_bank) 
# colors = ['r', 'g', 'b', 'orange', 'deeppink']
# alpha=[0.3, 0.1, 0.3, 0.3, 0.3]
# for i in range(5):	
#     index = (label_bank==i)
#     plt.scatter(output[index, 0], output[index, 1], s=5, c=colors[i], alpha=alpha[i])
# plt.legend(["Benign", "PTC", "FTC", "MTC", "sd-mtc"])
# plt.savefig('./t-sne_seed1.png')


#  n_components：int，可选（默认值：2）嵌入式空间的维度。

# perplexity：浮点型，可选（默认：30）较大的数据集通常需要更大的perplexity。考虑选择一个介于5和50之间的值。由于t-SNE对这个参数非常不敏感，所以选择并不是非常重要。

# early_exaggeration：float，可选（默认值：4.0）这个参数的选择不是非常重要。

# learning_rate：float，可选（默认值：1000）学习率可以是一个关键参数。它应该在100到1000之间。如果在初始优化期间成本函数增加，则早期夸大因子或学习率可能太高。如果成本函数陷入局部最小的最小值，则学习速率有时会有所帮助。

# n_iter：int，可选（默认值：1000）优化的最大迭代次数。至少应该200。

# n_iter_without_progress：int，可选（默认值：30）在我们中止优化之前，没有进展的最大迭代次数。

# 0.17新版​​功能：参数n_iter_without_progress控制停止条件。

# min_grad_norm：float，可选（默认值：1E-7）如果梯度范数低于此阈值，则优化将被中止。

# metric：字符串或可迭代的，可选，计算特征数组中实例之间的距离时使用的度量。如果度量标准是字符串，则它必须是scipy.spatial.distance.pdist为其度量标准参数所允许的选项之一，或者是成对列出的度量标准.PAIRWISE_DISTANCE_FUNCTIONS。如果度量是“预先计算的”，则X被假定为距离矩阵。或者，如果度量标准是可调用函数，则会在每对实例（行）上调用它，并记录结果值。可调用应该从X中获取两个数组作为输入，并返回一个表示它们之间距离的值。默认值是“euclidean”，它被解释为欧氏距离的平方。

# init：字符串，可选（默认值：“random”）嵌入的初始化。可能的选项是“随机”和“pca”。 PCA初始化不能用于预先计算的距离，并且通常比随机初始化更全局稳定。

# random_state：int或RandomState实例或None（默认）
# 伪随机数发生器种子控制。如果没有，请使用numpy.random单例。请注意，不同的初始化可能会导致成本函数的不同局部最小值。

# method：字符串（默认：‘barnes_hut’）
# 默认情况下，梯度计算算法使用在O（NlogN）时间内运行的Barnes-Hut近似值。 method ='exact’将运行在O（N ^ 2）时间内较慢但精确的算法上。当最近邻的误差需要好于3％时，应该使用精确的算法。但是，确切的方法无法扩展到数百万个示例。0.17新版​​功能：通过Barnes-Hut近似优化方法。

# angle：float（默认值：0.5）
# 仅当method ='barnes_hut’时才使用这是Barnes-Hut T-SNE的速度和准确性之间的折衷。 ‘angle’是从一个点测量的远端节点的角度大小（在[3]中称为theta）。如果此大小低于’角度’，则将其用作其中包含的所有点的汇总节点。该方法对0.2-0.8范围内该参数的变化不太敏感。小于0.2的角度会迅速增加计算时间和角度，因此0.8会快速增加误差。



import torch
import clip
from sklearn.manifold import TSNE
from PIL import Image
import seaborn as sns
import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes 
from sklearn.svm import SVR
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.manifold import TSNE 
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os
class Visual:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.umap = umap.UMAP(n_components=2, random_state=42) 
    
    def get_one_feat(self,image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        # feats = torch.zeros(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            # feats = torch.cat([feats,image_features],dim=0)
        return image_features
    
    def get_all_feat(self,img_root):
        dirs = os.listdir(img_root)
        labels = []
        feats = torch.zeros(0).to(self.device)
        for index,dir in enumerate(dirs):
            dirpath = os.path.join(img_root,dir)
            imgs = os.listdir(dirpath)
            label = dirpath.split('/')[-1]
            for img in imgs:
                labels.append(index)
                imgpath = os.path.join(dirpath,img)
                image = Image.open(imgpath)
                feat = self.get_one_feat(image)
                if int(label) in [1,3,5,7]:
                    feat = torch.cat([feat,torch.tensor([[int(label)-1]]).to(self.device)],dim=1)
                else:
                    feat =torch.cat([feat,torch.tensor([[int(label)]]).to(self.device)],dim=1)
                feats = torch.cat([feats,feat],dim=0)
        feats = feats.detach().cpu().numpy()
        X_umap = self.umap.fit_transform(feats).tolist() 
        tsne_out = X_umap
        colors = ['g', 'g', 'b', 'b', 'y', 'y', 'r', 'r']
        alpha=[0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8]
        marker = ['o', '^', 'o', '^', 'o', '^', 'o', '^']
        legend1 = ["Benign", "MeDF-Benign", "PTC", "MeDF-PTC", "FTC", "MeDF-FTC", "MTC", "MeDF-MTC"]
        plt.figure(figsize=(10, 10))
        for index in range(len(tsne_out)):	
            i = labels[index]
            plt.scatter(tsne_out[index][0], tsne_out[index][1], s=35, marker = marker[i], c=colors[i], alpha=alpha[i])
        for i in range(len(legend1)):
            plt.scatter(tsne_out[index][0], tsne_out[index][1], s=35, marker = marker[i], c=colors[i], alpha=alpha[i],label = legend1[i])

        plt.legend(ncol=2, fontsize=13)
        plt.savefig('Umap visualization of features.jpg')
        
if __name__ == "__main__":
    visual = Visual()
    #数据集根路径
    img_root = "/export/home/daifang/Diffusion/own_code/dataset/TSNEdata"
    visual.get_all_feat(img_root)
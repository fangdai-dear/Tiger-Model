# import os
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# from gtda.images import Binarizer, HeightFiltration
# from gtda.homology import CubicalPersistence
# from gtda.diagrams import Amplitude
# from sklearn.metrics import pairwise_distances

# # 设置图像处理转换
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 数据加载器
# class ImageFolderDataset(Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.file_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# def load_data(folder_path):
#     dataset = ImageFolderDataset(folder_path, transform=transform)
#     loader = DataLoader(dataset, batch_size=10, shuffle=False)
#     return loader

# # 计算Diversity Score
# def calculate_diversity_score(features):
#     distances = pairwise_distances(features, metric='euclidean')
#     diversity_score = np.mean(distances)
#     return diversity_score

# # 计算Geometry Score
# def calculate_geometry_score(images):
#     binarizer = Binarizer(threshold=0.5)
#     height_filtration = HeightFiltration(direction=np.array([1, 1, 1]))
#     cubical_persistence = CubicalPersistence(homology_dimensions=[0, 1], coeff=2)
#     amplitude = Amplitude(metric='wasserstein', metric_params={'p': 2})
    
#     # Preprocess images
#     images = np.array([img.numpy() if isinstance(img, torch.Tensor) else img for img in images])
#     images_binarized = binarizer.fit_transform(images)
#     images_filtered = height_filtration.fit_transform(images_binarized)
#     diagrams = cubical_persistence.fit_transform(images_filtered)
#     gs_score = amplitude.fit_transform(diagrams)
#     return gs_score.mean()

# # 加载数据
# generated_images_loader = load_data('/export/home/daifang/Diffusion/diffusers/figure/PTC_8-24')
# real_images_loader = load_data('/export/home/daifang/Diffusion/diffusers/figure/PTC_8-12')

# # 提取特征并计算分数
# generated_features = []
# real_features = []

# for img_batch in generated_images_loader:
#     generated_features.extend(img_batch.numpy())

# for img_batch in real_images_loader:
#     real_features.extend(img_batch.numpy())

# generated_features = np.array(generated_features)
# real_features = np.array(real_features)

# # 计算Diversity Score
# generated_div_score = calculate_diversity_score(generated_features.reshape(len(generated_features), -1))
# real_div_score = calculate_diversity_score(real_features.reshape(len(real_features), -1))

# # 计算Geometry Score
# generated_gs_score = calculate_geometry_score(generated_features)
# real_gs_score = calculate_geometry_score(real_features)

# print(f"Generated Images Diversity Score: {generated_div_score}")
# print(f"Real Images Diversity Score: {real_div_score}")
# print(f"Generated Images Geometry Score: {generated_gs_score}")
# print(f"Real Images Geometry Score: {real_gs_score}")


# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import os

# # Function to load and preprocess images
# def load_and_preprocess_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     img = preprocess(img).unsqueeze(0)  # Add batch dimension
#     return img

# # Function to compute image gradients
# def compute_gradients(img):
#     grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
#     grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
#     return grad_x, grad_y

# # Function to calculate Gradient Similarity (GS)
# def gradient_similarity(real_img, gen_img):
#     real_grad_x, real_grad_y = compute_gradients(real_img)
#     gen_grad_x, gen_grad_y = compute_gradients(gen_img)

#     grad_sim_x = F.cosine_similarity(real_grad_x, gen_grad_x, dim=1).mean()
#     grad_sim_y = F.cosine_similarity(real_grad_y, gen_grad_y, dim=1).mean()

#     gs = (grad_sim_x + grad_sim_y) / 2.0
#     return gs.item()

# # Example usage
# real_img_dir = '/export/home/daifang/Diffusion/own_code/GS/real'  # Replace with your real image directory
# gen_img_dir = '/export/home/daifang/Diffusion/own_code/GS/fake'  # Replace with your generated image directory

# real_img_paths = [os.path.join(real_img_dir, fname) for fname in os.listdir(real_img_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]
# gen_img_paths = [os.path.join(gen_img_dir, fname) for fname in os.listdir(gen_img_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

# # Ensure both directories have the same number of images
# assert len(real_img_paths) == len(gen_img_paths), "The number of images in both directories must be the same"

# gs_scores = []

# for real_img_path, gen_img_path in zip(real_img_paths, gen_img_paths):
#     real_img = load_and_preprocess_image(real_img_path)
#     gen_img = load_and_preprocess_image(gen_img_path)

#     gs = gradient_similarity(real_img, gen_img)
#     gs_scores.append(gs)

#     print(f'Processed {real_img_path} and {gen_img_path}: GS = {gs:.3e}')

# mean_gs = np.mean(gs_scores)
# print(f'Mean Gradient Similarity (GS) score: {mean_gs:.3e}')



import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from prdc import compute_prdc

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# Function to extract features using InceptionV3
def extract_features(img_paths, model):
    features = []
    with torch.no_grad():
        for img_path in img_paths:
            img = load_and_preprocess_image(img_path)
            feature = model(img).numpy().squeeze()
            features.append(feature)
    features = np.array(features)
    return features

# Load the InceptionV3 model
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('/export/home/daifang/Diffusion/Resnet/modelsaved/Thyroid_PTC+SD8000/e75_Thyroid_PTC+SD8000_V0.880T0.888.pth', map_location=lambda storage, loc: storage),strict=False)
model.fc = nn.Identity()  # Remove the final classification layer
model.eval()

# Example usage
real_img_dir = '/export/home/daifang/fairness/QuasiParerto/dataset/Thyroid/imageP/train/1'  # Replace with your real image directory
gen_img_dir = '/export/home/daifang/fairness/QuasiParerto/dataset/Thyroid/imageP/valid/1'  # Replace with your generated image directory

real_img_paths = [os.path.join(real_img_dir, fname) for fname in os.listdir(real_img_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]
gen_img_paths = [os.path.join(gen_img_dir, fname) for fname in os.listdir(gen_img_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

# Extract features for real and generated images
real_features = extract_features(real_img_paths, model)
gen_features = extract_features(gen_img_paths, model)

# Calculate PRDC metrics
metrics = compute_prdc(real_features=real_features,
                       fake_features=gen_features,
                       nearest_k=2)

print(metrics)



# import torch
# from torch import nn
# from clip import clip
# import numpy as np

# # 假设我们已经有了一个预训练的CLIP模型
# clip_model, preprocess = clip.load("ViT-L/14@336px", device="cuda")

# # 定义一个函数来获取图像的CLIP嵌入
# def get_clip_embedding(images):
#     with torch.no_grad():
#         images = preprocess(images).unsqueeze(0).to("cuda")
#         image_features = clip_model.encode_image(images)
#     return image_features

# # 定义一个函数来计算MMD距离
# def compute_mmd(x, y, kernel):
#     # 计算核矩阵
#     xx = kernel(x, x)
#     yy = kernel(y, y)
#     xy = kernel(x, y)
#     # 计算MMD距离
#     mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
#     return mmd

# # 定义高斯径向基函数核
# def gaussian_rbf_kernel(x, y, sigma=1.0):
#     # 计算平方距离
#     dist = torch.cdist(x, y, p=2.0)
#     # 计算高斯径向基函数核
#     return torch.exp(-dist**2 / (2 * sigma**2))

# # 假设我们有两个图像集合，一个是真实图像，另一个是生成的图像
# real_images = ... # 真实图像的张量
# generated_images = ... # 生成图像的张量

# # 获取CLIP嵌入
# real_features = get_clip_embedding(real_images)
# generated_features = get_clip_embedding(generated_images)

# # 计算CMMD
# sigma = 1.0  # 核函数的带宽参数
# mmd = compute_mmd(real_features, generated_features, lambda x, y: gaussian_rbf_kernel(x, y, sigma))
# cmmd = mmd * 1000  # 根据文章中的描述，将MMD值乘以1000进行缩放

# print(f"CMMD: {cmmd.item()}")
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class Thyroid_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))

        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")


    def __len__(self):
        return len(self.df)   
    

    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = self.df["label".strip()].iloc[idx].astype('int')
                subg = self.df["subtype".strip()].iloc[idx]
                if self.transform:
                    image = self.transform(image)

                return (image, label, subg)

class Multi_Thyroid_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))

        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")


    def __len__(self):
        return len(self.df)   
    

    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = self.df["label".strip()].iloc[idx].astype('int')
                subg = self.df["subtype".strip()].iloc[idx]
                if self.transform:
                    image = self.transform(image)

                return (image, label, subg)
                return (image, label, subg)
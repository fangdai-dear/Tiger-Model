import math
import os
import random
import shutil
from pathlib import Path
from pynvml import *
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoTokenizer, PretrainedConfig

tensor1 = torch.tensor([[49406,  1884, 33667,   267, 21263,   268,  1126,   268,  7771,   267,
         32955,   267, 38692,   267, 13989, 43204,   267,  1042, 13989, 49407,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,  1884, 33667,   267, 41122,  3633,   267, 21263,   268,  1126,
           268,  7771,   267,  6148,   267, 32955,   267, 13989, 43204,   267,
          1042, 13989,   267,  1579,  3396,   267,  2442,  1579,  3396, 49407,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,  1884, 33667,   267, 21263,   268,  1126,   268,  7771,   267,
          3143,   267,  6307,   267,  1070,  1042, 13989, 49407,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,  1884, 33667,   267, 21263,   268,  1126,   268,  7771,   267,
         46131,   267,  3143,   267,  6307, 49407,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,  1884, 33667,   267, 21263,   268,  1126,   268,  7771,   267,
          6148,   267, 32955,   267, 38692,   267, 13989, 43204,   267,  1042,
         13989,   267,  1579,  3396,   267,  5094,   268,   789,  1579,  3396,
         49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0],
        [49406,  1884, 33667,   267, 21263,   268,  1126,   268,  7771,   267,
         32955,   267, 38692,  6448, 49407,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0]])


l = tensor1.tolist()
list_2 = sum(l, [])
def remove_item(n):
    return n != 0 and n !=49407 and n!=49406 and n!=267
list_3 = list(filter(remove_item, list_2))

dict = {}
for key in list_3:
    dict[key] = dict.get(key, 0) + 1
print(dict)

revision = None
tokenizer = AutoTokenizer.from_pretrained(
            "/export/home/daifang/Diffusion/diffusers/model/sd-8_28",
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )
# captions = ['papillary blood flow', 'malignant follicular, solid, unclear, irregular, hales, circular, enormous, white point', 'papillary, wider-than-tall, solid, unclear, irregular, echo uneven, low echo, white points, sand-like white points', 'papillary, wider-than-tall, solid, unclear, irregular, echo uneven, extremely low echo, white points, sand-like white points', 'papillary, wider-than-tall, solid, unclear, irregular, echo uneven, low echo', 'papillary, wider-than-tall, solid, unclear, irregular, echo uneven, low echo, white points, sand-like white points']


['papillary, taller-than-wide, solid, unclear, irregular, echo uneven, low echo', 'papillary, wider-than-tall, solid, unclear, irregular, echo uneven, low echo, white points, sand-like white points', 'papillary, wider-than-tall, unclear, irregular, echo uneven, low echo, white points, sand-like white points', 'papillary, taller-than-wide, solid, unclear, irregular, echo uneven, low echo', 'papillary, taller-than-wide, solid, unclear, irregular, echo uneven, low echo, white points, large white points', 'No focus']
inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
print(inputs)

# Tiger Model
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
## Improving AI Models for Rare Thyroid Cancer Subtype by Text Guided Diffusion Models
Fang Dai†, Siqiong Yao†*, Min Wang, Yicheng Zhu, Xiangjun Qiu, Peng Sun, Jisheng Yin, Guangtai Shen7, Jingjing Sun, Maofeng Wang, Yun Wang, Zheyu Yang, Jianfeng Sang, 
Xiaolei Wang, Fenyong Sun*, Wei Cai*, Xingcai Zhang*, Hui Lu*
### Abstract
Artificial intelligence (AI) in oncology imaging struggles with diagnosing rare tumors. Our study identified performance gaps in detecting rare thyroid cancer subtypes using ultrasound, leading to misdiagnoses and adverse prognostic outcomes. Sample scarcity for rare conditions impedes effective model training. Although data augmentation techniques can alleviate sample size constraints, trainable examples cannot encompass the full spectrum of disease manifestations, rendering traditional generative augmentation approaches inadequate. Our approach integrates clinical knowledge with text-image generation, enabling fine-grained control and supplementation of unique features specific to rare subtypes, emphasizing text guidance. This results in augmented samples that more accurately reflect genuine disease cases. Our model, trained on data from 40,571 patients, including 5,099 rare cases, exceeds current state-of-the-art methods, enhancing the AUC for two rare subtypes by 14.64% and 9.45%, respectively. In Turing tests, we achieved 92.2% for authenticity, 90.96% for consistency, and 84.1% for diversity, surpassing competitors by 35.6%. Generalization ability of this methodology was validated on public datasets such as the BrEaST, BUSI, and VinDr-PCXR datasets. This approach mitigates the challenges of data diversity and representativeness for rare diseases, contributing to the model’s generalization ability and diagnostic accuracy, ultimately improving the effectiveness and practical outcomes of medical AI applications.
## Model architecture
The model architecture is included in the manuscript and will not be displayed before the article is published.
## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```
## Datasets
1. Thyroid dataset
```sh
├─Thyroid
    └─DATA
        ├─init_image
            20191101_094744_1.png
            ... ...
            metadata.jsonl
        ├─condition_nd
            20191101_094744_1.png
        ├─condition_bg
            20191101_094744_1.png
```
2. Chexpert chest radiograph multi-classification dataset
```sh
├─CheXpert-v1.0
│  ├─train
│  │  └─patient00001
│  │      └─study1
│  │              view1_frontal.jpg
│  │              
│  └─valid
│      └─patient64541
│          └─study1
│                  view1_frontal.jpg
```
3. ISIC2019 skin disease multi-classification dataset
```     sh              
├─ISIC
│  ├─ISIC_2018
│  │      ISIC_0024306.jpg
│  │      
│  └─ISIC_2019
│          ISIC_0000000.jpg
│          
```
Partial thyroid ultrasonography data used in this study are subject to privacy restrictions, but may be anonymized and made available upon reasonable request to the corresponding author.

## Training data preparation
metadata.josnl
{"file_name": "20191101_094744_1.png", 
"condition_nd": "../DATA/condition_nd/20191101_094744_1.png", 
"condition_bg": "../DATA/condition_bg/20191101_094744_1.png", 
"text_nd": "papillary, wider-than-tall, clear, regular", 
"text_bg": "145.819221, 51.008308, 2.096069"}

```sh
$ sh ./main.sh
```
```sh
├─CSV
│      CXP_female_age.csv
│      CXP_female_race.csv
│      CXP_male_age.csv
│      CXP_male_race.csv
│      CXP_test_age.csv
│      CXP_train_age.csv
│      CXP_train_race.csv
│      CXP_valid_race.csv
│      ISIC_2019_Test.csv
│      ISIC_2019_Training_age.csv
│      ISIC_2019_Training_sex.csv
│      ISIC_2019_valid.csv
```
## Reference
All references are listed in the article

## Licence
The code can be used for non-commercial purposes after the publication of the article. 

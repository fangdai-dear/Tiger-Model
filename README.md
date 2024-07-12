# Tiger Model
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
## Improving AI Models for Rare Thyroid Cancer Subtype by Text Guided Diffusion Models
Fang Dai†, Siqiong Yao†*, Min Wang, Yicheng Zhu, Xiangjun Qiu, Peng Sun, Jisheng Yin, Guangtai Shen7, Jingjing Sun, Maofeng Wang, Yun Wang, Zheyu Yang, Jianfeng Sang, 
Xiaolei Wang, Fenyong Sun*, Wei Cai*, Xingcai Zhang*, Hui Lu*
### Abstract
Artificial intelligence (AI) in oncology imaging struggles with diagnosing rare tumors. Our study identified performance gaps in detecting rare thyroid cancer subtypes using ultrasound, leading to misdiagnoses and adverse prognostic outcomes. Sample scarcity for rare conditions impedes effective model training. Although data augmentation techniques can alleviate sample size constraints, trainable examples cannot encompass the full spectrum of disease manifestations, rendering traditional generative augmentation approaches inadequate. Our approach integrates clinical knowledge with text-image generation, enabling fine-grained control and supplementation of unique features specific to rare subtypes, emphasizing text guidance. This results in augmented samples that more accurately reflect genuine disease cases. Our model, trained on data from 40,571 patients, including 5,099 rare cases, exceeds current state-of-the-art methods, enhancing the AUC for two rare subtypes by 14.64% and 9.45%, respectively. In Turing tests, we achieved 92.2% for authenticity, 90.96% for consistency, and 84.1% for diversity, surpassing competitors by 35.6%. Generalization ability of this methodology was validated on public datasets such as the BrEaST, BUSI, and VinDr-PCXR datasets. This approach mitigates the challenges of data diversity and representativeness for rare diseases, contributing to the model’s generalization ability and diagnostic accuracy, ultimately improving the effectiveness and practical outcomes of medical AI applications.
## Model architecture

## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```

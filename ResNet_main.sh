# coding=utf-8
# Copyright 2024 Hui Lu, Fang Dai, Siqiong Yao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes modifications based on the original implementation from:
# HuggingFace Inc. (2023), Diffusers Library, available at:
# https://github.com/huggingface/diffusers
# The original code is licensed under the Apache License, Version 2.0.


export MODELNAME="Thyroid_Benign&PTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&PTC_train"
export VALID="Benign&PTC_valid"
export TEST="Benign&PTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2



export MODELNAME="Thyroid_Benign&FTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&FTC_train"
export VALID="Benign&FTC_valid"
export TEST="Benign&FTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2

export MODELNAME="Thyroid_Benign&MTC"
export ARCH="resnet"
export IMAGEPATH="./dataset"
export TRAIN="Benign&MTC_train"
export VALID="Benign&MTC_valid"
export TEST="Benign&MTC_test" 
export BATCH=64

python main_train.py \
        --modelname $MODELNAME \
        --architecture $ARCH   \
        --imagepath $IMAGEPATH   \
        --train_data $TRAIN   \
        --valid_data $VALID   \
        --test_data $TEST  \
        --learning_rat 0.0005  \
        --batch_size $BATCH   \
        --num_epochs 200 \
        --Class 2

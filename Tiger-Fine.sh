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

export MODEL_DIR="../modelsaved/Tiger-Corase/checkpoint-xxx"
export OUTPUT_DIR="/modelsaved/Tiger-Fine"
export DATASET="../dataset/training data/init_image"

CUDA_VISIBLE_DEVICES='0,1,2' accelerate launch --multi_gpu --num_processes=3 Tiger Model/Fine-Training.py \
     --pretrained_model_name_or_path=$MODEL_DIR \
     --output_dir=$OUTPUT_DIR \
     --dataset_name=$DATASET \
     --resolution=512 \
     --seed=2023 \
     --learning_rate=1e-5 \
     --gradient_accumulation_steps=4 \
     --checkpointing_steps=200 \
     --max_train_steps=100000000000 \
     --train_batch_size=24 \ 
     --image_column="image" \
     --caption_column_nd="text_nd" --caption_column_bg="text_bd" \
     --conditioning_nd_column="condition_FG" --conditioning_bg_column="condition_BG" \
     --validation_image "../valid_Figure.png" \
     --validation_prompt "malignant papillary solid ..." \
     --validation_steps=500

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

export MODEL_NAME="../sdmodel/v2-1" # In this study, Stable diffusion v2-1 is used as the pre-training model
export DATASET_NAME="../dataset/thyroid_image/training_data"
export PROMPT="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape" # Prompt to verify the effect of model generation

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --num_processes=8 Tiger Model/Coarse-Training.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --caption_column="text" \
    --resolution=512 \
    --random_flip \
    --gradient_accumulation_steps=4 \
    --train_batch_size=24 \
    --num_train_epochs=4000 \
    --checkpointing_steps=1000 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --seed=2024 \
    --output_dir="./modelsaved/Tiger-Corase" \ # Model saving
    --validation_epochs=1 \
    --validation_prompt=$PROMPT
    --report_to="wandb"  
# For the description of parameter meanings, see Appendix

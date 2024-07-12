# export MODEL_NAME="/export/home/daifang/Diffusion/own_code/stable-diffusion-2-1"
# export DATASET_NAME="/export/home/daifang/Diffusion/own_code/dataset"
# export PROMPT="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape, solid, white points, clear, regular, uneven"
# # huggingface-cli login

# accelerate launch --dynamo_backend=no train_text_to_image_lora.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --dataset_name=$DATASET_NAME \
#     --caption_column="text" \?
#     --resolution=224 \
#     --gradient_accumulation_steps=1 \
#     --train_batch_size=1 \
#     --num_train_epochs=1 \
#     --checkpointing_steps=10 \
#     --learning_rate=1e-04 \
#     --lr_scheduler="constant" \
#     --lr_warmup_steps=1 \
#     --seed=2029 \
#     --output_dir="./modelsaved/write" \
#     --validation_prompt="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape, solid, white points, clear, regular, uneven" 
# # 
# 
# --multi_gpu --num_processes=2 
export MODEL_DIR="/export/home/daifang/Diffusion/diffusers/model/sd-8_28"
export OUTPUT_DIR="/export/home/daifang/Diffusion/own_code/modelsaved/control_try"
export DATASET="/export/home/daifang/Diffusion/own_code/dataset/Allclass/init_image"
# huggingface-cli login
# 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  fusing/fill50k $DATASET
accelerate launch /export/home/daifang/Diffusion/own_code/train_controlnet_nd.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=$DATASET \
 --resolution=512 \
 --seed=2023 \
 --learning_rate=1e-5 \
 --gradient_accumulation_steps=5 \
 --checkpointing_steps=2000 \
 --max_train_steps=100\
 --train_batch_size=6 \
 --image_column="image" \
 --caption_column_nd="text_nd" --caption_column_bg="text_bg" \
 --conditioning_nd_column="condition_nd" --conditioning_bg_column="condition_bg" \
 --validation_image "/export/home/daifang/Diffusion/diffusers/examples/controlnet/image/A_1960.png" \
 --validation_prompt "malignant papillary solid" \
 --validation_steps=500


# export MODEL_DIR="/export/home/daifang/Diffusion/diffusers/model/sd-8_28"
# export OUTPUT_DIR="/export/home/daifang/Diffusion/own_code/modelsaved/control_nd_single"
# export DATASET="/export/home/daifang/Diffusion/own_code/dataset/Allclass/init_image"

# CUDA_VISIBLE_DEVICES=7 accelerate launch /export/home/daifang/Diffusion/own_code/train_controlnet_s.py \
#     --pretrained_model_name_or_path=$MODEL_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --dataset_name=$DATASET \
#     --resolution=512 --seed=2023 \
#     --learning_rate=1e-5 \
#     --gradient_accumulation_steps=4 \
#     --checkpointing_steps=200 --max_train_steps=100000000000 --train_batch_size=4 \
#     --image_column="image" \
#     --caption_column_nd="text_nd" --caption_column_bg="text_bd" \
#     --conditioning_nd_column="condition_nd" --conditioning_bg_column="condition_bg" \
#     --validation_image "/export/home/daifang/Diffusion/diffusers/examples/controlnet/image/A_1960.png" \
#     --validation_prompt "malignant papillary solid" \
#     --validation_steps=500
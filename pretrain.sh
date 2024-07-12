export MODEL_NAME="/export/home/daifang/Diffusion/diffusers/model/v2-1"
export DATASET_NAME="/export/home/daifang/Diffusion/diffusers/dataset/thyroid_image/ALLclass"
export DATASET_NAME="/export/home/daifang/Diffusion/diffusers/dataset/thyroid_image/ALLclass"
export PROMPT="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape"

CUDA_VISIBLE_DEVICES='0,1,2' accelerate launch --multi_gpu --num_processes=3 ./example/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --caption_column="text" \
    --resolution=512 \
    --random_flip \
    --gradient_accumulation_steps=4 \
    --train_batch_size=8 \
    --num_train_epochs=600 \
    --checkpointing_steps=1000 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=500 \
    --seed=2029 \
    --output_dir="./modelsaved/sd-pretrain" \
    --validation_epochs=1 \
    --validation_prompt="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape" 
    # --report_to="wandb"
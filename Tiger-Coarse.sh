export MODEL_NAME="../sdmodel/v2-1" # In this study, Stable diffusion v2-1 is used as the pre-training model
export DATASET_NAME="../dataset/thyroid_image/training_data"
export PROMPT="ultrasound of papillary thyroid carcinoma, malignancy, wider-than-tall, shape" # Prompt to verify the effect of model generation

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --num_processes=8 ./Tiger-Model/train_text_to_image_lora.py \
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

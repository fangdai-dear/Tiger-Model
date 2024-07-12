# export MODEL_DIR="../modelsaved/OUTPUT_DIR-Corase"
# export OUTPUT_DIR="/modelsaved/OUTPUT_DIR-Fine"
# export DATASET="../dataset/training data/init_image"

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
#     --validation_image "../A_1960.png" \
#     --validation_prompt "malignant papillary solid" \
#     --validation_steps=500

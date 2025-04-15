#!/bin/bash

# Ensure we fail on any error
set -e

# Check if torch detects GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Base model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Checkpoint directory to resume from
MODEL_CKPT="/content/qwen2.5VL-R1/output/video_lora/checkpoint-5"

# Export Python path to include local src folder
export PYTHONPATH=src:$PYTHONPATH

# Training params
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=8
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Run GRPO trainer
python src/training/train_grpo.py \
    --model_id $MODEL_NAME \
    --model_ckpt $MODEL_CKPT \
    --data_path /content/qwen2.5VL-R1/data/synthetic_videos/train.json \
    --image_folder /content/qwen2.5VL-R1/data/synthetic_videos/videos \
    --output_dir output/grpo_video_lora \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --bf16 True \
    --fp16 False \
    --freeze_llm True \
    --freeze_vision_tower True \
    --tune_merger False \
    --lora_enable True \
    --vision_lora True \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['lm_head','embed_tokens']" \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --dataloader_num_workers 2

#scripts/finetune_video_lora.sh
#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=5
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path /content/qwen2.5VL-R1/data/synthetic_videos/train.json \
    --image_folder /content/qwen2.5VL-R1/data/synthetic_videos/videos \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --lora_enable True \
    --vision_lora True \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['lm_head','embed_tokens']" \
    --disable_flash_attn2 False \
    --output_dir output/video_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((64 * 64)) \
    --fps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --dataloader_num_workers 2

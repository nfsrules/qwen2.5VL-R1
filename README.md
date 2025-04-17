# Qwen2.5VL-R1: Video Action Recognition with Reinforcement Learning

🚀 **Qwen2.5VL-R1** is a project for fine-tuning Qwen2.5-VL on a synthetic video classification task, optimized for a single-GPU setup. It includes a complete pipeline for generating synthetic video data, applying augmentations, fine-tuning with LoRA, and running inference.

---

## 🧠 Objective

This project demonstrates fine-tuning a multimodal large language model (MLLM), specifically `Qwen2.5-VL-3B-Instruct`, for a simple video classification task (similar to Kinetics-400). The task involves classifying the direction of a moving ball in synthetic videos into one of four classes:

- Left to Right  
- Right to Left  
- Falling Down  
- Ascending

---

## 🔧 Setup

### Prerequisites

- OS: Ubuntu 20.04
- Python: 3.11.12 
- GPU: NVIDIA > 16GB (Tested with A100-SXM4-40GB) and CUDA 12.4 
- Dependencies: Listed in `requirements.txt` (install with `pip install -r requirements.txt`)


### Installation

Clone the repository:

```bash
git clone https://github.com/yourname/qwen2.5VL-R1.git
cd qwen2.5VL-R1
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📼 Generate Synthetic Dataset

Generate synthetic videos of a moving ball with corresponding labels and optional support for:

-  Video ugmentations (blur and crop) with default probability of 0.2
-  CoT (Chain-of-Thought) thinking generation for reasoning models (Optional, the code will work in simple mode removing the --cot flag)

```bash
# Optional: Enable Chain-of-Thought generation
export OPENAI_API_KEY=your-openai-api-key
```

```bash
python video_generator.py \
  --output_dir ./data/synthetic_videos \
  --num_samples 20 \
  --cot \  # or remove for no CoT generation
  --frame_size 64 \
  --video_length 30 \
  --split 0.8 \
  --augment_prob 0.2 \
  --augment blur,crop
```

- Output: Saves videos in `data/synthetic_videos/videos/` and metadata in `train.json` and `val.json`.  
- Note: If you do not have an OpenAI api key you can directly **[Download the CoT dataset](https://drive.google.com/drive/folders/1t_vJBkh1sPne_Qd-xirkPEFkZLwoq3Fc?usp=drive_link)**

- Video example: 

![Example](assets/ball-animation.gif)

---

## 🧪 Fine-Tuning

Once generated the training dataset you can fine-tune `Qwen2.5-VL-3B-Instruct` using LoRA for efficiency.
The pipeline leverages DeepSpeed ZeRO-2 for GPU memory optimization.

### SFT LoRA Fine-Tuning (Supervised Finetuning)

```bash
PYTHONPATH=src:$PYTHONPATH \
deepspeed src/training/train.py \
    --use_liger True \
    --deepspeed ./scripts/zero2_offload.json \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path ./data/synthetic_videos/train.json \
    --image_folder ./data/synthetic_videos/videos \
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
    --disable_flash_attn2 True \
    --output_dir output/video_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 5 \
    --video_max_pixels 4096 \
    --fps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --dataloader_num_workers 2
```

- Metrics: Training loss and accuracy are logged every step to Wandb.  

### GRPO Post-training (RL for reasoning)

We also provide a GRPO (Group Relative Policy Optimization) training script with rewards for derivating a reasoning version of the model.


```bash
PYTHONPATH=src:$PYTHONPATH \
python src/training/train_grpo.py \
    --model_id Qwen/Qwen2.5-VL-3B-Instruct \
    --model_ckpt ./output/video_lora/checkpoint-19 \ # 
    --data_path ./data/synthetic_videos/train.json \ # Put here the correct ckp after SFT!!
    --image_folder ./data/synthetic_videos/videos \
    --output_dir output/grpo_video_lora \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --disable_flash_attn2 True \
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

```

- Output: Checkpoints are saved in `./output/video_lora/` or `./output/grpo_video_lora/`.  
- Metrics: Training loss and accuracy are logged every step to Wandb.  
- Optimization: Uses bf16 precision, gradient checkpointing, and DeepSpeed ZeRO-2 for memory optimization.

---

## 🧠 Inference Demo

Test the fine-tuned model on a video:

```bash
python scripts/demo.py \
  --model_ckpt ./output/video_lora/checkcheckpoint-25 \ # Find the right checkcheckpoint path after finetuning
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --video_path ./data/synthetic_videos/videos/000.mp4 \ # Find the right video path
  --prompt "In which direction is the ball moving?\nOptions:\n(A) Left to Right\n(B) Right to Left\n(C) Falling Down\n(D) Ascending" \
  --fps 1.0
```

- Output: Prints the model’s prediction, including reasoning steps (if trained with CoT) and the final answer.


### Credits

- Base model documentation: [Transformers - Qwen2.5-VL](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_5_vl.md)  
  > 📌 Caveat: Supports **video inference**, but **not video training**.

- Fine-tuning code adapted from:  
  - [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)  
  - [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) *(includes only for full FT, not PEFT)*

- GRPO approach inspired by:  
  - [Gemma3 (1B) - GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb)  
  - [lll6gg/UI-R1](https://github.com/lll6gg/UI-R1/tree/main) 
  > 📌 Caveat: Do not support **videos**

---

## 🗂️ Directory Structure

```
qwen2.5VL-R1/
├── README.md
├── requirements.txt
├── video_generator.py
├── data/
│   └── synthetic_videos/
│       ├── train.json
│       ├── val.json
│       └── videos/
├── scripts/
│   ├── demo.py
│   └── zero2_offload.json
└── src/
    └── training/
        ├── __init__.py
        ├── constants.py
        ├── data.py
        ├── modality_patch.py
        ├── params.py
        ├── rewards.py
        ├── train.py
        ├── train_grpo.py
        ├── train_utils.py
        └── trainer.py
```

# Qwen2.5VL-R1: Video Classification Fine-Tuning

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

- OS: Ubuntu 20.04 or later  
- Python: 3.8+  
- GPU: Single NVIDIA GPU (T4, A100, etc.) with CUDA 11.7+  
- Dependencies: Listed in `requirements.txt`

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

-  CoT (Chain-of-Thought) thinking generation for reasoning models 
-  Video ugmentations (blur and crop) with default probability of 0.2


```bash

# Optional: Enable Chain-of-Thought generation
export OPENAI_API_KEY=your-openai-api-key

python video_generator.py \
  --output_dir ./data/synthetic_videos \
  --num_samples 20 \
  --cot \
  --frame_size 64 \
  --video_length 30 \
  --split 0.8 \
  --augment_prob 0.2 \
  --augment blur,crop
```

- Output: Saves videos in `data/synthetic_videos/videos/` and metadata in `train.json` and `val.json`.  
- Note: If you do not have an OpenAI api key you can directly **[Download the CoT dataset](https://drive.google.com/drive/folders/1t_vJBkh1sPne_Qd-xirkPEFkZLwoq3Fc?usp=drive_link)**

---

## 🧪 Fine-Tuning

Once generated the training dataset you can fine-tune `Qwen2.5-VL-3B-Instruct` using LoRA for efficiency.
The pipeline leverages DeepSpeed ZeRO-2 for GPU memory optimization.

### Regular LoRA Fine-Tuning

```bash
python scripts/run_finetuning.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path ./data/synthetic_videos/train.json \
  --image_folder ./data/synthetic_videos/videos \
  --output_dir ./output/video_lora \
  --num_train_epochs 1 \
  --batch_size 1 \
  --global_batch_size 5 \
  --num_devices 1 \
  --fps 1.0 \
  --video_max_pixels 4096 \
  --lr 2e-4 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05
```

### GRPO Post-training (For reasoning models)

For advanced users, GRPO (Gradient-based Reward Policy Optimization) with rewards fine-tuning is available:

```bash
python scripts/run_finetuning.py \
  --use_grpo True \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path ./data/synthetic_videos/train.json \
  --image_folder ./data/synthetic_videos/videos \
  --output_dir ./output/grpo_video_lora \
  --num_train_epochs 1 \
  --batch_size 1 \
  --global_batch_size 5 \
  --num_devices 1 \
  --fps 1.0 \
  --video_max_pixels 4096 \
  --lr 2e-4 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05
```

- Output: Checkpoints are saved in `./output/video_lora/` or `./output/grpo_video_lora/`.  
- Metrics: Training loss and accuracy are logged every step to TensorBoard.  
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


**Optional: Conda**

```bash
conda env create -f environment.yaml
conda activate qwen2.5VL-R1
```

**Optional: Docker**

```bash
docker build -t qwen2.5vl-r1 .
docker run -it --gpus all qwen2.5vl-r1 bash
```


---

## 🗂️ Directory Structure

```
qwen2.5VL-R1/
├── README.md
├── Dockerfile
├── environment.yaml
├── requirements.txt
├── video_generator.py
├── .dockerignore
├── data/
│   └── synthetic_videos/
│       ├── train.json
│       ├── val.json
│       └── videos/
├── scripts/
│   ├── demo.py
│   ├── run_finetuning.py
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

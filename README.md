# Qwen2.5VL-R1

🚀 **Qwen2.5VL-R1** is an optimized variant of [Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL) tailored for **video classification with reasoning** using Chain-of-Thought (CoT) prompting and visual grounding.

---

## 🧠 Key Features

- 📹 **Video Reasoning Dataset Generator**  
  Create synthetic motion-based video clips labeled with direction + reasoning steps using GPT-4V.

- 🧪 **LoRA Fine-tuning Support**  
  Lightweight fine-tuning of Qwen2.5-VL with visual adapters using [DeepSpeed ZeRO-2](https://www.deepspeed.ai/tutorials/zero/).

- 🧠 **Multimodal Reasoning Engine**  
  Custom forward methods for mixed visual/textual tokens and temporal CoT-style reasoning.

---

## 🗂️ Directory Structure

```
qwen2.5VL-R1/
├── README.md
├── environment.yaml
├── requirements.txt
├── video_generator.py
├── scripts/
│   ├── run_training.py
│   ├── demo.py
│   └── zero2_offload.json
└── src/
    └── training/
        ├── train.py
        ├── train_grpo.py
        ├── data.py
        ├── trainer.py
        ├── params.py
        ├── modality_patch.py
        └── constants.py
```

---

## 🔧 Setup

```bash
# Clone repo
git clone https://github.com/yourname/qwen2.5VL-R1.git
cd qwen2.5VL-R1

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📼 Generate Dataset

```bash
python video_generator.py \
  --output_dir ./data/synthetic_videos \
  --num_samples 20 \
  --cot \
  --frame_size 64 \
  --video_length 30 \
  --split 0.8
```

> ⚠️ Requires `OPENAI_API_KEY` in your environment if `--cot` is enabled.

---

## 🧪 Fine-tuning (LoRA)

Run the unified training CLI with the following options:

### ▶️ Regular LoRA fine-tuning

```bash
python scripts/run_finetuning.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path ./data/synthetic_videos/train.json \
  --image_folder ./data/synthetic_videos/videos \
  --output_dir ./output/video_lora
```

### ▶️ GRPO fine-tuning

```bash
python scripts/run_training.py \
  --use_grpo True \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --data_path ./data/synthetic_videos/train.json \
  --image_folder ./data/synthetic_videos/videos \
  --output_dir ./output/grpo_video_lora
```

You can tweak additional args like `--batch_size`, `--lr`, `--lora_rank`, `--fps`, and `--video_max_pixels`.

---

## 🧠 Inference Demo

```bash
python scripts/demo.py \
  --video_path ./data/synthetic_videos/videos/000.mp4 \
  --prompt "What is happening in this video?"
```

---

## 📎 Notes

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Chain-of-Thought reasoning uses `<think>` and `<answer>` tags
- Supports DeepSpeed + FlashAttention for scalable training

---

## 📚 Credits

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba DAMO
- PEFT + LoRA via [Hugging Face PEFT](https://github.com/huggingface/peft)

---

## 🛡 License

MIT (or inherit from base model — update as appropriate)

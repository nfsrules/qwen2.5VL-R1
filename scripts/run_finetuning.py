import argparse
import subprocess
import os


def run_lora_finetune(args):
    os.environ["PYTHONPATH"] = "src:" + os.environ.get("PYTHONPATH", "")
    grad_accum = args.global_batch_size // (args.batch_size * args.num_devices)

    if args.use_grpo:
        # GRPO fine-tuning
        cmd = [
            "python", "src/training/train_grpo.py",
            "--model_id", args.model_id,
            "--data_path", args.data_path,
            "--image_folder", args.image_folder,
            "--output_dir", args.output_dir,
            "--per_device_train_batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(grad_accum),
            "--num_train_epochs", str(args.num_train_epochs),
            "--learning_rate", str(args.lr),
            "--bf16", "True",
            "--fp16", "False",
            "--freeze_llm", "True",
            "--freeze_vision_tower", "True",
            "--tune_merger", "False",
            "--lora_enable", "True",
            "--vision_lora", "True",
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--num_lora_modules", "-1",
            "--lora_namespan_exclude", "['lm_head','embed_tokens']",
            "--gradient_checkpointing", "True",
            "--logging_steps", "1",
            "--dataloader_num_workers", "2",
        ]
    else:
        # Regular Deepspeed + LoRA fine-tuning
        cmd = [
            "deepspeed", "src/training/train.py",
            "--use_liger", "True",
            "--deepspeed", "scripts/zero2_offload.json",
            "--model_id", args.model_id,
            "--data_path", args.data_path,
            "--image_folder", args.image_folder,
            "--remove_unused_columns", "False",
            "--freeze_vision_tower", "True",
            "--freeze_llm", "True",
            "--tune_merger", "False",
            "--bf16", "True",
            "--lora_enable", "True",
            "--vision_lora", "True",
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--num_lora_modules", "-1",
            "--lora_namespan_exclude", "['lm_head','embed_tokens']",
            "--disable_flash_attn2", "False",
            "--output_dir", args.output_dir,
            "--num_train_epochs", str(args.num_train_epochs),
            "--per_device_train_batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(grad_accum),
            "--video_max_pixels", str(args.video_max_pixels),
            "--fps", str(args.fps),
            "--learning_rate", str(args.lr),
            "--weight_decay", "0.0",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--tf32", "True",
            "--gradient_checkpointing", "True",
            "--report_to", "tensorboard",
            "--lazy_preprocess", "True",
            "--save_strategy", "steps",
            "--save_steps", "5",
            "--save_total_limit", "2",
            "--dataloader_num_workers", "2",
        ]

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5VL-R1 LoRA fine-tuning")

    parser.add_argument("--use_grpo", action="store_true", help="Enable GRPO fine-tuning instead of standard LoRA")

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=5)
    parser.add_argument("--num_devices", type=int, default=1)

    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--video_max_pixels", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    run_lora_finetune(args)


if __name__ == "__main__":
    main()

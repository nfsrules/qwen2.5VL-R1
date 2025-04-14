# /scripts/demo.py

import argparse
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def load_pretrained_model(model_path, model_base, device_map="auto"):
    # Load base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_base, device_map=device_map, torch_dtype="auto"
    )

    # Load local LoRA adapter weights
    model = PeftModel.from_pretrained(
        model, model_path, is_trainable=False, local_files_only=True
    )

    # Merge LoRA weights into base model (optional but preferred for inference)
    model = model.merge_and_unload()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_base)

    return processor, model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_pretrained_model(
        model_path=args.model_ckpt,
        model_base=args.base_model,
        device_map=device,
    )
    model.eval()

    # Prepare conversation input
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video_path, "fps": args.fps},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Convert to generation prompt
    prompt_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # Process video inputs
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            do_sample=(args.temperature > 0),
            eos_token_id=processor.tokenizer.eos_token_id
        )

    output_text = processor.tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    print("\n========== MODEL OUTPUT ==========")
    print(output_text)
    print("==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/content/Qwen2-VL-Finetune/output/video_lora/checkpoint-25",
        help="Path to the LoRA fine-tuned checkpoint directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base pretrained model name or local path.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file (e.g. /content/000.mp4).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In which direction is the ball moving?\nOptions:\n(A) Left to Right\n(B) Right to Left\n(C) Falling Down\n(D) Ascending",
        help="Prompt to ask about the video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for sampling.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    args = parser.parse_args()
    main(args)

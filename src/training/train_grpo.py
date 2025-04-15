import torch
import ast
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    HfArgumentParser,
)
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.data import make_supervised_data_module
from rewards import (
    check_answer as reward_correct_answer,
)
from liger_kernel.transformers import (
    apply_liger_kernel_to_qwen2_5_vl,
)
from modality_patch import (
    replace_qwen2_5_with_mixed_modality_forward,
)
from training.train_utils import (
    find_target_linear_names,
    set_requires_grad
)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "Qwen2.5" in model_args.model_id:
        replace_qwen2_5_with_mixed_modality_forward(use_liger=training_args.use_liger)
        if training_args.use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        raise ValueError("Unsupported model type. Only Qwen2.5 is supported.")

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_args = {}
    if training_args.bits in [4, 8]:
        bnb_args.update(
            {
                "device_map": {"": training_args.device},
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            }
        )

    # Use --model_ckpt if provided
    model_path = model_args.model_ckpt or model_args.model_id
    if model_args.model_ckpt:
        print(f"[INFO] Overriding model_id with checkpoint: {model_args.model_ckpt}")

    model_arch_hint = model_args.model_id or model_args.model_ckpt or ""
    ModelClass = (
        Qwen2_5_VLForConditionalGeneration
        if "Qwen2.5" in model_arch_hint
        else Qwen2VLForConditionalGeneration
    )

    model = ModelClass.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
        ),
        **bnb_args,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        padding_side="right",
        use_fast=True
    )

    model.config.use_cache = False
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    # Apply freezing
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)

    if hasattr(model.visual, "merger"):
        set_requires_grad(model.visual.merger.parameters(), training_args.tune_merger)

    # Apply LoRA if enabled
    if training_args.lora_enable:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(
                training_args.lora_namespan_exclude
            )
        else:
            training_args.lora_namespan_exclude = []
        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                num_lora_modules=training_args.num_lora_modules,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        model = get_peft_model(model, peft_config)

    data_module = make_supervised_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    grpo_args = GRPOConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        logging_steps=training_args.logging_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module.get("eval_dataset"),
        processing_class=processor.tokenizer,
        reward_funcs=[
            reward_correct_answer
        ],
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()

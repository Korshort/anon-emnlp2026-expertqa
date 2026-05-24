"""SFT training script — launched via accelerate.

Loads a causal LM, applies LoRA/QLoRA, and fine-tunes with TRL SFTTrainer.
All parameters are received via argparse (injected by train.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# ── Thinking model handling ──────────────────────────────────
#
# Qwen3 등 thinking 모델은 apply_chat_template 시 assistant 응답 앞에
# <think>\n\n</think>\n\n 을 자동 삽입한다 (빈 thinking = non-thinking 모드).
# 이를 제거하려고 chat_template을 교체하거나 후처리해도,
# TRL SFTTrainer가 내부적으로 tokenizer를 재사용하여 다시 삽입된다.
#
# 해결: 싸우지 말고 맞춰준다.
# thinking 모델의 response_template에 빈 <think> 태그를 포함시켜서
# DataCollatorForCompletionOnlyLM이 정상 매칭하도록 한다.
# <think> 태그 자체는 loss에서 제외되고, 실제 응답만 학습된다.
#
# 향후 다른 thinking 모델 추가 시:
# 1. 해당 모델의 chat_template을 분석하여 thinking 태그 구조 확인
# 2. Qwen3과 동일 구조(<think>...</think>)면 아래 로직 그대로 적용
# 3. 다른 구조면 별도 핸들러 추가


# thinking 모델별 빈 thinking 블록 패턴.
# response_template 앞에 이 패턴이 삽입되므로, response_template에 포함시켜야 한다.
_THINKING_MODEL_PATTERNS: dict[str, str] = {
    "qwen3": "<think>\n\n</think>\n\n",
    # 향후 추가 예: "deepseek-r1": "<think>\n\n</think>\n\n",
}


def _get_thinking_pattern(model_id: str) -> str | None:
    """모델이 thinking 모델이면 빈 thinking 블록 패턴을 반환, 아니면 None."""
    model_lower = model_id.lower()
    for key, pattern in _THINKING_MODEL_PATTERNS.items():
        if key in model_lower:
            return pattern
    return None


# ── CLI argument parsing ─────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT fine-tuning with LoRA/QLoRA")

    # Required
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Training hyperparameters
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--micro_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default=None,
                   help="Comma-separated target modules (e.g. q_proj,k_proj,v_proj)")

    # LoRA / QLoRA
    p.add_argument("--use_lora", type=lambda x: x.lower() != 'false', default=True,
                   help="Enable LoRA (default True). Set to false for full fine-tuning.")
    p.add_argument("--use_qlora", action="store_true", default=False)
    p.add_argument("--optim", type=str, default=None,
                   help="Optimizer override (e.g. adamw_bnb_8bit). Defaults: paged_adamw_8bit (QLoRA) / adamw_torch (FT).")

    # Model loading
    p.add_argument("--trust_remote_code", action="store_true", default=False)

    # DeepSpeed config (optional, used with torchrun for multi-GPU)
    p.add_argument("--deepspeed", type=str, default=None,
                   help="Path to DeepSpeed config JSON. Forwarded to SFTConfig.")

    # Data
    p.add_argument("--response_template", type=str, default=None,
                   help="Response template string for completion-only training")

    # Eval
    p.add_argument("--eval_dataset_path", type=str, default=None,
                   help="Path to eval JSONL (e.g. golden set). If omitted, no eval.")

    return p.parse_args()


# ── Dataset loading ──────────────────────────────────────────


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a JSONL file with 'messages' field into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed JSON at line {line_num}: {e}",
                      file=sys.stderr)
                continue
            if "messages" not in record:
                print(f"WARNING: Skipping line {line_num}: missing 'messages' field",
                      file=sys.stderr)
                continue
            # Keep only 'messages' — drop non-training fields (meta, etc.)
            records.append({"messages": record["messages"]})

    if not records:
        raise ValueError(f"No valid records found in {path}")

    print(f"Loaded {len(records)} examples from {path}")
    return Dataset.from_list(records)


# ── Model + Tokenizer loading ────────────────────────────────


def load_model_and_tokenizer(
    model_id: str,
    use_qlora: bool,
    trust_remote_code: bool,
) -> tuple:
    """Load the base model and tokenizer. Apply quantization config if QLoRA."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Detect best attention implementation
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("flash_attn not available, using SDPA attention.", file=sys.stderr)

    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_impl,
    }

    # 260502: device_map 분기
    # - torchrun multi-GPU + QLoRA: LOCAL_RANK 기반 (각 process가 자기 GPU)
    # - 단일 process + QLoRA + multi-GPU 환경: device_map="auto" (model을 layer 단위로 GPU에 분산)
    # - Full FT + ZeRO-3: device_map 생략 (DeepSpeed가 sharding)
    local_rank = os.environ.get("LOCAL_RANK")
    if use_qlora:
        if local_rank is not None:
            model_kwargs["device_map"] = {"": int(local_rank)}
        elif torch.cuda.device_count() > 1:
            # 단일 process + multi-GPU 환경 (e.g., 70B QLoRA): naive model parallelism
            model_kwargs["device_map"] = "auto"

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config

    # EXAONE 등 일부 모델은 rope_theta가 config 최상위에 없고
    # rope_scaling dict 안에만 있어 transformers 5.x 검증에서 실패한다.
    # config를 먼저 로딩하여 rope_theta를 최상위로 복사한 뒤 모델에 전달한다.
    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code,
    )
    if not hasattr(config, "rope_theta") or config.rope_theta is None:
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        if "rope_theta" in rope_scaling:
            config.rope_theta = rope_scaling["rope_theta"]

    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, **model_kwargs)

    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    return model, tokenizer


# ── LoRA configuration ───────────────────────────────────────


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Build a PEFT LoraConfig from parsed arguments."""
    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


# ── Training arguments ───────────────────────────────────────


def build_training_args(args: argparse.Namespace) -> SFTConfig:
    """Build TRL SFTConfig from parsed arguments."""
    if args.optim:
        optimizer = args.optim
    else:
        optimizer = "paged_adamw_8bit" if args.use_qlora else "adamw_torch"

    eval_kwargs = {}
    if args.eval_dataset_path:
        eval_kwargs["eval_strategy"] = "steps"
        eval_kwargs["eval_steps"] = args.logging_steps

    if args.deepspeed:
        eval_kwargs["deepspeed"] = args.deepspeed

    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        lr_scheduler_type="cosine",
        optim=optimizer,
        max_grad_norm=1.0,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        max_length=args.max_seq_len,
        packing=False,
        **eval_kwargs,
    )


# ── Gemma3 token_type_ids handling ────────────────────────────
#
# Gemma3 (ForConditionalGeneration)은 VLM으로, 학습 시 token_type_ids를
# 필수로 요구한다. Text-only SFT에서는 모든 토큰이 텍스트이므로
# token_type_ids = 0 으로 채운다.
# data collator를 래핑하여 배치에 token_type_ids를 자동 추가한다.


def _model_needs_token_type_ids(model) -> bool:
    """모델이 token_type_ids를 필요로 하는지 검사."""
    config = getattr(model, "config", None)
    if config is None:
        config = getattr(model, "base_model", None)
        if config is not None:
            config = getattr(config, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", "")
    return model_type in ("gemma3",)


class _TokenTypeIdsCollatorWrapper:
    """기존 collator를 감싸서 token_type_ids=0을 배치에 추가."""

    def __init__(self, inner_collator):
        self.inner_collator = inner_collator

    def __call__(self, features, **kwargs):
        batch = self.inner_collator(features, **kwargs)
        if "token_type_ids" not in batch and "input_ids" in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


# ── Formatting function ──────────────────────────────────────


def format_chat(example: dict, tokenizer: AutoTokenizer) -> str:
    """Apply the tokenizer's chat template to a messages list."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    print(f"=== run_sft.py ===")
    print(f"  Model:       {args.model_id}")
    print(f"  Dataset:     {args.dataset_path}")
    print(f"  Output:      {args.output_dir}")
    print(f"  LoRA:        {args.use_lora}")
    print(f"  QLoRA:       {args.use_qlora}")
    print(f"  LoRA r/a:    {args.lora_r}/{args.lora_alpha}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"==================")

    # Load dataset
    dataset = load_jsonl_dataset(args.dataset_path)

    # Load eval dataset (golden set)
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_jsonl_dataset(args.eval_dataset_path)
        print(f"  Eval dataset: {args.eval_dataset_path} ({len(eval_dataset)} examples)")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        use_qlora=args.use_qlora,
        trust_remote_code=args.trust_remote_code,
    )

    # Build LoRA config and apply (skip if full fine-tuning)
    if args.use_lora:
        lora_config = build_lora_config(args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Build training arguments
    training_args = build_training_args(args)

    # response_template의 리터럴 \n을 실제 줄바꿈으로 치환.
    # 셸 환경변수(.env)에서 작은따옴표로 감싼 값은 \n이 리터럴로 전달되므로
    # Python에서 실제 줄바꿈 문자로 변환해야 한다.
    response_template = args.response_template
    if response_template:
        response_template = response_template.replace("\\n", "\n")
    thinking_pattern = _get_thinking_pattern(args.model_id)
    if thinking_pattern and response_template:
        response_template = response_template + thinking_pattern
        print(f"  ⚠ Thinking model detected ({args.model_id}).")
        print(f"    response_template adjusted: {repr(response_template)}")

    # Gemma3 등 VLM 모델의 text-only SFT 대응:
    # 1. token_type_ids 자동 추가 (텍스트=0)
    # 2. 비전 파라미터 미사용으로 인한 DDP 에러 방지
    collator = None
    needs_token_type_ids = _model_needs_token_type_ids(model)
    if needs_token_type_ids:
        from transformers import DataCollatorForLanguageModeling
        base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        collator = _TokenTypeIdsCollatorWrapper(base_collator)
        training_args.ddp_find_unused_parameters = True
        print("  ⚠ VLM text-only mode: token_type_ids collator + find_unused_parameters enabled.")

    # Build formatting function
    def formatting_func(example: dict) -> str:
        return format_chat(example, tokenizer)

    # Initialize SFTTrainer
    trainer_kwargs = {}
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        **trainer_kwargs,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save final model (LoRA adapter weights)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")

    print("Training complete.")


if __name__ == "__main__":
    main()

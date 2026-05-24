"""run.sh generator — 모델키/조건/도메인 → run.sh 자동 생성.

원칙:
  - configs/models.yaml + training.yaml + vram_estimates.yaml 정본 참조
  - CUDA_VISIBLE_DEVICES 하드코딩 금지 (trainq가 GPU 자동 할당)
  - response_template/target_modules/deepspeed_stage 자동 삽입
  - supports_system_role: false → main_gemma/ 데이터 경로 자동 선택
  - Llama 70B 같이 use_qlora: false 모델은 USE_QLORA=false 자동 처리

Usage:
  python scripts/generate_run.py \
      --model exaone3.5-7.8b \
      --condition exp0 \
      --domain medical \
      --experiment main \
      --out-dir .claude/progress/260430_main_sft_training_evaluation/runs

  또는 일괄 생성:
  python scripts/generate_run.py \
      --batch wave1 \
      --experiment main \
      --out-dir .claude/progress/260430_main_sft_training_evaluation/runs
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

DIVERGENCE_PROTOCOL_NOTE = """\
# ── Run divergence protocol ─────────────────────────────────
# loss 발산 판정: NaN 발생 또는 5% 상승 유지(3 epoch 누적)
#   1차 대응: 재시도 1회 + lr 0.5× (LEARNING_RATE 환경변수로 override)
#   2차 실패: 해당 cell 제외, statistics/sft_main_results.csv에 NaN 기록
# 출처: 260430_main_sft_training_evaluation.md flow_4 §4단계 운영 프로토콜
"""


# 모델 크기 → batch tier
TIER_BATCH_MAP = {
    "small": "small",
    "medium": "medium_small",  # 7-9B 기본. 27B는 별도 처리.
    "large": "large",
}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _batch_key_for(model_cfg: dict) -> str:
    tier = model_cfg.get("tier", "small")
    size = model_cfg.get("size", "")
    # 22~27B는 medium_large
    if tier == "medium" and size and any(s in size for s in ["22", "27"]):
        return "medium_large"
    return TIER_BATCH_MAP.get(tier, "small")


def _resolve_dataset_path(condition: str, domain: str, experiment: str, supports_system_role: bool) -> str:
    """experiment + condition + domain → 데이터 경로. Gemma 분기 포함."""
    base = f"paper_works/1.expert_qa/data/training"
    if experiment.startswith("whisper"):
        # Wave 4 — placeholder
        return f"{base}/whisper/{condition}_{domain}.jsonl"
    folder = experiment if supports_system_role else f"{experiment}_gemma"
    return f"{base}/{folder}/{condition}_{domain}.jsonl"


def _approx_qa_count(condition: str, domain: str) -> int:
    """train_main 8 JSONL 평균. line count 측정 필요 시 별도 스크립트."""
    counts = {
        ("exp0", "medical"): 1411, ("exp0", "finance"): 1252,
        ("exp1_1", "medical"): 1468, ("exp1_1", "finance"): 1297,
        ("exp1_2", "medical"): 1402, ("exp1_2", "finance"): 1270,
        ("exp2", "medical"): 1376, ("exp2", "finance"): 1222,
    }
    return counts.get((condition, domain), 1300)


def _estimated_hours(model_cfg: dict, training_cfg: dict, vram_cfg: dict, qa_count: int, num_gpus: int = 4) -> float:
    """vram_estimates.yaml의 sec_per_step + tier batch + epoch 기반 추정."""
    batch_key = _batch_key_for(model_cfg)
    batch = training_cfg["training"]["batch"][batch_key]
    micro = batch["micro_batch"]
    accum = batch["grad_accum"]
    effective_batch = micro * accum * num_gpus
    epochs = training_cfg["training"].get("num_train_epochs", 3)
    total_steps = math.ceil(qa_count / effective_batch) * epochs

    sec_per_step_map = vram_cfg["sec_per_step"]["l40s"]
    tier = model_cfg.get("tier", "small")
    size = model_cfg.get("size", "")
    if tier == "medium" and any(s in size for s in ["22", "27"]):
        sec = sec_per_step_map["medium_large"]
    elif tier == "large":
        sec = sec_per_step_map["large"]
    elif tier == "medium":
        sec = sec_per_step_map["medium_small"]
    else:
        sec = sec_per_step_map["small"]
    return (total_steps * sec) / 3600.0


def _vram_mb(model_cfg: dict, vram_cfg: dict, model_key: str) -> int:
    gb = vram_cfg["vram"].get(model_key)
    if gb is None:
        # fallback by tier
        tier_default = {"small": 12, "medium": 24, "large": 160}
        gb = tier_default.get(model_cfg.get("tier", "small"), 16)
    return int(gb * 1024)


def build_run_sh(
    model_key: str,
    model_cfg: dict,
    condition: str,
    domain: str,
    experiment: str,
    training_cfg: dict,
    vram_cfg: dict,
    out_dir: Path,
    placeholder: bool = False,
    full_ft: bool = False,
) -> Path:
    run_id = f"{model_key}_{condition}_{domain}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_sh = run_dir / "run.sh"

    supports_system_role = model_cfg.get("supports_system_role", True)
    use_qlora = model_cfg.get("use_qlora", True)  # default true. 70B는 false.
    deepspeed_stage = model_cfg.get("deepspeed_stage", 2)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    target_modules = ",".join(model_cfg.get("target_modules", []))
    response_template = model_cfg.get("response_template", "")
    hf_id = model_cfg["hf_id"]
    conda_env = model_cfg.get("conda_env")  # None이면 train.sh default(nmt) 사용

    # Full Fine-Tuning mode: override LoRA settings
    use_lora = not full_ft
    if full_ft:
        use_qlora = False
        # 7.8B 이상은 stage 3, 그 외는 stage 2 (260501_full_ft_sanity_check.md §3 기준)
        size = model_cfg.get("size", "")
        try:
            params_b = float(re.findall(r"\d+\.?\d*", size or "")[0]) if size else 0
        except Exception:
            params_b = 0
        deepspeed_stage = 3 if params_b >= 7 else 2

    batch_key = _batch_key_for(model_cfg)
    batch = training_cfg["training"]["batch"][batch_key]
    micro_batch = batch["micro_batch"]
    grad_accum = batch["grad_accum"]
    epochs = training_cfg["training"].get("num_train_epochs", 3)
    lr = training_cfg["training"].get("learning_rate", 2.0e-4)
    # Full FT 전용 hyperparameter override (260501_full_ft_sanity_check.md §3)
    if full_ft:
        lr = 1.0e-5  # 1e-5 ~ 2e-5 권고. 발산 방지를 위해 보수적 1e-5.
        if params_b >= 7:
            micro_batch = 1
            grad_accum = 16
        else:
            micro_batch = 2
            grad_accum = 8
    max_seq_len = training_cfg["training"].get("max_seq_len", 2048)
    lora_r = training_cfg["training"]["lora"]["r"]
    lora_alpha = training_cfg["training"]["lora"]["alpha"]
    lora_dropout = training_cfg["training"]["lora"]["dropout"]
    warmup_ratio = training_cfg["training"].get("warmup_ratio", 0.03)
    logging_steps = training_cfg["training"].get("logging_steps", 10)

    # Full FT sanity는 main 데이터를 그대로 쓰지만, 출력 경로는 별도 디렉토리.
    data_experiment = "main" if full_ft else experiment
    dataset_path = _resolve_dataset_path(condition, domain, data_experiment, supports_system_role)
    output_dir_remote = f"/mnt/data/runs/{experiment}_sft/{run_id}"

    placeholder_note = ""
    if placeholder:
        placeholder_note = (
            "# [PLACEHOLDER] Whisper 데이터 경로 미확정 — G1 통과 후 경로 업데이트 필요\n"
            "# 본 run.sh는 G1 PASS 전까지 trainq 큐잉 금지\n"
        )

    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by scripts/generate_run.py — do not hand-edit.",
        f"# run_id: {run_id}",
        f"# experiment: {experiment} / condition: {condition} / domain: {domain}",
        f"# model: {model_key} ({hf_id})",
        placeholder_note,
        DIVERGENCE_PROTOCOL_NOTE,
        "set -euo pipefail",
        "",
        "# Required env (consumed by scripts/train.sh)",
        *([f'export CONDA_ENV="{conda_env}"'] if conda_env else []),
        f'export MODEL_ID="{hf_id}"',
        f'export DATASET_PATH="{dataset_path}"',
        f'export OUTPUT_DIR="{output_dir_remote}"',
        "",
        "# Hyperparameters",
        f'export NUM_EPOCHS="{epochs}"',
        f'export MAX_SEQ_LEN="{max_seq_len}"',
        f'export MICRO_BATCH="{micro_batch}"',
        f'export GRAD_ACCUM="{grad_accum}"',
        f'export LEARNING_RATE="{lr}"',
        f'export WARMUP_RATIO="{warmup_ratio}"',
        f'export LOGGING_STEPS="{logging_steps}"',
        f'export LORA_R="{lora_r}"',
        f'export LORA_ALPHA="{lora_alpha}"',
        f'export LORA_DROPOUT="{lora_dropout}"',
        f'export TARGET_MODULES="{target_modules}"',
        f'export RESPONSE_TEMPLATE="{response_template}"',
        f'export DEEPSPEED_STAGE="{deepspeed_stage}"',
        f'export USE_LORA="{str(use_lora).lower()}"',
        f'export USE_QLORA="{str(use_qlora).lower()}"',
        f'export TRUST_REMOTE_CODE="{str(trust_remote_code).lower()}"',
        f'export NUM_GPUS="${{NUM_GPUS:-4}}"',
        f'export SEED="${{SEED:-42}}"',
        "",
        "# CUDA_VISIBLE_DEVICES is set by trainq automatically — DO NOT hardcode here.",
        "",
        "# Resolve repo root (script is under .claude/progress/.../runs/{run_id}/)",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        '# train.sh on remote: /mnt/data/runs/main_sft/scripts/train.sh',
        'TRAIN_SH="${TRAIN_SH:-/mnt/data/runs/' + experiment + '_sft/scripts/train.sh}"',
        'if [[ ! -f "${TRAIN_SH}" ]]; then',
        '  echo "ERROR: train.sh not found at ${TRAIN_SH}" >&2',
        '  exit 1',
        'fi',
        "",
        'bash "${TRAIN_SH}"',
    ]

    # placeholder note는 첫 줄 다음에 두기 위해 위에서 이미 삽입됨
    content = "\n".join(line for line in lines if line is not None) + "\n"
    run_sh.write_text(content, encoding="utf-8")
    run_sh.chmod(0o755)
    return run_sh


def dry_run_validate(run_sh: Path, dataset_path: str, expect_dataset_exists: bool) -> tuple[bool, list[str]]:
    """기본 형식 + 경로 존재 검증."""
    issues = []
    text = run_sh.read_text()
    if "CUDA_VISIBLE_DEVICES=" in text and "DO NOT hardcode" not in text.split("CUDA_VISIBLE_DEVICES=", 1)[0][-80:]:
        # naive check — not perfect, but flags accidental hardcoding
        # 더 엄격하게 lhs assignment 검사
        for ln in text.splitlines():
            stripped = ln.strip()
            if stripped.startswith("export CUDA_VISIBLE_DEVICES=") or stripped.startswith("CUDA_VISIBLE_DEVICES="):
                issues.append("CUDA_VISIBLE_DEVICES hardcoded")
                break
    if "MODEL_ID=" not in text:
        issues.append("MODEL_ID export missing")
    if "DATASET_PATH=" not in text:
        issues.append("DATASET_PATH export missing")
    if "OUTPUT_DIR=" not in text:
        issues.append("OUTPUT_DIR export missing")
    if expect_dataset_exists:
        full = REPO_ROOT / dataset_path
        if not full.exists():
            issues.append(f"dataset not found: {dataset_path}")
    return (len(issues) == 0, issues)


# ── Batch presets ──────────────────────────────────────────

WAVE1_MODELS = ["exaone3.5-7.8b", "qwen3.5-9b"]
WAVE2_MODELS = ["exaone3.5-2.4b", "gemma3-4b", "llama3.2-3b", "phi4-mini", "qwen3.5-4b", "gemma3-27b"]
WAVE3_MODELS = ["llama3.3-70b"]
WAVE4_MODELS = ["exaone3.5-7.8b", "qwen3.5-9b"]

CONDITIONS_MAIN = ["exp0", "exp1_1", "exp1_2", "exp2"]
CONDITIONS_WHISPER = ["W-Exp0", "W-Exp2"]
DOMAINS = ["medical", "finance"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="single model key (or use --batch)")
    p.add_argument("--condition", help="single condition (or use --batch)")
    p.add_argument("--domain", help="single domain (or use --batch)")
    p.add_argument("--batch", choices=["wave1", "wave2", "wave3", "wave4"], help="batch preset")
    p.add_argument("--experiment", default="main", help="main / whisper")
    p.add_argument("--out-dir", required=True, help="output dir for runs/{run_id}/run.sh")
    p.add_argument("--validate", action="store_true", help="dry-run validate after generation")
    p.add_argument("--full-ft", action="store_true", help="Enable full fine-tuning (disable LoRA)")
    args = p.parse_args()

    models_cfg = _load_yaml(REPO_ROOT / "configs" / "models.yaml")["models"]
    training_cfg = _load_yaml(REPO_ROOT / "configs" / "training.yaml")
    vram_cfg = _load_yaml(REPO_ROOT / "configs" / "vram_estimates.yaml")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []  # list of (model_key, cond, dom, experiment, placeholder)
    if args.batch:
        if args.batch == "wave1":
            for m in WAVE1_MODELS:
                for c in CONDITIONS_MAIN:
                    for d in DOMAINS:
                        targets.append((m, c, d, "main", False))
        elif args.batch == "wave2":
            for m in WAVE2_MODELS:
                for c in CONDITIONS_MAIN:
                    for d in DOMAINS:
                        targets.append((m, c, d, "main", False))
        elif args.batch == "wave3":
            for m in WAVE3_MODELS:
                for c in CONDITIONS_MAIN:
                    for d in DOMAINS:
                        targets.append((m, c, d, "main", False))
        elif args.batch == "wave4":
            for m in WAVE4_MODELS:
                for c in CONDITIONS_WHISPER:
                    for d in DOMAINS:
                        targets.append((m, c, d, "whisper", True))
    else:
        if not (args.model and args.condition and args.domain):
            print("ERROR: --batch or (--model --condition --domain) required", file=sys.stderr)
            sys.exit(2)
        targets.append((args.model, args.condition, args.domain, args.experiment, False))

    pass_count = 0
    fail_items = []
    for model_key, cond, dom, exp_name, placeholder in targets:
        if model_key not in models_cfg:
            fail_items.append(f"{model_key}_{cond}_{dom}: model not in models.yaml")
            continue
        mcfg = models_cfg[model_key]
        run_sh = build_run_sh(
            model_key=model_key,
            model_cfg=mcfg,
            condition=cond,
            domain=dom,
            experiment=exp_name,
            training_cfg=training_cfg,
            vram_cfg=vram_cfg,
            out_dir=out_dir,
            placeholder=placeholder,
            full_ft=args.full_ft,
        )
        supports = mcfg.get("supports_system_role", True)
        # Full FT sanity는 main 데이터 사용 (build_run_sh와 동일 분기)
        ds_exp = "main" if args.full_ft else exp_name
        ds_path = _resolve_dataset_path(cond, dom, ds_exp, supports)
        ok, issues = dry_run_validate(run_sh, ds_path, expect_dataset_exists=not placeholder)
        if ok:
            pass_count += 1
            try:
                rel = run_sh.relative_to(REPO_ROOT)
            except ValueError:
                rel = run_sh
            print(f"  OK  {model_key}_{cond}_{dom} → {rel}")
        else:
            fail_items.append(f"{model_key}_{cond}_{dom}: {issues}")
            print(f"  FAIL {model_key}_{cond}_{dom}: {issues}")

    print(f"\nGenerated {pass_count}/{len(targets)} run.sh successfully.")
    if fail_items:
        print("Failures:")
        for it in fail_items:
            print(f"  - {it}")
        sys.exit(1)


if __name__ == "__main__":
    main()

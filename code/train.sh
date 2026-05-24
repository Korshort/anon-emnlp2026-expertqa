#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SFT Training Launch Script
# Wraps accelerate launch + DeepSpeed for QLoRA/LoRA fine-tuning.
# All parameters are injected via environment variables.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 0. Conda environment activation ──────────────────────────
CONDA_ENV="${CONDA_ENV:-nmt}"
CONDA_BASE="${CONDA_BASE:-/mnt/data/miniconda3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "Activated conda env: ${CONDA_ENV} (python: $(which python))"
  # HF 캐시를 현재 유저의 캐시로 강제 설정 (/etc/environment의 공유 캐시 권한 문제 방지)
  export HF_HOME="${HOME}/.cache/huggingface"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  echo "  HF_HOME=${HF_HOME}"
else
  echo "WARNING: conda not found at ${CONDA_BASE}. Using system python." >&2
fi

# ── 1. Required environment variables ───────────────────────
for var in MODEL_ID DATASET_PATH OUTPUT_DIR; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: Required environment variable ${var} is not set." >&2
    exit 1
  fi
done

# ── 2. Validate paths ───────────────────────────────────────
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ── 3. Defaults for optional variables ───────────────────────
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NUM_GPUS="${NUM_GPUS:-4}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-2}"
USE_LORA="${USE_LORA:-true}"
USE_QLORA="${USE_QLORA:-true}"
SEED="${SEED:-42}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
TARGET_MODULES="${TARGET_MODULES:-}"
RESPONSE_TEMPLATE="${RESPONSE_TEMPLATE:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-}"
DEEPSPEED_OFFLOAD="${DEEPSPEED_OFFLOAD:-false}"
OPTIM="${OPTIM:-}"

# ── 4. DeepSpeed config selection ────────────────────────────
if [[ "${DEEPSPEED_STAGE}" == "2" ]]; then
  DS_CONFIG="${SCRIPT_DIR}/train_deepspeed_z2.json"
elif [[ "${DEEPSPEED_STAGE}" == "3" ]]; then
  if [[ "${DEEPSPEED_OFFLOAD}" == "true" ]]; then
    DS_CONFIG="${SCRIPT_DIR}/train_deepspeed_z3_offload.json"
  else
    DS_CONFIG="${SCRIPT_DIR}/train_deepspeed_z3.json"
  fi
else
  echo "ERROR: DEEPSPEED_STAGE must be 2 or 3, got: ${DEEPSPEED_STAGE}" >&2
  exit 1
fi

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "ERROR: DeepSpeed config not found: ${DS_CONFIG}" >&2
  exit 1
fi

echo "=== SFT Training ==="
echo "  Model:          ${MODEL_ID}"
echo "  Dataset:        ${DATASET_PATH}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  GPUs:           ${NUM_GPUS}"
echo "  DeepSpeed:      ZeRO-${DEEPSPEED_STAGE}"
echo "  QLoRA:          ${USE_QLORA}"
echo "  Epochs:         ${NUM_EPOCHS}"
echo "  Micro batch:    ${MICRO_BATCH}"
echo "  Grad accum:     ${GRAD_ACCUM}"
echo "  LR:             ${LEARNING_RATE}"
echo "  LoRA r/alpha:   ${LORA_R}/${LORA_ALPHA}"
echo "  Max seq len:    ${MAX_SEQ_LEN}"
echo "  Seed:           ${SEED}"
echo "===================="

# ── 5. HuggingFace cache check — download model if needed ────
echo "Checking model cache for ${MODEL_ID} ..."
export MODEL_ID TRUST_REMOTE_CODE
CACHE_CHECK=$(python -c '
import os, sys
from huggingface_hub import snapshot_download
model_id = os.environ["MODEL_ID"]
try:
    snapshot_download(model_id, local_files_only=True)
    print("CACHED")
except Exception:
    print("NOT_CACHED")
' 2>/dev/null)

if [[ "${CACHE_CHECK}" == "CACHED" ]]; then
  echo "Model cache found."
else
  echo "Model not cached. Downloading ${MODEL_ID} ..."
  python -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"])
print("Download complete.")
'
fi

# PyTorch CUDA allocator: fragmentation 줄여 marginal OOM 회피
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── 6. Build launch command ──────────────────────────────────
# 260502: launcher 선택 로직
# - QLoRA + multi-GPU: ZeRO-3가 bnb 4-bit weights 분산 못 함 → single process + device_map=auto (model parallelism)
# - Full FT + multi-GPU: torchrun + ZeRO-3 (params/grad/optimizer 분산)
# - Single GPU: python 단일 process
USE_QLORA="${USE_QLORA:-false}"
if [[ "${NUM_GPUS}" -gt 1 ]] && [[ "${USE_QLORA}" != "true" ]]; then
  CMD=(
    torchrun
    --nproc_per_node "${NUM_GPUS}"
    --nnodes 1
    --master_port "$((29500 + RANDOM % 10000))"
    "${SCRIPT_DIR}/run_sft.py"
    --deepspeed "${DS_CONFIG}"
  )
else
  CMD=(
    python
    "${SCRIPT_DIR}/run_sft.py"
  )
fi
CMD+=(
  --model_id "${MODEL_ID}"
  --dataset_path "${DATASET_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_epochs "${NUM_EPOCHS}"
  --max_seq_len "${MAX_SEQ_LEN}"
  --micro_batch "${MICRO_BATCH}"
  --grad_accum "${GRAD_ACCUM}"
  --learning_rate "${LEARNING_RATE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --seed "${SEED}"
  --warmup_ratio "${WARMUP_RATIO}"
  --logging_steps "${LOGGING_STEPS}"
)

if [[ "${USE_LORA}" == "true" ]]; then
  CMD+=(--use_lora true)
else
  CMD+=(--use_lora false)
fi

if [[ "${USE_QLORA}" == "true" ]]; then
  CMD+=(--use_qlora)
fi

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  CMD+=(--trust_remote_code)
fi

if [[ -n "${TARGET_MODULES}" ]]; then
  CMD+=(--target_modules "${TARGET_MODULES}")
fi

if [[ -n "${RESPONSE_TEMPLATE}" ]]; then
  CMD+=(--response_template "${RESPONSE_TEMPLATE}")
fi

if [[ -n "${EVAL_DATASET_PATH}" ]]; then
  CMD+=(--eval_dataset_path "${EVAL_DATASET_PATH}")
fi

if [[ -n "${OPTIM}" ]]; then
  CMD+=(--optim "${OPTIM}")
fi

# ── 7. Execute with tee to log ───────────────────────────────
# DeepSpeed 런처가 CUDA_VISIBLE_DEVICES를 리셋할 수 있으므로
# 실행 직전에 다시 export 한다.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

LOG_FILE="${OUTPUT_DIR}/train.log"
echo "Logging to: ${LOG_FILE}"
echo "Command: ${CMD[*]}"

set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "Training completed successfully."
else
  echo "ERROR: Training failed with exit code ${EXIT_CODE}." >&2
fi

exit ${EXIT_CODE}

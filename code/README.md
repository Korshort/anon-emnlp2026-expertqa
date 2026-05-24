# Code

## Training

| File | Purpose |
|---|---|
| `run_sft.py` | TRL `SFTTrainer` wrapper. LoRA / QLoRA / Full FT supported. PEFT + DeepSpeed integrated. |
| `train.sh` | Bash wrapper: conda env activation + DeepSpeed config selection (`DEEPSPEED_STAGE`, `DEEPSPEED_OFFLOAD`) + `torchrun` for multi-GPU full FT. |
| `train_deepspeed_z2.json` / `z3.json` / `z3_offload.json` | DeepSpeed configs. ZeRO-2 (LoRA), ZeRO-3 (Full FT), ZeRO-3 + CPU offload (memory-constrained). |
| `generate_run.py` | Auto-generate per-cell `run.sh` from `configs/models.yaml` and condition/domain. |

## Evaluation

| File | Purpose |
|---|---|
| `lm_eval_runner.py` | Wrapper around `lm_eval_harness` CLI with PEFT monkey-patch (Exaone tied-embedding workaround). |

## Result aggregation & analysis

| File | Purpose |
|---|---|
| `collect_eval.py` | KMMLU/MMLU aggregate from raw `results_*.json` |
| `collect_domain_eval.py` | Per-domain target subset aggregate (KMMLU-Med/Fin, MMLU-Med/Fin) |
| `collect_full_ft.py` | Full FT result aggregate |
| `collect_kmmlu_subtask.py` | KMMLU per-sub-task raw acc |
| `analyze_kmmlu_subtask.py` | Sub-task factorial effect analysis (A, B, Int) |

## Whisper sanity (Appendix on WER)

| File | Purpose |
|---|---|
| `g1_whisper_asr.py` | Whisper-medium ASR for the WER reproduction sub-sample |
| `g1_wer.py` | WER computation against the in-house transcript reference |

## Sample QA / Judge utilities

| File | Purpose |
|---|---|
| `select_public_samples.py` | Stratified sampling of 200-QA public sample from raw pipeline output |
| `prepare_judge_batches.py` | Randomize and cell-blind 200-QA for LLM-as-judge batch calls |
| `aggregate_judge.py` | Aggregate per-batch judge responses into per-condition/domain means |

## Usage

```bash
# 1. Generate per-cell run.sh
python generate_run.py --batch wave1 --experiment main --out-dir runs/

# 2. Train (single cell)
bash runs/qwen3.5-9b_exp2_medical/run.sh

# 3. Eval (lm_eval task)
lm_eval --model hf --model_args pretrained=...,peft=... --tasks kmmlu,mmlu

# 4. Aggregate results
python collect_domain_eval.py > results.json

# 5. Analyze
python analyze_kmmlu_subtask.py

# 6. LLM-as-judge pipeline (5-dim Likert, see paper §5.1)
python prepare_judge_batches.py --sample-dir ../sample_qa --out-dir ./judge_batches
# (Run 4-judge cross-provider scoring externally; place results as
#  judge_batches/result_batch_{1..8}.json with the same item ids.)
python aggregate_judge.py --batch-dir ./judge_batches
```

Source data (Korean medical/finance interview transcripts) and Phase 1 generation prompts are not included; see top-level `README.md` Limitations.

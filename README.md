# A Factorial Ablation of a Speech-to-SFT Pipeline: Diverging Effects on Data Quality and Downstream Transfer

> Reproducibility package for the paper above (EMNLP 2026 Industry Track, under review).

## What is included

| Folder | Contents |
|---|---|
| `code/` | Training scripts (`run_sft.py`, `train.sh`), evaluation runners (`lm_eval_runner.py` w/ PEFT tied-emb wrapper for ExaoneModel), analysis scripts, sample selection (`select_public_samples.py`), judge batch preparation (`prepare_judge_batches.py`), judge aggregation (`aggregate_judge.py`), Whisper G1 reproduction (`g1_*.py`) |
| `configs/` | Model configs (`models.yaml`), training hyperparameters (`training.yaml`), DeepSpeed configs |
| `prompts/` | Evaluation rubrics (`llm_judge_rubric.md`, `binary_judge_rubric.md`, `unknown_admission_instruction.md`); pipeline LLM disclosure manifest (`pipeline_llm_disclosure.md`); Phase 2 pipeline prompts in `prompts/pipeline/` (judge, refine, dedup) |
| `results/` | Per-cell aggregate evaluation results (KMMLU/MMLU per-subtask acc, Full-FT comparisons, factorial Δ) + per-item judge raw scores. See \"Results layout\" below. |
| `sample_qa/` | **200-QA primary public sample** — PII-filtered, no Phase-2 filter (the main 200-QA used for the §5.1 4-judge evaluation and the §5.4 LLM-difficulty audit) |
| `sample_qa_filtered/` | **200-QA Phase-2-filtered variant** — PII-filtered, Phase-2 `status=passed` filter applied uniformly across the four conditions (sampling-design sensitivity check, Appendix N) |

## What we disclose

All model identifiers across the entire pipeline are public. For Phase 2 — the quality-refinement stage that constitutes the measurement instrument of our ablation — we additionally release the verbatim judge and refine prompts in `prompts/pipeline/`.

**Model identifiers** (full manifest in `prompts/pipeline_llm_disclosure.md`):

- **Phase 0 discourse_summary / NER / proofread**: `claude-sonnet-4` / `mimo-v2-flash` / `claude-haiku-4-5`. Multi-STT cross-validation: `gpt-4o-transcribe` + ElevenLabs Scribe.
- **Phase 1 analyze_document / context_augment / strategize / generate**: `claude-haiku-4-5` / `claude-haiku-4-5` / `claude-sonnet-4` / `claude-sonnet-4` + `mimo-v2-pro` (generator ensemble).
- **Phase 2 judge / refine**: `gpt-5.2` (5-judge ensemble) / `claude-haiku-4-5` (≤2 iterations). Dedup: multilingual sentence-embedding cosine similarity.
- **Evaluation judges** (paper §5.1 + §5.4): `claude-sonnet-4-6`, `claude-opus-4-7`, `gpt-4o`, `gpt-5.4`.

**Verbatim prompts** released in this repository:

| Prompt | Used in | File |
|---|---|---|
| 4-judge 6-dim Likert rubric (QA data quality) | paper §5.1 | `prompts/llm_judge_rubric.md` |
| 2-judge binary rubric (unknown-admission re-classification) | paper §5.4 | `prompts/binary_judge_rubric.md` |
| Unknown-admission answer instruction (LLM-difficulty audit) | paper §5.4 | `prompts/unknown_admission_instruction.md` |
| Phase 2 judge prompt | paper §3.4 | `prompts/pipeline/phase2_judge.md` |
| Phase 2 refine prompt | paper §3.4 | `prompts/pipeline/phase2_refine.md` |
| Phase 2 dedup algorithm specification | paper §3.4 | `prompts/pipeline/phase2_dedup.md` |

Phase 0 and Phase 1 tool prompts (including `generate`) are not released. Model identifiers are listed in `prompts/pipeline_llm_disclosure.md`; per-stage costs are reported in paper §6.2. Per-tool functional descriptions and I/O schemas are not separately documented.

## Public sample schema

Each item in `sample_qa/` and `sample_qa_filtered/` exposes only seven fields: `question`, `answer`, `language`, `domain`, `sub_domain`, `difficulty`, `question_type`. PII (personal names, specific corporate/institutional identifiers) has been replaced with neutral placeholders (e.g. `[증권사 A]`, `[해외 대학 메디컬 센터]`).

## Results layout

```
results/
├── kmmlu_mmlu_lora_target.json    — §5.2 main 72 LoRA cells (KMMLU/MMLU domain-aligned)
├── kmmlu_mmlu_full_ft.json        — §5.5 + App K Full-FT sanity (12 cells)
├── kmmlu_per_subtask.json         — App B KMMLU 45-subject breakdown
├── llm_judge_aggregate.json       — Sonnet 4.6 single-judge per-cell aggregate over the 5 external-quality dimensions (paper Table 3 is the 4-judge mean; compute by aggregating across the four sources listed below using code/aggregate_judge.py)
├── judge_primary_batch_{1..8}.json — primary 200-QA per-item Sonnet 4.6 scores (8 batches × 25 items)
├── judge_filtered/                — Phase-2-filtered 200-QA 4-judge per-item raw (App N)
│   ├── sonnet_46/sonnet_batch_{1..8}.json
│   ├── opus_47/opus_batch_{1..8}.json
│   ├── gpt_4o/gpt_4o_batch_{1..8}.json
│   └── gpt_5_4/gpt_5_4_batch_{1..8}.json
└── w_grid_judge/                  — §5.3 Table 4 W-grid 4-judge raw (W-Exp0/W-Exp2 × med+fin × 25 items, plus in-house baseline aggregate); see w_grid_judge/README.md
```

## What is NOT included (Limitations)

The following are proprietary corpus and cannot be redistributed:

- **Source audio** (40 conference recordings under speaker-consent agreements)
- **ASR transcripts** of the source audio (institution-internal + Whisper-medium reproduction)
- **Generated QA pairs** (10,698 items; only the 200-QA primary + 200-QA Phase-2-filtered samples are released)
- **Phase 0 institution-internal STT system weights** (vendor-specific)
- **Phase 0 and Phase 1 tool prompts** (verbatim prompts, per-tool functional descriptions, and I/O schemas are not released). Model identifiers and per-stage costs are disclosed. Phase 2 prompts are released in `prompts/pipeline/`.

These are protected by speaker-consent agreements and corporate IP. External replication on user-provided audio is supported end-to-end via the released code, configs, and disclosed LLM identifiers.

## Reproducibility scope

| Element | Reproducible | How |
|---|---|---|
| **Trained model weights** | ✓ | 172 SFT runs (LoRA 72 main + LoRA 8 W-grid + LoRA 4 Whisper sanity + Full FT 12 + auxiliary) public on Hugging Face under `Korshort/emnlp2026_expertqa_*` ([collection](https://huggingface.co/collections/Korshort/expert-qa-pipeline-emnlp-2026-industry-anonymous-6a12aadc9e442908e29b48c4); upload in progress, anonymous account for review). Includes `final/` + intermediate `checkpoint-N/`. |
| **Eval results numerical** | ✓ | `results/` per-cell `result_batch_*.json` (judge raw scores) + `kmmlu_mmlu_*.json` (MCQA). |
| **Statistical analysis** | ✓ | `code/aggregate_judge.py`, `code/analyze_kmmlu_subtask.py`, `code/collect_*.py` reproduce all factorial tables and Δ values from raw results. |
| **LLM-judge protocol** | ✓ | `prompts/llm_judge_rubric.md` + `prompts/binary_judge_rubric.md` + `prompts/unknown_admission_instruction.md` + `code/prepare_judge_batches.py` + `code/aggregate_judge.py`. |
| **Training pipeline** | ✓ | `code/run_sft.py` + `code/generate_run.py` + `configs/` + `code/train.sh`; trained weights included. |
| **Exact data reproduction** | ✗ | Source corpus proprietary. Methodology fully described to enable application on user-provided Korean medical/finance audio. |

## Citation

```bibtex
[paper bibtex placeholder — fill after acceptance]
```

## License

- Code: Apache-2.0
- Trained model weights: inherit base model licenses
  - EXAONE 3.5: NC (research-only, see EXAONE-3.5 license)
  - Llama 3.2/3.3: Llama Community License
  - Gemma 3: Gemma License
  - Qwen 3.5: Apache-2.0
  - Phi-4 Mini: MIT

## Contact

[author info placeholder]

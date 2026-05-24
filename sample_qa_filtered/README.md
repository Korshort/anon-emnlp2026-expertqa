# Sample QA (Public Release)

## Overview

Representative QA samples (25 per condition × 4 conditions × 2 domains = **200 QA**) from the main experiment training data, released for inspection of pipeline output quality across ablation conditions.

## Files

| File | Condition | Domain | n |
|---|---|---|---|
| `exp0_medical.jsonl` | Phase 0 OFF, Phase 2 OFF (raw) | medical | 25 |
| `exp1_1_medical.jsonl` | Phase 0 ON, Phase 2 OFF | medical | 25 |
| `exp1_2_medical.jsonl` | Phase 0 OFF, Phase 2 ON | medical | 25 |
| `exp2_medical.jsonl` | Phase 0 ON, Phase 2 ON (full) | medical | 25 |
| `exp0_finance.jsonl` ~ `exp2_finance.jsonl` | (same conditions) | finance | 25 each |

## Selection criteria

QA items are selected from the full training set (10,698 items) using:
1. **PII filter**: items containing personal identifiers (phone, RRN, email, hospital/company names, patient/case mentions) are excluded
2. **Quality**: `status: passed` (Phase 2 judge passed) only
3. **Stratification**: balanced across `sub_domain`
4. **Length**: representative of each condition's typical answer richness

The selection does not preserve session-level identifiers (those are tracked internally for copyright clearance).

## Schema

Each line is a JSON object with the following public fields:

| Field | Description |
|---|---|
| `question` | Korean question generated from the source transcript |
| `answer` | Korean answer from the QA Generator pipeline |
| `language` | `ko` (rare `en` for English-language interview segments) |
| `domain` | `Med` or `Fin` |
| `sub_domain` | Sub-domain tag (e.g. `Clinical`, `Banking`) |
| `difficulty` | `easy` / `medium` / `hard` (assigned by QA Generator) |
| `question_type` | `comparative`, `analytical`, `causal`, `explanatory`, etc. |
| `status` | `passed` (Phase 2 judge passed) |
| `source_sentences` | Anchor sentences from the source transcript (truncated for sample) |
| `fact_check_verdict` | (None or judge note) |

Internal fields (`trace_id`, `si_seq`, `session_id`, `_meta`) are stripped.

## License

The QA pairs in this folder are released for academic inspection of pipeline output quality under fair-use research provision. Source audio and full transcripts remain proprietary to the authors' institution. For commercial reuse contact the authors.

## Statistics

See `selection_stats.json`.

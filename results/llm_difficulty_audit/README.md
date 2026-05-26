# LLM-difficulty audit (paper §5.4, Table 6)

Cell-blind 2-judge binary re-classification of 4 frontier LLMs' answers on the
200-QA public sample. Reproduces paper §5.4 Table 6 strict-bound rates.

## Protocol

1. Four frontier LLMs (Claude Sonnet 4.6, Claude Opus 4.7, OpenAI GPT-4o,
   OpenAI GPT-5.4) answer the same 200-QA public sample under an explicit
   non-hallucination instruction (`../../prompts/unknown_admission_instruction.md`).
2. Each (model, item) pair gets an anonymized id `a001`..`a800` (16 batches ×
   50 items). The judge sees only `{question, answer}`; model and condition
   identifiers are withheld at dispatch (cell-blind).
3. Two binary judges from non-Anthropic / non-OpenAI families (Claude Haiku 4.5
   + Gemini 2.5 Pro) label each anonymized answer as `UNKNOWN` or `ANSWERED`
   per `../../prompts/binary_judge_rubric.md`.
4. Per-model strict bound = mean of the two judges' per-model UNKNOWN rates
   (NOT label intersection; the two judges agree on ~96.8% of items so
   intersection runs 1-3pp lower than the per-judge mean).

## Files

| Path | Contents |
|---|---|
| `input_batches/batch_{1..16}.json` | Dispatch inputs (anonymized id + question + answer; no model/cond identifiers). Same content as raw frontier answers, organized anonymously. |
| `haiku45_kr/scores_batch_{1..16}.json` | Claude Haiku 4.5 labels per anonymized item |
| `gemini25pro/scores_batch_{1..16}.json` | Gemini 2.5 Pro labels per anonymized item |
| `sample_with_meta.json` | 200 QA × `{gid, cond, dom, sub_domain}` (no answers) |
| `id_to_model_mapping.json` | anon id (a001..a800) → `{model, gid}`. Used **only** for post-experiment aggregation; never shown to the judges during dispatch (the cell-blind constraint applies to judge inputs, not to reviewer-facing release). |

## Reproducing Table 6

```bash
python ../../code/aggregate_unknown_admission.py \
    --audit-dir .
```

Expected output (matches paper Table 6):

```
## Paper Table 6 — strict bound (2-judge mean), per-model
model       Haiku% Gemini%   avg2%
gpt4o         60.5    61.0    60.8
gpt54         12.5     9.0    10.8
sonnet        10.5     7.0     8.8
opus           6.5     3.0     4.8

4-model strict avg: 21.2%        (paper 21.3%, 0.1pp rounding)
Frontier-2 (Opus + GPT-5.4):  7.8%
Agreement: raw 96.8%, Cohen's κ 0.903
```

## Cell-blind verification

- `id_to_model_mapping.json` is loaded ONLY by post-experiment aggregation
  (`aggregate_unknown_admission.py`).
- `input_batches/batch_*.json` (judge dispatch inputs) contain `id`, `question`,
  `answer` only — no `model`, `cond`, `dom` fields. Verify with:
  ```bash
  python -c "import json,glob; [print({k for r in json.load(open(f))['items'] for k in r}) for f in glob.glob('input_batches/batch_*.json')]"
  ```
- `haiku45_kr/`, `gemini25pro/` outputs contain `id`, `label` only.

## Cochran-Armitage trend test (paper §5.4)

Per-condition rates (19.5%, 24.0%, 19.8%, 21.8%) tested for monotonic trend
across Exp~0 → Exp~1-1 → Exp~1-2 → Exp~2: χ²=0.075, p=0.785 (non-significant),
i.e., no monotonic stage effect on unknown-admission rate.

## Source

This is the same raw used in paper Table 6. The Gemini 3 Pro Preview
cross-check (3-judge robustness, internal-only) is omitted from this release
since paper §5.4 reports only the 2-judge bound.

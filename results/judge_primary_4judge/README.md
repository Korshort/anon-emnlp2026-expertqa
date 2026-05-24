# Primary 200-QA 4-judge per-item raw scores (paper Table 3, Table 16)

4-judge cross-provider Likert scoring on the primary 200-QA public sample
(50 items per condition × 4 conditions = 200 items, 4 judges).

## Files

```
judge_primary_4judge/
├── id_to_cell_mapping.json     — q000~q199 → {cond, dom} (8 cells × 25 items)
├── sonnet_46/sonnet_46_batch_{1..8}.json  — Claude Sonnet 4.6 (renamed from judge_primary_batch_*)
├── opus/exp1_opus_batch_{1..8}.json       — Claude Opus 4.7
├── gpt4o/exp1_gpt4o_batch_{1..8}.json     — OpenAI GPT-4o
└── gpt54/exp1_gpt54_batch_{1..8}.json     — OpenAI GPT-5.4
```

8 batches × 25 items × 6-dim Likert (1-5) per judge.

## Rubric

Same 6-dim rubric as `../../prompts/llm_judge_rubric.md`:
faithfulness, domain_accuracy, question_quality, answer_depth, coherence,
difficulty_calibration. Paper reports the overall as the mean of the first
five external-quality dimensions; `difficulty_calibration` is reported
per-dim only.

## Table 3 / Table 16 reproduction

```python
import json, glob
from collections import defaultdict

DIMS_5 = ["faithfulness", "domain_accuracy", "question_quality", "answer_depth", "coherence"]
m = json.load(open("id_to_cell_mapping.json"))

judges = {
    "sonnet_46": "sonnet_46/sonnet_46_batch_*.json",
    "opus":      "opus/exp1_opus_batch_*.json",
    "gpt4o":     "gpt4o/exp1_gpt4o_batch_*.json",
    "gpt54":     "gpt54/exp1_gpt54_batch_*.json",
}

scores = {j: {} for j in judges}
for j, pattern in judges.items():
    for f in glob.glob(pattern):
        d = json.load(open(f))
        for r in d["results"]:
            scores[j][r["id"]] = r["scores"]

# Table 3 overall (4-judge mean, 5-dim) per condition
for c in ["exp0", "exp1_1", "exp1_2", "exp2"]:
    items = [q for q in m if m[q]["cond"] == c]
    overall = 0
    for j in judges:
        per_judge_avg = sum(sum(scores[j][q][d] for d in DIMS_5)/5 for q in items) / len(items)
        overall += per_judge_avg
    print(f"{c}: {overall/4:.2f}")
```

## Schema note

`sonnet_46/*` files (renamed from `../judge_primary_batch_*.json`) use the
`results` key. Other judge files use the same `results` key for primary
200-QA scoring.

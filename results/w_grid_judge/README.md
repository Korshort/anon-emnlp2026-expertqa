# W-grid judge scores (Table 5 in paper §5.3)

4-judge cross-provider Likert scoring on the W-grid sub-sample
(EXAONE 3.5 7.8B + Qwen 3.5 4B × medical + finance × {W-Exp0, W-Exp2},
25 QA per cell, 4 cells, 100 items total).

## Files

| File | Contents |
|---|---|
| `exp2_{judge}_batch_{1..3}.json` | W-Exp0 medical (batch 1), W-Exp0 finance (batch 2), W-Exp2 finance (batch 3); 25 items each per batch. judge ∈ {sonnet, opus, gpt4o, gpt54}. |
| `w_exp2_medical_{judge}.json` | W-Exp2 medical (25 items each). judge ∈ {sonnet, opus, gpt4o, gpt54}. |
| `w_exp0_and_w_exp2_finance_id_to_cell.json` | id → cell mapping for the 75 items covered by the batch files (`w_exp0_medical`, `w_exp0_finance`, `w_exp2_finance`). |
| `w_exp2_medical_id_to_cell.json` | id → cell mapping for the 25 W-Exp2 medical items. |
| `in_house_baseline_aggregate.json` | 4-judge per-cell means on the matched in-house Exp 0 / Exp 1-1 / Exp 1-2 / Exp 2 sample sessions (the in-house side of Table 5 in paper §5.3). |

## Rubric

Same 5-dim 1-5 Likert rubric as the primary 200-QA scoring (paper §4 Experimental Setup +
`../prompts/llm_judge_rubric.md`). The paper's reported overall is the
mean of the five external-quality dimensions: faithfulness, domain_accuracy,
question_quality, answer_depth, coherence.

## Table 5 reproduction (paper §5.3)

```python
import json, glob
from collections import defaultdict

DIMS = ["faithfulness", "domain_accuracy", "question_quality", "answer_depth", "coherence"]
JUDGES = ["sonnet", "opus", "gpt4o", "gpt54"]

# Load mappings
m1 = json.load(open("w_exp0_and_w_exp2_finance_id_to_cell.json"))
m2 = json.load(open("w_exp2_medical_id_to_cell.json"))

# For each W-cell × judge: mean overall = mean of 5-dim means
# Δ = W cell mean − matched in-house cell mean (in_house_baseline_aggregate.json)
```

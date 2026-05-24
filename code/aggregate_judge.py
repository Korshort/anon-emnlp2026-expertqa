"""LLM-judge 8 batch results → cell × dim aggregate + factorial.

Usage: python aggregate_judge.py [--batch-dir DIR] [--include-difficulty-calibration]
Default: BATCH_DIR=./judge_batches (relative to CWD).

The paper's "overall" metric is the mean of five external-quality dimensions
(faithfulness, domain_accuracy, question_quality, answer_depth, coherence).
The 6th rubric dimension `difficulty_calibration` measures Phase-1 label
alignment with content and is reported per-dim only (not in the overall
aggregate). Pass --include-difficulty-calibration to revert to the legacy
6-dim overall.
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--batch-dir", default=os.environ.get("BATCH_DIR", str(Path.cwd() / "judge_batches")))
parser.add_argument("--include-difficulty-calibration", action="store_true",
                    help="legacy 6-dim overall (default: 5-dim, matches paper Table 3)")
args = parser.parse_args()

BATCH_DIR = Path(args.batch_dir)

# Load mapping
with (BATCH_DIR / "id_to_cell_mapping.json").open() as f:
    mapping = json.load(f)

# Load all results
all_scores = {}  # qid -> {dim: score, ..., rationale}
for i in range(1, 9):
    with (BATCH_DIR / f"result_batch_{i}.json").open() as f:
        d = json.load(f)
    for r in d["results"]:
        all_scores[r["id"]] = r

print(f"Total scored: {len(all_scores)}/200")

DIMS_EXTERNAL = ["faithfulness", "domain_accuracy", "question_quality", "answer_depth", "coherence"]
DIMS = DIMS_EXTERNAL + ["difficulty_calibration"]
OVERALL_DIMS = DIMS if args.include_difficulty_calibration else DIMS_EXTERNAL
CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
DOMS = ["medical", "finance"]

# Aggregate per (cond, dom) × dim
cell_scores = defaultdict(lambda: defaultdict(list))
for qid, info in mapping.items():
    if qid not in all_scores:
        continue
    cond = info["cond"]; dom = info["dom"]
    s = all_scores[qid].get("scores", {})
    for dim in DIMS:
        if dim in s and isinstance(s[dim], (int, float)):
            cell_scores[(cond, dom)][dim].append(s[dim])

# Mean per cell × dim
print("\n## Cell × Dim 평균 (1-5 scale)\n")
header = "| Cond | Dom | overall | " + " | ".join(d[:8] for d in DIMS) + " |"
sep = "|---|---|---:|" + "|".join("---:" for _ in DIMS) + "|"
print(header); print(sep)
cell_dim_avg = {}
for cond in CONDS:
    for dom in DOMS:
        scores = cell_scores[(cond, dom)]
        dim_avgs = {}
        for dim in DIMS:
            vals = scores[dim]
            dim_avgs[dim] = sum(vals)/len(vals) if vals else None
        overall_vals = [dim_avgs[d] for d in OVERALL_DIMS if dim_avgs[d] is not None]
        overall = sum(overall_vals) / len(overall_vals) if overall_vals else None
        cell_dim_avg[(cond, dom)] = (overall, dim_avgs)
        cells = " | ".join(f"{dim_avgs[d]:.2f}" if dim_avgs[d] is not None else "—" for d in DIMS)
        overall_str = f"{overall:.2f}" if overall is not None else "—"
        print(f"| {cond} | {dom} | {overall_str} | {cells} |")

# Factorial per dim
print("\n## Factorial (overall) per dim\n")
print("| dim | A (Phase0) | B (Phase2) | Int | Exp2−Exp0 |")
print("|---|---:|---:|---:|---:|")
for dim in DIMS + ["overall"]:
    A_vals, B_vals, Int_vals, E2E0_vals = [], [], [], []
    for dom in DOMS:
        if dim == "overall":
            v = {c: cell_dim_avg[(c, dom)][0] for c in CONDS}
        else:
            v = {c: cell_dim_avg[(c, dom)][1].get(dim) for c in CONDS}
        if all(v[c] is not None for c in CONDS):
            A = (v["exp1_1"] + v["exp2"])/2 - (v["exp0"] + v["exp1_2"])/2
            B = (v["exp1_2"] + v["exp2"])/2 - (v["exp0"] + v["exp1_1"])/2
            Int = v["exp2"] - v["exp1_1"] - v["exp1_2"] + v["exp0"]
            E2E0 = v["exp2"] - v["exp0"]
            A_vals.append(A); B_vals.append(B); Int_vals.append(Int); E2E0_vals.append(E2E0)
    if A_vals:
        n = len(A_vals)
        print(f"| {dim} | {sum(A_vals)/n:+.2f} | {sum(B_vals)/n:+.2f} | {sum(Int_vals)/n:+.2f} | {sum(E2E0_vals)/n:+.2f} |")

# Save aggregate JSON
with (BATCH_DIR / "aggregate.json").open("w") as f:
    json.dump({
        "n_scored": len(all_scores),
        "overall_dims": OVERALL_DIMS,
        "cell_dim_avg": {f"{c}|{d}": {"overall": cell_dim_avg[(c,d)][0], "dims": cell_dim_avg[(c,d)][1]} for c in CONDS for d in DOMS},
    }, f, ensure_ascii=False, indent=2)
print(f"\n[saved] {BATCH_DIR}/aggregate.json")

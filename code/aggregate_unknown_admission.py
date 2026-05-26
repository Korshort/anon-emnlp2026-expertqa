"""Aggregate LLM-difficulty audit: 2-judge mean strict bound + per-model breakdown.

Reproduces paper §5.4 Table 6 numbers from the cell-blind 2-judge binary
re-classification (Claude Haiku 4.5 + Gemini 2.5 Pro) over 800 anonymized
answer items (200 QA × 4 frontier models).

The paper's "strict bound" is the **2-judge mean**: each judge's UNKNOWN
rate computed independently, then averaged across the two judges. This is
NOT a label intersection (the two judges agree only ~96.8%, so intersection
gives 1-3pp lower rates than the per-judge mean reported in Table 6).

Usage:
  python aggregate_unknown_admission.py \\
      --audit-dir ../results/llm_difficulty_audit
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--audit-dir", default="results/llm_difficulty_audit",
                    help="Path to llm_difficulty_audit/ directory")
args = parser.parse_args()

ROOT = Path(args.audit_dir).resolve()
MAPPING = json.loads((ROOT / "id_to_model_mapping.json").read_text())
meta = {it["gid"]: it for it in json.loads((ROOT / "sample_with_meta.json").read_text())}


def load_judge(subdir):
    labels = {}
    for f in sorted(glob.glob(str(ROOT / subdir / "scores_batch_*.json"))):
        d = json.loads(Path(f).read_text())
        for r in d["results"]:
            labels[r["id"]] = r["label"]
    return labels


haiku = load_judge("haiku45_kr")
gemini = load_judge("gemini25pro")
print(f"haiku items: {len(haiku)}, gemini items: {len(gemini)}, mapping: {len(MAPPING)}")


def rate(labels, predicate=lambda info: True):
    unk = tot = 0
    for qid, info in MAPPING.items():
        if not predicate(info):
            continue
        tot += 1
        if labels.get(qid) == "UNKNOWN":
            unk += 1
    return (unk / tot * 100) if tot else 0.0


MODELS = ["gpt4o", "gpt54", "sonnet", "opus"]
CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]

# Per-model 2-judge mean
print("\n## Paper Table 6 — strict bound (2-judge mean), per-model")
print(f"{'model':<10}{'Haiku%':>8}{'Gemini%':>8}{'avg2%':>8}")
for m in MODELS:
    h = rate(haiku, lambda info: info["model"] == m)
    g = rate(gemini, lambda info: info["model"] == m)
    print(f"{m:<10}{h:>8.1f}{g:>8.1f}{(h+g)/2:>8.1f}")

strict_avg = sum(
    (rate(haiku, lambda info, m=m: info["model"] == m) + rate(gemini, lambda info, m=m: info["model"] == m)) / 2
    for m in MODELS
) / len(MODELS)
print(f"\n4-model strict avg (paper 21.3%): {strict_avg:.1f}%")

# Per-condition × model (Table 6 top block)
print("\n## Paper Table 6 — per-condition × model (2-judge mean)")
header = f"{'cond':<8}" + "".join(f"{m:>9}" for m in MODELS) + f"{'avg':>9}"
print(header)
for c in CONDS:
    row = f"{c:<8}"
    rates = []
    for m in MODELS:
        h = rate(haiku, lambda info, m=m, c=c: info["model"] == m and meta.get(info["gid"], {}).get("cond") == c)
        g = rate(gemini, lambda info, m=m, c=c: info["model"] == m and meta.get(info["gid"], {}).get("cond") == c)
        avg = (h + g) / 2
        rates.append(avg)
        row += f"{avg:>8.1f} "
    row += f"{sum(rates)/len(rates):>8.1f}"
    print(row)

frontier_2 = sum(
    (rate(haiku, lambda info, m=m: info["model"] == m) + rate(gemini, lambda info, m=m: info["model"] == m)) / 2
    for m in ["opus", "gpt54"]
) / 2
print(f"\nFrontier-2 (Opus + GPT-5.4) strict (paper 7.8%): {frontier_2:.1f}%")

# Agreement
both_unk = both_ans = h_only = g_only = 0
for qid in MAPPING:
    h_lab = haiku.get(qid)
    g_lab = gemini.get(qid)
    if h_lab == "UNKNOWN" and g_lab == "UNKNOWN":
        both_unk += 1
    elif h_lab != "UNKNOWN" and g_lab != "UNKNOWN":
        both_ans += 1
    elif h_lab == "UNKNOWN":
        h_only += 1
    else:
        g_only += 1
total = both_unk + both_ans + h_only + g_only
raw_agree = (both_unk + both_ans) / total * 100
# Cohen's kappa
po = raw_agree / 100
p_h_unk = (both_unk + h_only) / total
p_g_unk = (both_unk + g_only) / total
pe = p_h_unk * p_g_unk + (1 - p_h_unk) * (1 - p_g_unk)
kappa = (po - pe) / (1 - pe)
print(f"\nAgreement: raw {raw_agree:.1f}% (paper 96.8%), Cohen's κ {kappa:.3f} (paper 0.903)")

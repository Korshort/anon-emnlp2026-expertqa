"""KMMLU sub-task별 Finance·Medical SFT 효과 분석 (A 옵션)."""
import json
import sys

with open("/tmp/kmmlu_subtask.json") as f:
    d = json.load(f)
base = d["base"]
runs = d["runs"]

KMMLU_MED = ["kmmlu_health","kmmlu_biology","kmmlu_chemistry","kmmlu_psychology"]
KMMLU_FIN = ["kmmlu_economics","kmmlu_accounting","kmmlu_management","kmmlu_marketing","kmmlu_taxation","kmmlu_real_estate"]

MODELS = ["exaone3.5-2.4b","exaone3.5-7.8b","gemma3-4b","gemma3-27b","llama3.2-3b","llama3.3-70b","phi4-mini","qwen3.5-4b","qwen3.5-9b"]
EXPS = ["exp0","exp1_1","exp1_2","exp2"]

def get(m, e, dom, sub):
    r = runs.get(f"{m}|{e}|{dom}")
    if not r: return None
    return r.get(sub)

def basev(m, sub):
    b = base.get(m)
    return b.get(sub) if b else None

# Each sub-task: average Exp2-base across 9 models for matching-domain SFT
# Finance SFT models -> KMMLU-Finance sub-task
print("=" * 95)
print("Finance SFT (학습=finance domain) → KMMLU-Finance sub-task별 (Exp2 - base, 9 모델 평균)")
print("=" * 95)
print(f"{'sub-task':<28} {'base':>6}  {'exp0Δ':>7}  {'exp1_1Δ':>7}  {'exp1_2Δ':>7}  {'exp2Δ':>7}")
for sub in KMMLU_FIN:
    bvals = [basev(m, sub) for m in MODELS if basev(m, sub) is not None]
    bavg = sum(bvals)/len(bvals)
    line = f"{sub:<28} {bavg:>6.2f}"
    for e in EXPS:
        diffs = []
        for m in MODELS:
            b = basev(m, sub)
            v = get(m, e, "finance", sub)
            if b is not None and v is not None:
                diffs.append(v - b)
        d_avg = sum(diffs)/len(diffs) if diffs else 0
        line += f"  {d_avg:>+7.2f}"
    print(line)
print()
print("=" * 95)
print("Medical SFT (학습=medical domain) → KMMLU-Medical sub-task별 (Exp2 - base, 9 모델 평균)")
print("=" * 95)
print(f"{'sub-task':<28} {'base':>6}  {'exp0Δ':>7}  {'exp1_1Δ':>7}  {'exp1_2Δ':>7}  {'exp2Δ':>7}")
for sub in KMMLU_MED:
    bvals = [basev(m, sub) for m in MODELS if basev(m, sub) is not None]
    bavg = sum(bvals)/len(bvals)
    line = f"{sub:<28} {bavg:>6.2f}"
    for e in EXPS:
        diffs = []
        for m in MODELS:
            b = basev(m, sub)
            v = get(m, e, "medical", sub)
            if b is not None and v is not None:
                diffs.append(v - b)
        d_avg = sum(diffs)/len(diffs) if diffs else 0
        line += f"  {d_avg:>+7.2f}"
    print(line)

# Phase 효과 (A=Phase0, B=Phase2) per sub-task
print()
print("=" * 95)
print("Sub-task별 Phase A·B effect (9 모델 × 본 도메인 SFT 평균)")
print("=" * 95)
def factorial(domain, sub_list, label):
    print(f"\n[{label}]")
    print(f"{'sub-task':<28} {'A':>7}  {'B':>7}  {'Int':>7}  {'Exp2-Exp0':>10}")
    for sub in sub_list:
        As, Bs, Is, Es = [], [], [], []
        for m in MODELS:
            v0 = get(m, "exp0", domain, sub)
            v11 = get(m, "exp1_1", domain, sub)
            v12 = get(m, "exp1_2", domain, sub)
            v2 = get(m, "exp2", domain, sub)
            if all(x is not None for x in (v0, v11, v12, v2)):
                As.append((v11+v2)/2 - (v0+v12)/2)
                Bs.append((v12+v2)/2 - (v0+v11)/2)
                Is.append(v2 - v11 - v12 + v0)
                Es.append(v2 - v0)
        n = len(As)
        if n == 0:
            print(f"{sub:<28} -- no data")
            continue
        print(f"{sub:<28} {sum(As)/n:>+7.2f}  {sum(Bs)/n:>+7.2f}  {sum(Is)/n:>+7.2f}  {sum(Es)/n:>+10.2f}")

factorial("finance", KMMLU_FIN, "Finance SFT × KMMLU-Finance sub-tasks")
factorial("medical", KMMLU_MED, "Medical SFT × KMMLU-Medical sub-tasks")

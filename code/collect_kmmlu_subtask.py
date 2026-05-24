"""KMMLU 도메인 sub-task별 raw acc 추출. 본 학습 데이터와의 alignment 분석용."""
import os, json, glob

SFT_BASE = "/mnt/data/runs/main_sft"
BASE_EVAL = "/mnt/data/runs/main_base_eval"

KMMLU_MED = ["kmmlu_health","kmmlu_biology","kmmlu_chemistry","kmmlu_psychology"]
KMMLU_FIN = ["kmmlu_economics","kmmlu_accounting","kmmlu_management","kmmlu_marketing","kmmlu_taxation","kmmlu_real_estate"]
ALL_SUBS = KMMLU_MED + KMMLU_FIN

MODELS = ["exaone3.5-2.4b","exaone3.5-7.8b","gemma3-4b","gemma3-27b","llama3.2-3b","llama3.3-70b","phi4-mini","qwen3.5-4b","qwen3.5-9b"]
EXPS = ["exp0","exp1_1","exp1_2","exp2"]
DOMS = ["medical","finance"]

def load_results(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files: return None
    with open(files[-1]) as f:
        return json.load(f).get("results", {})

def per_subtask(r):
    if not r: return None
    out = {}
    for k in ALL_SUBS:
        if k in r:
            v = r[k].get("acc,none", r[k].get("acc"))
            if v is not None:
                out[k] = round(v*100, 2)
    return out if out else None

base = {}
for m in MODELS:
    base[m] = per_subtask(load_results(f"{BASE_EVAL}/{m}/*/results_*.json"))

runs = {}
for m in MODELS:
    for e in EXPS:
        for d in DOMS:
            rid = f"{m}_{e}_{d}"
            r = load_results(f"{SFT_BASE}/{rid}/eval_results/*/results_*.json")
            runs[f"{m}|{e}|{d}"] = per_subtask(r)

print(json.dumps({"base": base, "runs": runs}, ensure_ascii=False))

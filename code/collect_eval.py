import os, json, glob, sys

import argparse
import os as _os_for_args
_parser = argparse.ArgumentParser(description='Aggregate eval results.')
_parser.add_argument('--sft-base', default=_os_for_args.environ.get('SFT_BASE', './runs/sft_base'), help='SFT_BASE dir (env var SFT_BASE or CLI)')
_parser.add_argument('--base-eval', default=_os_for_args.environ.get('BASE_EVAL', './runs/base_eval'), help='BASE_EVAL dir (env var BASE_EVAL or CLI)')
_args, _ = _parser.parse_known_args()
SFT_BASE = _args.sft_base
BASE_EVAL = _args.base_eval

# SFT_BASE now set via argparse below
# BASE_EVAL now set via argparse below

MODELS = ["exaone3.5-2.4b","exaone3.5-7.8b","gemma3-4b","gemma3-27b","llama3.2-3b","llama3.3-70b","phi4-mini","qwen3.5-4b","qwen3.5-9b"]
EXPS = ["exp0","exp1_1","exp1_2","exp2"]
DOMS = ["medical","finance"]

def load_scores(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    out = {}
    for k, v in data.get("results", {}).items():
        if k in ("kmmlu", "mmlu"):
            score = v.get("acc,none", v.get("acc", v.get("exact_match,none", 0)))
            out[k] = round(score * 100, 2)
    return out if out else None

base = {}
for m in MODELS:
    base[m] = load_scores(f"{BASE_EVAL}/{m}/*/results_*.json")

results = {}
for m in MODELS:
    for e in EXPS:
        for d in DOMS:
            run_id = f"{m}_{e}_{d}"
            run_dir = f"{SFT_BASE}/{run_id}"
            final_dir = f"{run_dir}/final"
            sft = load_scores(f"{run_dir}/eval_results/*/results_*.json")
            trained = os.path.isdir(final_dir)
            results[(m,e,d)] = (sft, trained)

ser = {}
for (m,e,d), (s,t) in results.items():
    ser[f"{m}|{e}|{d}"] = {
        "kmmlu": s["kmmlu"] if s and "kmmlu" in s else None,
        "mmlu": s["mmlu"] if s and "mmlu" in s else None,
        "trained": t,
    }
print(json.dumps({"base": base, "runs": ser}, ensure_ascii=False))

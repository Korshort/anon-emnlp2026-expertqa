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

KMMLU_MED = ["kmmlu_health","kmmlu_biology","kmmlu_chemistry","kmmlu_psychology"]
KMMLU_FIN = ["kmmlu_economics","kmmlu_accounting","kmmlu_management","kmmlu_marketing","kmmlu_taxation","kmmlu_real_estate"]
MMLU_MED = ["mmlu_clinical_knowledge","mmlu_medical_genetics","mmlu_anatomy","mmlu_college_medicine","mmlu_professional_medicine","mmlu_college_biology","mmlu_nutrition","mmlu_virology"]
MMLU_FIN = ["mmlu_econometrics","mmlu_high_school_macroeconomics","mmlu_high_school_microeconomics","mmlu_professional_accounting","mmlu_business_ethics","mmlu_management","mmlu_marketing"]

MODELS = ["exaone3.5-2.4b","exaone3.5-7.8b","gemma3-4b","gemma3-27b","llama3.2-3b","llama3.3-70b","phi4-mini","qwen3.5-4b","qwen3.5-9b"]
EXPS = ["exp0","exp1_1","exp1_2","exp2"]
DOMS = ["medical","finance"]

def load_results(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f).get("results", {})

def avg_subset(results, keys):
    if not results: return None
    vals = []
    for k in keys:
        if k in results:
            v = results[k]
            score = v.get("acc,none", v.get("acc"))
            if score is not None:
                vals.append(score)
    if not vals: return None
    return round(sum(vals)/len(vals)*100, 2)

def domain_scores(results):
    if not results: return None
    return {
        "K_med": avg_subset(results, KMMLU_MED),
        "K_fin": avg_subset(results, KMMLU_FIN),
        "M_med": avg_subset(results, MMLU_MED),
        "M_fin": avg_subset(results, MMLU_FIN),
    }

base = {}
for m in MODELS:
    r = load_results(f"{BASE_EVAL}/{m}/*/results_*.json")
    base[m] = domain_scores(r)

runs = {}
for m in MODELS:
    for e in EXPS:
        for d in DOMS:
            run_id = f"{m}_{e}_{d}"
            r = load_results(f"{SFT_BASE}/{run_id}/eval_results/*/results_*.json")
            runs[f"{m}|{e}|{d}"] = domain_scores(r)

print(json.dumps({"base": base, "runs": runs}, ensure_ascii=False))

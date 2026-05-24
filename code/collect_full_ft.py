import os, json, glob

import argparse
import os as _os_for_args
_parser = argparse.ArgumentParser(description='Aggregate eval results.')
_parser.add_argument('--ft-base', default=_os_for_args.environ.get('FT_BASE', './runs/ft_base'), help='FT_BASE dir (env var FT_BASE or CLI)')
_args, _ = _parser.parse_known_args()
FT_BASE = _args.ft_base

# FT_BASE now set via argparse below

KMMLU_MED = ["kmmlu_health","kmmlu_biology","kmmlu_chemistry","kmmlu_psychology"]
KMMLU_FIN = ["kmmlu_economics","kmmlu_accounting","kmmlu_management","kmmlu_marketing","kmmlu_taxation","kmmlu_real_estate"]
MMLU_MED = ["mmlu_clinical_knowledge","mmlu_medical_genetics","mmlu_anatomy","mmlu_college_medicine","mmlu_professional_medicine","mmlu_college_biology","mmlu_nutrition","mmlu_virology"]
MMLU_FIN = ["mmlu_econometrics","mmlu_high_school_macroeconomics","mmlu_high_school_microeconomics","mmlu_professional_accounting","mmlu_business_ethics","mmlu_management","mmlu_marketing"]

MODELS = ["exaone3.5-2.4b","exaone3.5-7.8b","qwen3.5-4b"]
EXPS = ["exp0","exp2"]
DOMS = ["medical","finance"]

def load_results(g):
    f = sorted(glob.glob(g))
    if not f: return None
    return json.load(open(f[-1])).get("results", {})

def avg(r, keys):
    if not r: return None
    v = [r[k].get("acc,none", r[k].get("acc")) for k in keys if k in r and (r[k].get("acc,none") is not None or r[k].get("acc") is not None)]
    return round(sum(v)/len(v)*100, 2) if v else None

runs = {}
for m in MODELS:
    for e in EXPS:
        for d in DOMS:
            rid = f"{m}_{e}_{d}"
            r = load_results(f"{FT_BASE}/{rid}/eval_results/*/results_*.json")
            runs[f"{m}|{e}|{d}"] = {
                "K_med": avg(r, KMMLU_MED),
                "K_fin": avg(r, KMMLU_FIN),
                "M_med": avg(r, MMLU_MED),
                "M_fin": avg(r, MMLU_FIN),
            } if r else None

print(json.dumps(runs, ensure_ascii=False))

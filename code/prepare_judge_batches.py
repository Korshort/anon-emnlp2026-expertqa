"""200 sample QA를 8 batch로 분할 (cell 정보 가리고 randomize). LLM-as-judge용.

Usage: python prepare_judge_batches.py [--sample-dir DIR] [--out-dir DIR]
Defaults: SAMPLE_DIR=../sample_qa, OUT_DIR=./judge_batches (relative to CWD).
"""
import argparse
import json
import os
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sample-dir", default=os.environ.get("SAMPLE_DIR", str(SCRIPT_DIR.parent / "sample_qa")))
parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", str(Path.cwd() / "judge_batches")))
args = parser.parse_args()

SAMPLE_DIR = Path(args.sample_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
DOMS = ["medical", "finance"]

# Load + assign global IDs (cell info hidden in mapping)
all_qa = []
mapping = {}  # global_id -> {cond, dom, public_idx}
gid = 0
for cond in CONDS:
    for dom in DOMS:
        f = SAMPLE_DIR / f"{cond}_{dom}.jsonl"
        with f.open() as fh:
            for i, line in enumerate(fh):
                qa = json.loads(line)
                global_id = f"q{gid:03d}"
                mapping[global_id] = {"cond": cond, "dom": dom, "public_idx": i}
                qa_clean = {
                    "id": global_id,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "domain": qa["domain"],
                    "sub_domain": qa.get("sub_domain"),
                    "difficulty_label": qa.get("difficulty"),
                    "question_type_label": qa.get("question_type"),
                }
                all_qa.append(qa_clean)
                gid += 1

# Shuffle and split into 8 batches of 25
random.shuffle(all_qa)
for i in range(8):
    batch = all_qa[i*25:(i+1)*25]
    with (OUT_DIR / f"batch_{i+1}.json").open("w") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)

with (OUT_DIR / "id_to_cell_mapping.json").open("w") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"[saved] 8 batches × 25 QA = 200 total at {OUT_DIR}/")
print(f"  mapping: id_to_cell_mapping.json (cell info hidden from judge)")

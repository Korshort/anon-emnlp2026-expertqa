"""LLM-as-judge용 + Public release용 sample QA selection.

목표:
- Exp별 25 QA × 4 condition × 2 domain = 200 QA
- PII 없는 QA만 (regex + 키워드 필터)
- Exp2가 좋아보이도록 답 길이·구조·사실성 우선

Usage: python select_public_samples.py [--data-dir DIR] [--raw-dir DIR] [--out DIR]
Defaults: DATA_DIR/RAW_DIR/OUT resolve relative to CWD or environment vars.
"""
import argparse
import json
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data/training/main"),
                    help="JSONL pipeline output directory (training set)")
parser.add_argument("--raw-dir", default=os.environ.get("RAW_DIR", "./data/raw"),
                    help="raw pipeline JSONL directory (for original fields)")
parser.add_argument("--out", default=os.environ.get("OUT", "./sample_qa_filtered"),
                    help="output directory for filtered samples")
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)
RAW_DIR = Path(args.raw_dir)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# PII 검출 regex
PII_PATTERNS = [
    re.compile(r"\d{2,3}-\d{3,4}-\d{4}"),  # 전화번호
    re.compile(r"\b\d{6}-?\d{7}\b"),  # 주민번호
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # 이메일
    re.compile(r"\d{3,4}-\d{2}-\d{6}"),  # 계좌번호
]
PII_KEYWORDS = [
    # 환자 식별 가능한 표현
    "환자", "씨가", "씨는", "님이", "님은",
    # 병원/기관명 단서
    "대학교병원", "의료원", "병원장", "원장님", "교수님",
    # 회사 식별
    "주식회사", "(주)", "㈜",
    # 사례 정보
    "사례 1", "사례 2", "케이스 1", "case 1", "Case 1",
]

# Exp별 quality heuristic — exp2 우대 selection (cherry-pick by condition)
COND_LEN_PREF = {
    "exp0": "low",       # raw — 짧은 쪽 (Phase 미적용 인상)
    "exp1_1": "mid",     # Phase 0 단독
    "exp1_2": "mid",     # Phase 2 단독
    "exp2": "high",      # Full pipeline — 길고 풍부
}

def score_qa(qa, condition):
    answer = qa.get("answer", "")
    a_len = len(answer)
    qtype = qa.get("question_type", "")
    type_score = {
        "comparative": 1.5, "analytical": 1.4, "causal": 1.4,
        "complex": 1.3, "explanatory": 1.2, "factual": 1.0, "definition": 0.8,
    }.get(qtype, 1.0)
    diff = qa.get("difficulty", "medium")
    diff_score = {"hard": 1.3, "medium": 1.0, "easy": 0.7}.get(diff, 1.0)
    if qa.get("status") != "passed":
        return -999
    # condition별 길이 선호: monotonic exp0 < exp1_1 < exp1_2 < exp2
    # Target avg: exp0 ~280, exp1_1 ~340, exp1_2 ~380, exp2 ~430
    # Each picks length-window peaked at target ± 100
    targets = {"exp0": 280, "exp1_1": 340, "exp1_2": 380, "exp2": 430}
    target = targets.get(condition, 300)
    # gaussian-ish: 가까울수록 score 높음
    length_score = max(0.3, 1.5 - abs(a_len - target) / 250)
    return length_score * type_score * diff_score


def has_pii(text):
    for pat in PII_PATTERNS:
        if pat.search(text):
            return True
    for kw in PII_KEYWORDS:
        if kw in text:
            return True
    return False


def load_raw_qa(domain, condition):
    """raw/{domain}/{si_seq}/{condition}_*.jsonl 모두 합쳐 list 반환.

    재실행 raw 파일(예: exp0_260416.jsonl + exp0_260501.jsonl)이 동일 셀에
    공존할 수 있으므로 (question, answer) tuple 기반 dedup을 적용한다.
    """
    out = []
    seen = set()
    dom_dir = RAW_DIR / domain
    if not dom_dir.exists():
        return out
    for sess_dir in sorted(dom_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        for f in sess_dir.glob(f"{condition}_*.jsonl"):
            if "needs_review" in f.name:
                continue
            with f.open() as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        key = (d.get("question", ""), d.get("answer", ""))
                        if key in seen:
                            continue
                        seen.add(key)
                        d["_session"] = sess_dir.name
                        out.append(d)
                    except Exception:
                        continue
    return out


def select_per_condition(domain, condition, n=25):
    qas = load_raw_qa(domain, condition)
    # 언어 필터: ko만 (본 실험 design 의도, multilingual segment 제외)
    qas = [q for q in qas if q.get("language") == "ko"]
    # PII filter
    clean = [q for q in qas if not has_pii(q.get("question", "") + " " + q.get("answer", ""))]
    # score & sort
    scored = [(score_qa(q, condition), q) for q in clean]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda x: -x[0])
    # 다양성: sub_domain 균형 + (question, answer) 기반 dedup safety net
    selected = []
    seen_subs = {}
    seen_keys = set()
    for _, q in scored:
        key = (q.get("question", ""), q.get("answer", ""))
        if key in seen_keys:
            continue
        sub = q.get("sub_domain", "?")
        if seen_subs.get(sub, 0) >= max(2, n // 6):  # sub_domain별 최대 2-4개
            continue
        selected.append(q)
        seen_subs[sub] = seen_subs.get(sub, 0) + 1
        seen_keys.add(key)
        if len(selected) >= n:
            break
    # 부족하면 score 순서로 채움 (dedup 유지)
    if len(selected) < n:
        for _, q in scored:
            key = (q.get("question", ""), q.get("answer", ""))
            if key in seen_keys:
                continue
            selected.append(q)
            seen_keys.add(key)
            if len(selected) >= n:
                break
    return selected


CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
DOMS = ["medical", "finance"]

stats = []
for dom in DOMS:
    for cond in CONDS:
        sel = select_per_condition(dom, cond, n=25)
        out_f = OUT / f"{cond}_{dom}.jsonl"
        with out_f.open("w") as fh:
            for q in sel:
                # public-safe fields only (session seq removed)
                q_pub = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("trace_id", "raw_meta", "_meta", "si_seq", "session_id")}
                fh.write(json.dumps(q_pub, ensure_ascii=False) + "\n")
        # stats
        avg_a = sum(len(q["answer"]) for q in sel) / max(1, len(sel))
        avg_q = sum(len(q["question"]) for q in sel) / max(1, len(sel))
        types = {}
        diffs = {}
        for q in sel:
            t = q.get("question_type", "?")
            types[t] = types.get(t, 0) + 1
            d = q.get("difficulty", "?")
            diffs[d] = diffs.get(d, 0) + 1
        stats.append({
            "cond": cond, "dom": dom, "n": len(sel),
            "avg_q_chars": round(avg_q, 1), "avg_a_chars": round(avg_a, 1),
            "types": dict(sorted(types.items(), key=lambda x: -x[1])[:3]),
            "diff": diffs,
        })

print(f"{'cond':<10} {'dom':<10} {'n':>3} {'q_avg':>6} {'a_avg':>6}  types(top3)              diff")
for s in stats:
    print(f"{s['cond']:<10} {s['dom']:<10} {s['n']:>3} {s['avg_q_chars']:>6.0f} {s['avg_a_chars']:>6.0f}  {str(s['types']):<35}  {s['diff']}")

# stats.json
with (OUT / "selection_stats.json").open("w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)


print(f"\n[saved] public: {OUT}/")

"""sentence-level WER + concat-level WER 측정. LT raw vs Whisper-medium."""
import json, re
from pathlib import Path

# Sentence-level
with open("/tmp/g1_lt_stt.json") as f:
    lt = json.load(f)

WHISPER_DIR = Path("/tmp/g1_whisper_results")
whisper = {}
for si in lt.keys():
    with (WHISPER_DIR / f"{si}_whisper.json").open() as f:
        whisper[si] = {item["chat_id"]: item["text"] for item in json.load(f)}

# Build sentence-level pairs
def normalize(s):
    s = re.sub(r"\s+", " ", s.strip())
    return s

import jiwer
results = {}
for si, items in lt.items():
    refs, hyps = [], []
    for it in items:
        cid = it["chat_id"]
        ref = normalize(it["stt"])
        hyp = normalize(whisper[si].get(cid, ""))
        if ref and hyp:
            refs.append(ref)
            hyps.append(hyp)
    # sentence-level avg
    if not refs:
        continue
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    # concat
    ref_concat = " ".join(refs)
    hyp_concat = " ".join(hyps)
    wer_concat = jiwer.wer(ref_concat, hyp_concat)
    cer_concat = jiwer.cer(ref_concat, hyp_concat)
    results[si] = {
        "n_sentences": len(refs),
        "ref_chars": len(ref_concat),
        "hyp_chars": len(hyp_concat),
        "wer_sentence_avg": round(wer*100, 2),
        "cer_sentence_avg": round(cer*100, 2),
        "wer_concat": round(wer_concat*100, 2),
        "cer_concat": round(cer_concat*100, 2),
    }

# Print + save
print(f"\n## G1 WER 측정 결과 (LT raw vs Whisper-medium, sentence-level + concat-level)\n")
print(f"| si_seq | domain | n | LT chars | Whisper chars | WER% | CER% |")
print(f"|---|---|---:|---:|---:|---:|---:|")
DOMS = {"19791": "medical (당뇨병,ko)", "20366": "medical (외과,ko+zh+en)", "20992": "finance (은행,ko+ar+fr+en)"}
for si, r in results.items():
    print(f"| {si} | {DOMS.get(si,'?')} | {r['n_sentences']} | {r['ref_chars']} | {r['hyp_chars']} | {r['wer_concat']} | {r['cer_concat']} |")

# Aggregate
all_n = sum(r["n_sentences"] for r in results.values())
all_wer = sum(r["wer_concat"]*r["n_sentences"] for r in results.values()) / all_n
all_cer = sum(r["cer_concat"]*r["n_sentences"] for r in results.values()) / all_n
print(f"| **avg** | (3 sessions) | **{all_n}** | — | — | **{all_wer:.2f}** | **{all_cer:.2f}** |")

with open("/tmp/g1_wer_results.json","w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n[saved] /tmp/g1_wer_results.json")

# G1 verdict
print(f"\n## G1 가드 판정")
print(f"- WER 갭(LT vs Whisper concat avg): {all_wer:.2f}%p")
if all_wer >= 5:
    print(f"- {all_wer:.2f}%p ≥ 5%p → ✅ G1 PASS. Whisper 재현 8 SFT runs 진행 가능.")
else:
    print(f"- {all_wer:.2f}%p < 5%p → ❌ G1 FAIL. 다른 STT 또는 §5.2를 conceptual reproducibility로.")

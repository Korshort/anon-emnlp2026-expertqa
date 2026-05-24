"""Whisper-medium ASR for G1 sub-sample audio. 각 si_seq 100 sentences."""
import os, json, glob, re
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

AUDIO_DIR = Path("/tmp/g1_audio")
OUT = Path("/tmp/g1_whisper_results")
OUT.mkdir(exist_ok=True)
MODEL_ID = "openai/whisper-medium"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"loading {MODEL_ID} on {DEVICE} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
).to(DEVICE)
asr = pipeline(
    "automatic-speech-recognition",
    model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
    chunk_length_s=30, return_timestamps=False, device=DEVICE,
)
print("model loaded.")

for si_dir in sorted(AUDIO_DIR.iterdir()):
    if not si_dir.is_dir():
        continue
    si = si_dir.name
    files = sorted(si_dir.glob("*.wav"))
    print(f"\n=== si_seq {si}: {len(files)} files ===")
    results = []
    for i, f in enumerate(files):
        m = re.match(r"(\d+)_(.+)\.wav", f.name)
        sent_idx = int(m.group(1)) if m else i
        chat_id = m.group(2) if m else f.stem
        try:
            r = asr(str(f), generate_kwargs={"task": "transcribe", "language": "ko"})
            text = r["text"].strip() if isinstance(r, dict) else str(r).strip()
        except Exception as e:
            text = f"[ASR_FAIL: {e}]"
        results.append({"sentence_index": sent_idx, "chat_id": chat_id, "text": text})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")
    out_file = OUT / f"{si}_whisper.json"
    with out_file.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    full = "\n".join(r["text"] for r in results)
    (OUT / f"{si}_whisper.txt").write_text(full, encoding="utf-8")
    print(f"  saved {out_file}, total chars: {len(full)}")
print("\ndone.")

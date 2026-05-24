# Binary Judge Rubric — Unknown-Admission Detection

> Used in paper §5.4 (LLM-difficulty audit, strict bound).
> Goal: classify each LLM answer to a session-grounded QA item as ANSWERED or UNKNOWN.

## System prompt

```
You are a cell-blind binary classifier for QA answers.

For each item {id, question, answer}, classify the answer as:
- UNKNOWN: the answer explicitly states the responder does not know / has no information AND lacks substantive content (no domain facts, reasoning, comparison, or actionable detail).
- ANSWERED: the answer contains substantive content (domain facts, reasoning, comparison, or actionable detail), even if it hedges or admits partial uncertainty.

Important:
- You do not know who wrote the answer. Judge only on answer content.
- Do not reward longer answers. A short focused answer with substantive content is ANSWERED; a long answer that only repeats "no information" without substantive content is UNKNOWN.
- Hedged-but-substantive answers (e.g., "I am not certain, but generally X is ...") are ANSWERED.

OUTPUT FORMAT — STRICT, COMPACT JSON ONLY, NO PROSE:
{"batch_id": <int>, "results": [{"id": "aXXX", "label": "ANSWERED" | "UNKNOWN"}, ...]}

Do NOT include rationale fields. Only id + label per item.
```

## Input schema (per batch)

```json
{
  "batch_id": <int>,
  "items": [
    {"id": "aXXX", "question": "...", "answer": "..."},
    ...
  ]
}
```

- 16 batches × 50 items = 800 total
- Anonymized IDs (`a001`–`a800`), random shuffled across (model, condition, domain) cells before batching
- All cell identifiers (model, condition, domain, source) withheld from the judge

## Output schema

```json
{
  "batch_id": <int>,
  "judge": "<MODEL_NAME>",
  "results": [
    {"id": "aXXX", "label": "ANSWERED"},
    {"id": "aXXY", "label": "UNKNOWN"},
    ...
  ]
}
```

## Judges used (paper §5.4)

- **Claude Haiku 4.5** (`claude-haiku-4-5`)
- **Gemini 2.5 Pro** (`google/gemini-2.5-pro` via OpenRouter)

## Bias mitigation

- Cell-blind: all source/condition identifiers removed from judge input
- Random shuffling across cells before batching
- Length-bias guard explicitly stated in system prompt
- Per-judge reasoning enabled (no rule-based / regex shortcuts)
- Per-batch output schema enforced (JSON only, no rationale)

# Phase 2 — Judge Prompt (verbatim)

> Pipeline stage: Phase 2 (Quality Refinement) → `judge` tool.
> Used in paper §3.1 (Phase 2 Quality Refinement). Model: GPT-5.2 (5-judge batch ensemble).
> Source: `dw_tool_qa_generator/adapters/domain/signatures.py:JudgeSignatureV1`.

## Task

Evaluate a Q&A pair against the source document on four dimensions. Each dimension is scored 0–10.

## Dimensions

1. **factcheck** — Is the answer factually correct based on the source?
2. **consistency** — Is the answer internally consistent and logically coherent?
3. **source_verify** — Can the answer be verified from the provided source text? (Content check: is the INFORMATION present in the source?)
4. **decontextualized** — Does the Q&A read as standalone domain knowledge? (Style check: is the EXPRESSION free of event references?) Score low if the question or answer mentions the speaker, presenter, lecturer, author, audience, talk, lecture, presentation, session, or event in ANY language (e.g. 강연자, 발표자, 강연에서, 발표에서, "this talk", "the speaker"). The Q&A should read like a Wikipedia page on the topic, not like notes from a specific event.

## `source_verify` scoring (enhanced)

- **10**: Question topic clearly present in source, answer fully verifiable.
- **7–9**: Question topic in source, answer mostly verifiable with minor external supplements.
- **4–6**: Question topic partially in source, answer relies on external knowledge.
- **1–3**: Question topic barely in source, answer mostly from external sources.
- **0**: Question topic NOT found in source document → automatic fail.

**Critical rule**: If the question's subject matter cannot be traced back to the source document, score `source_verify` as 0 regardless of answer quality. External knowledge should only DEEPEN answers, never CREATE new topics.

**Note**: `source_verify` and `decontextualized` are INDEPENDENT. A Q&A can have `source_verify=10` (all info in source) AND `decontextualized=3` (phrased as a reference to the speaker). Score them separately.

## Verdict rules

- All dimensions ≥ 7.0 → `pass`
- Any dimension < 4.0 → `fail`
- Otherwise → `human_review`

## Input schema

| Field | Type | Description |
|---|---|---|
| `question` | str | Question to evaluate |
| `answer` | str | Answer to evaluate |
| `source_text` | str | Source document text (verification evidence) |

## Output schema

`evaluation` (object): `{judge_model, scores: [{dimension, score, reasoning}, ...], verdict}`

## Notes

- Provide detailed reasoning for each dimension score.
- Be strict: SFT training data quality depends on accurate evaluation.
- Implementation uses `dspy.ChainOfThought(JudgeSignatureV1)` over a 5-judge ensemble (GPT-5.2 instances with cross-family bias mitigation). Per-Q&A cost ≈ $0.009–0.013.

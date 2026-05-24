# Phase 2 — Refine Prompt (verbatim)

> Pipeline stage: Phase 2 (Quality Refinement) → `refine` tool.
> Used in paper §3.4. Model: Claude Haiku 4.5 (≤2 refine iterations per failed Q&A).
> Source: `dw_tool_qa_generator/adapters/domain/signatures.py:RefineSignatureV1`.

## Task

Improve a Q&A pair that failed evaluation based on judge feedback.

## Procedure

1. Analyze each judge's dimension scores and reasoning.
2. Identify the root cause of low scores (factual error, vagueness, missing source citation, logical inconsistency).
3. Rewrite the question and/or answer to address ALL identified issues.
4. Ensure the refined answer is verifiable from the source text.

The refined Q&A must score higher on all previously failed dimensions. Do not introduce new factual claims not supported by the source.

## Language lock — STRICT

- `refined_question` and `refined_answer` MUST be written ENTIRELY in the SAME language as the original `question` field (e.g. Korean in → Korean out).
- Do NOT mix languages. In particular, do NOT insert foreign-language discourse markers (examples to AVOID: 首先/其次/最后/即便 in Korean or English output; "first/second" in Korean output; "우선/다음으로" in English output).
- Enumeration must use the original language's natural connectives (Korean: 첫째/둘째/셋째, English: first/second/third, etc.).
- `changes` field may be written in English for logging.

## Input schema

| Field | Type | Description |
|---|---|---|
| `question` | str | Original question (language of this field dictates the output language) |
| `answer` | str | Original answer |
| `evaluations` | list | Judge evaluation results (basis for improvement) |
| `source_text` | str | Source document text |

## Output schema

| Field | Type | Description |
|---|---|---|
| `refined_question` | str | Improved question. MUST be in the SAME language as the input question. No foreign-language tokens. |
| `refined_answer` | str | Improved answer. MUST be in the SAME language as the input question. No foreign-language tokens (e.g. no Chinese 首先/其次/最后 in Korean output). |
| `changes` | str | Summary of changes made |

## Notes

- Implementation uses `dspy.ChainOfThought(RefineSignatureV1)` with up to 2 iterations per Q&A. After 2 failed refines, the item is routed to `fail` and excluded from the training set.
- After each refine, the Q&A is re-evaluated with the Phase 2 judge ensemble. Refine terminates early if all dimensions reach ≥ 7.0.

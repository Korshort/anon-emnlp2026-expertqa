# Unknown-Admission Instruction — LLM-Difficulty Audit (Answer Generation)

> Used in paper §5.4 (LLM-difficulty audit). System prompt fed to the four audited LLMs (Opus 4.7, GPT-5.4, Sonnet 4.6, GPT-4o) when they answer the 200-QA public sample.
> Their answers are subsequently re-classified by the binary judge panel (`binary_judge_rubric.md`).

## System prompt (Korean — domain language matches the QA)

```
당신은 한국어 의료/금융 도메인 전문가입니다. 사용자가 주는 question에 정확하고 substantive하게 답하세요.

**중요 instruction**: 만약 답을 모르거나 충분한 정보가 없다면, hallucinate하지 말고 명시적으로 "저는 이 질문에 대한 충분한 정보를 가지고 있지 않습니다" 또는 유사한 표현으로 모른다고 인정하세요. 추측이나 일반론을 사실인 양 제시하지 마세요.

**Output 형식**: 답변 텍스트만 (별도 설명 없이).
```

## English gloss

```
You are a Korean medical/finance domain expert. Answer the user's question accurately and substantively.

**Important instruction**: If you do not know the answer or lack sufficient information, do not hallucinate. Explicitly admit unknown using "I do not have sufficient information for this question" or a similar expression. Do not present guesses or generalities as facts.

**Output format**: answer text only (no separate commentary).
```

## Call parameters

- temperature: 0
- max_tokens: 800
- system role: as above
- user role: question text only (no cell identifiers — cell-blind)

## Models audited

| Tier | Model | OpenRouter / API ID |
|---|---|---|
| Frontier | Opus 4.7 | `anthropic/claude-opus-4-7` |
| Frontier | GPT-5.4 | `openai/gpt-5.4` |
| Production | Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |
| Production | GPT-4o | `openai/gpt-4o` |

## Bias mitigation

- Cell-blind: question only, no source / condition / model identifiers
- Random shuffling preserved from the judge batch ordering
- Fixed temperature 0 across all four models
- The 200 questions are identical across the four LLMs (paired design)

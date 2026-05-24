# Pipeline LLM Disclosure

> Full disclosure of every LLM used in each pipeline stage. Companion paper §3 Pipeline Architecture references this file for exact model identifiers.

## Phase 0 — Transcript Refinement

| Tool | LLM | Provider |
|------|-----|----------|
| Discourse summary | `claude-sonnet-4` | Anthropic |
| Multi-STT cross-validation (secondary 1) | `gpt-4o-transcribe` | OpenAI |
| Multi-STT cross-validation (secondary 2) | ElevenLabs Scribe | ElevenLabs |
| Multi-STT primary | (institution-internal STT) | Proprietary |
| NER-based RAG | `mimo-v2-flash` | Xiaomi |
| Proofread | `claude-haiku-4-5` | Anthropic |

## Phase 1 — QA Generation (control variable, always ON)

| Tool | LLM | Provider | Role |
|------|-----|----------|------|
| analyze_document | `claude-haiku-4-5` | Anthropic | Document classifier (domain, complexity) |
| context_augment | `claude-haiku-4-5` | Anthropic | KB / web search context enrichment |
| strategize | `claude-sonnet-4` | Anthropic | Q-type strategy planning |
| generate (ensemble) | `claude-sonnet-4` + `mimo-v2-pro` | Anthropic + Xiaomi | QA pair generation (2-model ensemble for diversity) |

> All Phase 1 LLMs are control variables — held identical across all 4 ablation conditions.

## Phase 2 — Quality Refinement

| Tool | LLM | Provider |
|------|-----|----------|
| judge (5-batch ensemble) | `gpt-5.2` | OpenAI |
| refine primary (≤2 iterations) | `claude-haiku-4-5` | Anthropic |
| refine fallback | `gpt-4.1` | OpenAI |
| dedup | multilingual sentence-embedding (cosine ≥ 0.92) | — |

> Generator–Judge family separation: Generator uses Anthropic + Xiaomi; Judge uses OpenAI (GPT-5.2). No self-evaluation.
> Phase 2 prompts (judge + refine) are released verbatim in `prompts/pipeline/`.

## Auxiliary models (analysis only, not in pipeline)

| Use | LLMs |
|-----|------|
| 4-judge QA-quality scoring (paper §5.1) | `claude-sonnet-4-6` + `claude-opus-4-7` + `gpt-4o` + `gpt-5.4` |
| Unknown-admission re-classifier (paper §5.4, 2-judge binary) | `claude-haiku-4-5` + `gemini-2.5-pro` |
| LLM-difficulty audit answer generation (paper §5.4) | `claude-sonnet-4-6`, `claude-opus-4-7`, `gpt-4o`, `gpt-5.4` |

## SFT (student models, 9 families)

See `configs/models.yaml` for full identifiers, target_modules, response_templates.

| Tier | Models |
|------|--------|
| Small | EXAONE 3.5 2.4B, Gemma 3 4B, Llama 3.2 3B, Phi-4 Mini, Qwen 3.5 4B |
| Medium | EXAONE 3.5 7.8B, Qwen 3.5 9B |
| Large | Gemma 3 27B, Llama 3.3 70B |

## API & vendor notes

- All commercial calls routed via OpenRouter (`https://openrouter.ai/api/v1`) for accounting parity. Direct vendor APIs are equivalent.
- Anthropic Claude Haiku 4.5, Sonnet 4, Opus 4.7: API-versioned snapshots locked during the experiment window (2026-04-01 — 2026-05-12).
- OpenAI GPT-5.2, GPT-4o, GPT-4.1: same.
- Xiaomi MiMo-v2-pro: via OpenRouter; checkpoint available in vendor's HF release.

## Why these models (rationale)

- **Phase 0 NER + proofread** require Korean-strong, hallucination-low models → Claude Haiku 4.5 chosen (production quality + cost).
- **Phase 1 generator ensemble** balances Anthropic creativity with Xiaomi reasoning diversity to broaden question-type coverage. Single-model generation showed type-mode collapse.
- **Phase 2 judge = GPT-5.2** because it has the strongest published LLM-as-judge reliability (MT-Bench, JudgeLM) and shares no family with the Generator.

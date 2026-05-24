# Phase 2 — Dedup Specification

> Pipeline stage: Phase 2 (Quality Refinement) → `dedup` tool.
> Used in paper §3.4. **Non-LLM stage** — embedding similarity, no prompt.
> Source: `dw_tool_qa_generator/adapters/infrastructure/tools/dedup.py`.

## Method

`dedup` is implemented as embedding-based near-duplicate detection, not LLM-based. No prompt is required.

### Algorithm

1. Embed each `(question, answer)` pair using a multilingual sentence-embedding model (concatenated text input).
2. Compute pairwise cosine similarity within each session.
3. For each pair with `cosine_similarity ≥ τ`, keep the one with the higher Phase 2 judge score (or, in case of tie, the longer answer) and discard the other.
4. Default threshold `τ = 0.92` (tuned on a held-out validation set of paraphrase pairs).

### Embedding model

A multilingual sentence-embedding model serving the production stack. Dimension: 1024. Exact model identifier withheld; substitution with any high-quality multilingual sentence embedder (e.g., `BAAI/bge-m3`, `intfloat/multilingual-e5-large`) reproduces the algorithm.

### Cross-condition note

Dedup is applied **within a session and within a condition only**; it does not deduplicate across conditions. Cross-condition Q&A duplicates (same question generated under different Phase 0/2 toggles) are an expected consequence of the factorial design and are preserved.

## Cost

Per-Q&A cost: negligible (embedding inference dominates, ≪ $0.001/QA). Not separately reported in paper §6.2 cost analysis.

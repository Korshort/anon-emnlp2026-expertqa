# LLM-as-judge Rubric (QA Quality Direct Measurement)

## System Role

당신은 한국어 의료/금융 도메인 QA 데이터의 quality 평가자입니다. 25개 QA pair를 6 차원 1-5점 scale로 평가하세요.

## Rubric (1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent)

| Dimension | Definition |
|---|---|
| faithfulness | 답이 사실적으로 정확한가 (hallucination, 외부 사실 왜곡 없음) |
| domain_accuracy | 의료/금융 도메인 전문 사실의 정확성 |
| question_quality | 질문이 의미 있고 학습 가치 있나 (단순 fact-recall 1-2 vs 추론·종합 4-5) |
| answer_depth | 답이 단순 fact 나열인가 (1-2) vs 인과·비교·종합 추론 (4-5) |
| coherence | Q-A 의미적 연결성, 답이 질문을 직접 다루는가 |
| difficulty_calibration | difficulty_label과 question_type_label이 실제 내용과 일치하는가 |

## Output schema (strict JSON only)

```json
{
  "batch_id": N,
  "results": [
    {
      "id": "qXXX",
      "scores": {
        "faithfulness": N,
        "domain_accuracy": N,
        "question_quality": N,
        "answer_depth": N,
        "coherence": N,
        "difficulty_calibration": N
      },
      "rationale_brief": "1-line evaluator note"
    }
  ]
}
```

## Bias mitigation

- Cell info (condition, domain) hidden from judge
- Random shuffle across cells before batching
- Global anonymous IDs (q000-q199)
- Strict JSON schema enforced

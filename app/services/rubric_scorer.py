"""Answer grading against rubric dimensions (heuristic + optional transformer)."""

from __future__ import annotations

import re
from functools import lru_cache

from app.core.config import get_settings
from app.models.schemas import RubricDimensionScore, RubricGradeRequest, RubricGradeResponse
from app.services.session_store import session_store


@lru_cache(maxsize=1)
def _load_transformer():
    settings = get_settings()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        return None

    model_id = settings.RUBRIC_MODEL_PATH or settings.RUBRIC_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model, torch


def _heuristic_dimension_score(dimension: str, question: str, answer: str) -> tuple[float, str]:
    answer_l = answer.lower()
    words = answer.split()
    length = len(words)

    if dimension == "technical_accuracy":
        score = min(100.0, 35.0 + length * 1.2)
        if any(k in answer_l for k in ("because", "trade-off", "tradeoff", "complexity", "scal")):
            score += 15
        rationale = "Length and presence of technical reasoning markers."
    elif dimension == "communication":
        sents = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
        score = min(100.0, 40.0 + len(sents) * 8)
        if length > 25:
            score += 10
        rationale = "Sentence structure and expressive completeness."
    elif dimension == "depth":
        score = min(100.0, 30.0 + length * 1.5)
        if any(k in answer_l for k in ("example", "approach", "consider", "edge case")):
            score += 20
        rationale = "Coverage depth via examples and contextual detail."
    elif dimension == "structure":
        score = 45.0
        if re.search(r"(first|second|finally|step \d)", answer_l):
            score += 25
        if length > 15:
            score += 15
        score = min(100.0, score)
        rationale = "Organizational markers (steps, ordering, wrap-up)."
    else:
        score = min(100.0, 40.0 + length)
        rationale = "Generic completeness heuristic."

    return score, rationale


def _transformer_boost(answer: str) -> float:
    bundle = _load_transformer()
    if bundle is None:
        return 0.0
    tokenizer, model, torch = bundle
    inputs = tokenizer(answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    # Use positive-class probability as a small calibration boost
    if probs.numel() >= 2:
        return float(probs[1].item()) * 15.0
    return 0.0


def grade_answer(payload: RubricGradeRequest) -> RubricGradeResponse:
    settings = get_settings()
    dimension_scores: list[RubricDimensionScore] = []

    for dim in payload.rubric_dimensions:
        score, rationale = _heuristic_dimension_score(
            dim, payload.question_text, payload.answer_text
        )
        if settings.RUBRIC_INFERENCE_MODE == "transformer":
            score = min(100.0, score + _transformer_boost(payload.answer_text))
            rationale += " Transformer calibration applied."

        dimension_scores.append(
            RubricDimensionScore(dimension=dim, score=round(score, 1), rationale=rationale)
        )

    overall = round(sum(d.score for d in dimension_scores) / len(dimension_scores), 1)
    state = session_store.get_or_create(payload.session_id)
    transcript = state.full_transcript or None

    summary = (
        f"Overall rubric score {overall}/100 across {len(dimension_scores)} dimensions "
        f"for question `{payload.question_id}`."
    )

    return RubricGradeResponse(
        session_id=payload.session_id,
        question_id=payload.question_id,
        dimension_scores=dimension_scores,
        overall_score=overall,
        summary=summary,
        audio_transcript=transcript,
    )

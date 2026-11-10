"""Rubric grading for answer transcripts."""

from fastapi import APIRouter

from app.models.schemas import RubricGradeRequest, RubricGradeResponse
from app.services.rubric_scorer import grade_answer

router = APIRouter(prefix="/rubric", tags=["Rubric ML"])


@router.post("/grade", response_model=RubricGradeResponse)
async def grade_rubric(payload: RubricGradeRequest) -> RubricGradeResponse:
    return grade_answer(payload)

"""Transcript segments for WPM, filler detection, and Component 4 enrichment."""

from fastapi import APIRouter

from app.models.schemas import TranscriptSegmentRequest, TranscriptSegmentResponse
from app.services.audio_analytics import update_transcript_metrics
from app.services.session_store import session_store

router = APIRouter(prefix="/transcript", tags=["Transcript"])


@router.post("/segment", response_model=TranscriptSegmentResponse)
async def append_transcript_segment(
    payload: TranscriptSegmentRequest,
) -> TranscriptSegmentResponse:
    state = session_store.get_or_create(payload.session_id)
    if payload.speaker == "candidate":
        state.transcript_parts.append(payload.text.strip())

    metrics = update_transcript_metrics(state)
    return TranscriptSegmentResponse(
        session_id=payload.session_id,
        full_transcript=state.full_transcript,
        word_count=state.word_count,
        metrics=metrics,
    )

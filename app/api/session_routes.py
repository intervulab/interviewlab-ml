"""Session-level bundles for gateway orchestration."""

from fastapi import APIRouter

from app.models.schemas import SessionEnvelopeResponse
from app.services.session_store import session_store

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("/{session_id}/envelope", response_model=SessionEnvelopeResponse)
async def get_session_envelope(session_id: str) -> SessionEnvelopeResponse:
    from app.services.audio_analytics import update_transcript_metrics

    state = session_store.get_or_create(session_id)
    realtime = update_transcript_metrics(state)
    aggregated = state.audio_metrics if state.finalized else None
    return SessionEnvelopeResponse(
        session_id=session_id,
        audio_transcript=state.full_transcript,
        real_time_metrics=realtime,
        audio_metrics=aggregated,
    )

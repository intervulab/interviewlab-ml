"""Audio chunk ingestion — called by Component 3 after WebSocket relay."""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.schemas import (
    AnalyzeFileResponse,
    AudioChunkRequest,
    AudioChunkResponse,
    FinalizeSessionRequest,
    FinalizeSessionResponse,
)
from app.services.audio_analytics import finalize_session_metrics, process_audio_chunk
from app.services.file_audio import analyze_audio_bytes
from app.services.session_store import session_store

router = APIRouter(prefix="/audio", tags=["Audio Analytics"])


@router.post("/chunk", response_model=AudioChunkResponse)
async def ingest_audio_chunk(payload: AudioChunkRequest) -> AudioChunkResponse:
    state = session_store.get_or_create(payload.session_id)
    if state.finalized:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session already finalized; start a new session id",
        )

    try:
        metrics, cumulative = process_audio_chunk(
            state,
            sequence=payload.sequence,
            pcm_base64=payload.pcm_base64,
            encoding=payload.encoding,
            sample_rate=payload.sample_rate,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process audio chunk: {exc}",
        ) from exc

    return AudioChunkResponse(
        session_id=payload.session_id,
        sequence=payload.sequence,
        metrics=metrics,
        cumulative_hesitation_spikes=cumulative,
    )


@router.post("/finalize", response_model=FinalizeSessionResponse)
async def finalize_session(payload: FinalizeSessionRequest) -> FinalizeSessionResponse:
    state = session_store.get_or_create(payload.session_id)
    if state.finalized and state.audio_metrics is not None:
        audio_metrics = state.audio_metrics
    else:
        audio_metrics = finalize_session_metrics(state)
        state.audio_metrics = audio_metrics
    return FinalizeSessionResponse(
        session_id=payload.session_id,
        audio_metrics=audio_metrics,
        audio_transcript=state.full_transcript,
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def reset_session(session_id: str) -> None:
    session_store.reset(session_id)


@router.post("/analyze-file", response_model=AnalyzeFileResponse)
async def analyze_uploaded_audio(
    session_id: str = Form(...),
    transcript_text: str | None = Form(default=None),
    duration_seconds: float | None = Form(default=None),
    audio: UploadFile = File(...),
) -> AnalyzeFileResponse:
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file")

    try:
        metrics = analyze_audio_bytes(
            session_id,
            data,
            suffix=suffix,
            transcript_text=transcript_text,
            duration_hint_secs=duration_seconds,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio analysis failed: {exc}",
        ) from exc

    state = session_store.get_or_create(session_id)
    return AnalyzeFileResponse(
        session_id=session_id,
        audio_metrics=metrics,
        audio_transcript=state.full_transcript,
        chunks_processed=state.chunk_count,
    )

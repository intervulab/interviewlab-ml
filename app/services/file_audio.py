"""Analyze uploaded interview audio files (Component 3 Multer uploads)."""

from __future__ import annotations

import base64
import struct
import tempfile
from pathlib import Path

import numpy as np

from app.models.schemas import AudioEncoding, AudioMetrics
from app.core.config import get_settings
from app.services.audio_analytics import finalize_session_metrics, process_audio_chunk
from app.services.session_store import reset_acoustic_counters, session_store
from app.services.stt_local import (
    local_stt_available,
    transcribe_audio_file_with_timing,
)


def _float32_chunk_b64(samples: np.ndarray) -> str:
    payload = struct.pack(f"<{samples.size}f", *samples.astype(np.float32))
    return base64.b64encode(payload).decode("ascii")


def analyze_audio_bytes(
    session_id: str,
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
    transcript_text: str | None = None,
    duration_hint_secs: float | None = None,
) -> AudioMetrics:
    """Load audio file bytes, run chunk pipeline, optional transcript, finalize."""
    import librosa

    state = session_store.get_or_create(session_id)
    reset_acoustic_counters(state)

    incoming = (transcript_text or "").strip()
    stt_speech_secs: float | None = None

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        settings = get_settings()
        if (
            not incoming
            and settings.ENABLE_LOCAL_STT
            and local_stt_available()
        ):
            stt_text, stt_speech_secs = transcribe_audio_file_with_timing(tmp_path)
            if stt_text:
                incoming = stt_text

        if incoming and incoming not in state.transcript_parts:
            state.transcript_parts.append(incoming)

        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    chunk_samples = int(0.25 * sr)
    sequence = 0
    for start in range(0, len(y), chunk_samples):
        chunk = y[start : start + chunk_samples]
        if chunk.size == 0:
            continue
        pcm_b64 = _float32_chunk_b64(chunk)
        process_audio_chunk(
            state,
            sequence=sequence,
            pcm_base64=pcm_b64,
            encoding=AudioEncoding.FLOAT32,
            sample_rate=sr,
        )
        sequence += 1

    total_duration = len(y) / float(sr) if sr else 0.0
    hint = duration_hint_secs if duration_hint_secs and duration_hint_secs > 0 else 0.0
    wpm_duration = max(
        hint,
        total_duration,
        state.speaking_duration_secs,
        stt_speech_secs or 0.0,
    )
    if wpm_duration <= 0 and total_duration > 0:
        wpm_duration = total_duration

    answer_transcript = incoming

    metrics = finalize_session_metrics(
        state,
        transcript=answer_transcript or None,
        wpm_duration_secs=wpm_duration if answer_transcript else None,
    )
    state.audio_metrics = metrics
    return metrics

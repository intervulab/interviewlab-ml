"""Analyze uploaded interview audio files (Component 3 Multer uploads)."""

from __future__ import annotations

import base64
import struct
import tempfile
from pathlib import Path

import numpy as np

from app.models.schemas import AudioEncoding, AudioMetrics
from app.services.audio_analytics import finalize_session_metrics, process_audio_chunk
from app.services.session_store import session_store


def _float32_chunk_b64(samples: np.ndarray) -> str:
    payload = struct.pack(f"<{samples.size}f", *samples.astype(np.float32))
    return base64.b64encode(payload).decode("ascii")


def analyze_audio_bytes(
    session_id: str,
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
    transcript_text: str | None = None,
) -> AudioMetrics:
    """Load audio file bytes, run chunk pipeline, optional transcript, finalize."""
    import librosa

    session_store.reset(session_id)
    state = session_store.get_or_create(session_id)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
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

    if transcript_text and transcript_text.strip():
        state.transcript_parts.append(transcript_text.strip())

    metrics = finalize_session_metrics(state)
    state.audio_metrics = metrics
    return metrics

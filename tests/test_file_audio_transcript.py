"""Analyze-file must use transcript while computing WPM."""

import numpy as np
import struct

from app.services.file_audio import analyze_audio_bytes


def _sine_wav_bytes(duration_secs: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_secs, int(sr * duration_secs), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return struct.pack(f"<{y.size}f", *y)


def test_analyze_file_wpm_with_transcript() -> None:
    metrics = analyze_audio_bytes(
        "test-session-wpm",
        _sine_wav_bytes(),
        suffix=".wav",
        transcript_text="We use caching and a load balancer for scale",
    )
    assert metrics.clarityScore >= 0
    assert metrics.averageWPM > 0

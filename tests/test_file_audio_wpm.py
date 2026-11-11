"""WPM for analyze-file should use this answer's text and recording duration."""

import numpy as np
import struct

from app.services.file_audio import analyze_audio_bytes
from app.services.session_store import session_store


def _sine_wav_bytes(duration_secs: float = 12.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_secs, int(sr * duration_secs), endpoint=False)
    y = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return struct.pack(f"<{y.size}f", *y)


def test_analyze_file_wpm_uses_answer_transcript_not_session_cumulative() -> None:
    session_id = "test-wpm-cumulative"
    session_store.reset(session_id)
    state = session_store.get_or_create(session_id)
    state.transcript_parts.extend(["word " * 200])  # simulate prior live captions

    metrics = analyze_audio_bytes(
        session_id,
        _sine_wav_bytes(12.0),
        suffix=".wav",
        transcript_text="one two three four five six seven eight",
        duration_hint_secs=12.0,
    )
    # eight words in 12s => 40 WPM
    assert 20 <= metrics.averageWPM <= 80

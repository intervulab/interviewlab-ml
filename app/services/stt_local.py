"""Optional offline STT when browser Web Speech API is unavailable."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def local_stt_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cpu", compute_type="int8")


def transcribe_audio_file(path: str) -> str:
    """Transcribe an on-disk audio file (webm/wav). First run may download the tiny model."""
    text, _ = transcribe_audio_file_with_timing(path)
    return text


def transcribe_audio_file_with_timing(path: str) -> tuple[str, float | None]:
    """Return transcript text and Whisper segment span (seconds), if available."""
    try:
        model = _model()
        segments, _info = model.transcribe(path, beam_size=1, vad_filter=True, language="en")
        seg_list = list(segments)
        text = " ".join(seg.text.strip() for seg in seg_list if seg.text and seg.text.strip())
        speech_secs: float | None = None
        if len(seg_list) >= 1:
            speech_secs = max(0.0, float(seg_list[-1].end) - float(seg_list[0].start))
        return text.strip(), speech_secs or None
    except Exception as exc:
        logger.warning("Local STT failed for %s: %s", path, exc)
        return "", None

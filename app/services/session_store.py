"""In-memory per-session analytics state (Component 3 can add Redis later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import AudioMetrics


@dataclass
class SessionState:
    session_id: str
    transcript_parts: list[str] = field(default_factory=list)
    hesitation_spikes: int = 0
    chunk_count: int = 0
    speaking_duration_secs: float = 0.0
    pause_durations_secs: list[float] = field(default_factory=list)
    volume_samples: list[float] = field(default_factory=list)
    last_volume: float = 0.0
    last_is_speaking: bool = False
    silence_ms_carry: float = 0.0
    last_hesitation_mark_secs: float = -999.0
    wpm_samples: list[float] = field(default_factory=list)
    processed_sequences: set[int] = field(default_factory=set)
    finalized: bool = False
    audio_metrics: "AudioMetrics | None" = None

    @property
    def full_transcript(self) -> str:
        return " ".join(p.strip() for p in self.transcript_parts if p.strip())

    @property
    def word_count(self) -> int:
        text = self.full_transcript
        return len(text.split()) if text else 0


def reset_acoustic_counters(state: SessionState) -> None:
    """Clear chunk-derived acoustic stats but keep accumulated transcript text."""
    state.hesitation_spikes = 0
    state.chunk_count = 0
    state.speaking_duration_secs = 0.0
    state.pause_durations_secs.clear()
    state.volume_samples.clear()
    state.wpm_samples.clear()
    state.processed_sequences.clear()
    state.silence_ms_carry = 0.0
    state.last_volume = 0.0
    state.last_is_speaking = False
    state.last_hesitation_mark_secs = -999.0
    state.finalized = False
    state.audio_metrics = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = SessionStore()

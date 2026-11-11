"""
Pydantic schemas aligned with Component 1 (`api.types.ts`) and Component 4 injection points.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AudioEncoding(str, Enum):
    FLOAT32 = "float32"
    INT16 = "int16"


class RealTimeMetrics(BaseModel):
    """Mirrors `RealTimeMetrics` in interviewlab-frontend."""

    currentWPM: float = 0
    fillerWords: int = 0
    hesitationSpikes: int = 0
    silenceDurationMs: float = 0
    volumeLevel: float = 0
    isSpeaking: bool = False


class AudioMetrics(BaseModel):
    """Mirrors `AudioMetrics` in interviewlab-frontend (session aggregate)."""

    averageWPM: float = 0
    fillerWordCount: int = 0
    fillerWordPercentage: float = 0
    hesitationCount: int = 0
    longestPauseSecs: float = 0
    averagePauseSecs: float = 0
    confidenceScore: float = 0
    clarityScore: float = 0


class AudioChunkRequest(BaseModel):
    """Called by Component 3 when relaying a mic chunk from the WebSocket pipeline."""

    session_id: str = Field(..., min_length=1)
    sequence: int = Field(..., ge=0)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    encoding: AudioEncoding = AudioEncoding.FLOAT32
    pcm_base64: str = Field(..., description="Little-endian PCM payload for the chunk")
    duration_ms: Optional[float] = Field(
        default=None,
        description="Optional chunk duration; inferred from sample count if omitted",
    )


class AudioChunkResponse(BaseModel):
    session_id: str
    sequence: int
    metrics: RealTimeMetrics
    cumulative_hesitation_spikes: int = 0


class TranscriptSegmentRequest(BaseModel):
    """STT segments from Component 3 (or future in-service STT)."""

    session_id: str
    text: str = Field(..., min_length=1)
    is_final: bool = False
    confidence: float = Field(default=1.0, ge=0, le=1)
    speaker: Literal["candidate", "ai"] = "candidate"
    timestamp_ms: Optional[int] = None


class TranscriptSegmentResponse(BaseModel):
    session_id: str
    full_transcript: str
    word_count: int
    metrics: RealTimeMetrics


class SessionEnvelopeResponse(BaseModel):
    """
    Bundle for Component 3 → Component 4 `/interview/answer` enrichment.
    """

    session_id: str
    audio_transcript: str
    real_time_metrics: RealTimeMetrics
    audio_metrics: Optional[AudioMetrics] = None


class RubricDimensionScore(BaseModel):
    dimension: str
    score: float = Field(..., ge=0, le=100)
    rationale: str


class RubricGradeRequest(BaseModel):
    session_id: str
    question_id: str
    question_text: str
    answer_text: str = Field(..., min_length=1)
    rubric_dimensions: list[str] = Field(
        default_factory=lambda: [
            "technical_accuracy",
            "communication",
            "depth",
            "structure",
        ]
    )


class RubricGradeResponse(BaseModel):
    session_id: str
    question_id: str
    dimension_scores: list[RubricDimensionScore]
    overall_score: float = Field(..., ge=0, le=100)
    summary: str
    audio_transcript: Optional[str] = None


class FinalizeSessionRequest(BaseModel):
    session_id: str


class FinalizeSessionResponse(BaseModel):
    session_id: str
    audio_metrics: AudioMetrics
    audio_transcript: str


class AnalyzeFileResponse(BaseModel):
    session_id: str
    audio_metrics: AudioMetrics
    audio_transcript: str
    chunks_processed: int

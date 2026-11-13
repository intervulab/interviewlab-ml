"""Real-time and session-level audio cognitive analytics."""

from __future__ import annotations

import re

import numpy as np

from app.core.config import get_settings
from app.models.schemas import AudioMetrics, RealTimeMetrics
from app.services.feature_extraction import (
    chunk_duration_secs,
    decode_pcm,
    extract_mfcc_deltas,
    rms_energy,
)
from app.services.session_store import SessionState


SPEECH_RMS_THRESHOLD = 0.015
SILENCE_TAIL_MS = 120.0
PAUSE_HESITATION_SECS = 0.4


def _pause_hesitation_count(state: SessionState) -> int:
    """Meaningful mid-answer pauses (more stable than per-chunk MFCC spikes)."""
    return len([p for p in state.pause_durations_secs if p >= PAUSE_HESITATION_SECS])


def _reported_hesitation_count(state: SessionState) -> int:
    pauses = _pause_hesitation_count(state)
    if pauses > 0:
        return pauses
    return min(state.hesitation_spikes, 3)


def count_filler_words(text: str, fillers: list[str]) -> int:
    lowered = text.lower()
    total = 0
    for token in fillers:
        if " " in token:
            total += len(re.findall(re.escape(token), lowered))
        else:
            total += len(re.findall(rf"\b{re.escape(token)}\b", lowered))
    return total


def process_audio_chunk(
    state: SessionState,
    *,
    sequence: int,
    pcm_base64: str,
    encoding,
    sample_rate: int,
) -> tuple[RealTimeMetrics, int]:
    """Update session state from one PCM chunk; return live metrics."""
    settings = get_settings()

    if sequence in state.processed_sequences:
        return _build_realtime_metrics(state), state.hesitation_spikes

    y = decode_pcm(pcm_base64, encoding)
    if y.size == 0:
        state.processed_sequences.add(sequence)
        return _build_realtime_metrics(state), state.hesitation_spikes

    duration_secs = chunk_duration_secs(y.size, sample_rate)
    volume = min(1.0, rms_energy(y) * 12.0)
    is_speaking = volume >= SPEECH_RMS_THRESHOLD

    mfcc_deltas = extract_mfcc_deltas(y, sample_rate)
    spike_frames = int(np.sum(mfcc_deltas > settings.HESITATION_MFCC_DELTA_THRESHOLD))
    n_frames = max(1, mfcc_deltas.size)
    spike_ratio = spike_frames / n_frames
    # Require sustained instability in the chunk, not a single noisy frame (was always ~100/100).
    if (
        is_speaking
        and spike_frames >= 6
        and spike_ratio >= 0.55
        and state.speaking_duration_secs - state.last_hesitation_mark_secs >= 2.0
    ):
        state.hesitation_spikes += 1
        state.last_hesitation_mark_secs = state.speaking_duration_secs

    if is_speaking:
        state.speaking_duration_secs += duration_secs
        if not state.last_is_speaking and state.silence_ms_carry > SILENCE_TAIL_MS:
            state.pause_durations_secs.append(state.silence_ms_carry / 1000.0)
        state.silence_ms_carry = 0.0
    else:
        state.silence_ms_carry += duration_secs * 1000.0

    state.volume_samples.append(volume)
    state.last_volume = volume
    state.last_is_speaking = is_speaking
    state.chunk_count += 1
    state.processed_sequences.add(sequence)

    transcript = state.full_transcript
    if state.speaking_duration_secs > 0 and state.word_count > 0:
        wpm = (state.word_count / state.speaking_duration_secs) * 60.0
        state.wpm_samples.append(wpm)

    fillers = count_filler_words(transcript, settings.filler_word_list)

    metrics = RealTimeMetrics(
        currentWPM=round(state.wpm_samples[-1], 1) if state.wpm_samples else 0.0,
        fillerWords=fillers,
        hesitationSpikes=_reported_hesitation_count(state),
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(volume, 3),
        isSpeaking=is_speaking,
    )
    return metrics, _reported_hesitation_count(state)


def update_transcript_metrics(state: SessionState) -> RealTimeMetrics:
    settings = get_settings()
    transcript = state.full_transcript
    fillers = count_filler_words(transcript, settings.filler_word_list)

    if state.speaking_duration_secs > 0 and state.word_count > 0:
        wpm = (state.word_count / state.speaking_duration_secs) * 60.0
        state.wpm_samples.append(wpm)

    return RealTimeMetrics(
        currentWPM=round(state.wpm_samples[-1], 1) if state.wpm_samples else 0.0,
        fillerWords=fillers,
        hesitationSpikes=_reported_hesitation_count(state),
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(state.last_volume, 3),
        isSpeaking=state.last_is_speaking,
    )


def finalize_session_metrics(
    state: SessionState,
    *,
    transcript: str | None = None,
    wpm_duration_secs: float | None = None,
) -> AudioMetrics:
    settings = get_settings()
    transcript = (transcript if transcript is not None else state.full_transcript).strip()
    words = len(transcript.split()) if transcript else 0
    fillers = count_filler_words(transcript, settings.filler_word_list)
    filler_pct = (fillers / words * 100.0) if words else 0.0

    avg_wpm = float(np.mean(state.wpm_samples)) if state.wpm_samples else 0.0
    pace_denom = wpm_duration_secs if wpm_duration_secs and wpm_duration_secs > 0 else state.speaking_duration_secs
    if pace_denom > 0.1 and words > 0:
        pace_from_transcript = (words / pace_denom) * 60.0
        # Uploaded answers: wall-clock / hint beats VAD-only time (and session-cumulative transcript).
        if wpm_duration_secs or avg_wpm <= 0:
            avg_wpm = pace_from_transcript
        elif avg_wpm > pace_from_transcript * 1.25:
            avg_wpm = pace_from_transcript
    elif avg_wpm <= 0 and state.speaking_duration_secs > 0.1 and words > 0:
        avg_wpm = (words / state.speaking_duration_secs) * 60.0

    # Typical conversational speech rarely exceeds ~220 WPM; widen denominator if math says otherwise
    # (often duplicate live captions + final STT, or VAD under-counting speech time).
    if avg_wpm > 220 and words > 0:
        min_secs = (words * 60.0) / 220.0
        if pace_denom < min_secs:
            avg_wpm = 220.0
    pauses = state.pause_durations_secs
    longest_pause = max(pauses) if pauses else 0.0
    avg_pause = float(np.mean(pauses)) if pauses else 0.0

    vol_std = float(np.std(state.volume_samples)) if state.volume_samples else 0.0
    vol_mean = float(np.mean(state.volume_samples)) if state.volume_samples else 0.0

    reported_hesitation = _reported_hesitation_count(state)

    # Heuristic confidence/clarity scores (0–100) from acoustic + disfluency signals
    hesitation_penalty = min(40.0, reported_hesitation * 4.0)
    filler_penalty = min(30.0, filler_pct * 1.5)
    volume_stability = max(0.0, 100.0 - vol_std * 200.0)
    speaking_presence = min(100.0, vol_mean * 120.0)

    confidence = max(
        0.0,
        min(100.0, 70.0 + speaking_presence * 0.2 - hesitation_penalty - filler_penalty * 0.3),
    )
    clarity = max(
        0.0,
        min(100.0, volume_stability * 0.6 + (100.0 - filler_penalty * 2) * 0.4),
    )

    state.finalized = True
    return AudioMetrics(
        averageWPM=round(avg_wpm, 1),
        fillerWordCount=fillers,
        fillerWordPercentage=round(filler_pct, 2),
        hesitationCount=reported_hesitation,
        longestPauseSecs=round(longest_pause, 2),
        averagePauseSecs=round(avg_pause, 2),
        confidenceScore=round(confidence, 1),
        clarityScore=round(clarity, 1),
    )


def _build_realtime_metrics(state: SessionState) -> RealTimeMetrics:
    settings = get_settings()
    transcript = state.full_transcript
    fillers = count_filler_words(transcript, settings.filler_word_list)
    return RealTimeMetrics(
        currentWPM=round(state.wpm_samples[-1], 1) if state.wpm_samples else 0.0,
        fillerWords=fillers,
        hesitationSpikes=_reported_hesitation_count(state),
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(state.last_volume, 3),
        isSpeaking=state.last_is_speaking,
    )

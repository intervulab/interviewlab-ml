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
    spike_count = int(
        np.sum(mfcc_deltas > settings.HESITATION_MFCC_DELTA_THRESHOLD)
    )
    if is_speaking and spike_count > 0:
        # One hesitation event per chunk max (avoids frame-level noise inflation)
        state.hesitation_spikes += 1

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
        hesitationSpikes=state.hesitation_spikes,
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(volume, 3),
        isSpeaking=is_speaking,
    )
    return metrics, state.hesitation_spikes


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
        hesitationSpikes=state.hesitation_spikes,
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(state.last_volume, 3),
        isSpeaking=state.last_is_speaking,
    )


def finalize_session_metrics(state: SessionState) -> AudioMetrics:
    settings = get_settings()
    transcript = state.full_transcript
    words = state.word_count
    fillers = count_filler_words(transcript, settings.filler_word_list)
    filler_pct = (fillers / words * 100.0) if words else 0.0

    avg_wpm = float(np.mean(state.wpm_samples)) if state.wpm_samples else 0.0
    pauses = state.pause_durations_secs
    longest_pause = max(pauses) if pauses else 0.0
    avg_pause = float(np.mean(pauses)) if pauses else 0.0

    vol_std = float(np.std(state.volume_samples)) if state.volume_samples else 0.0
    vol_mean = float(np.mean(state.volume_samples)) if state.volume_samples else 0.0

    # Heuristic confidence/clarity scores (0–100) from acoustic + disfluency signals
    hesitation_penalty = min(40.0, state.hesitation_spikes * 4.0)
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
        hesitationCount=state.hesitation_spikes,
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
        hesitationSpikes=state.hesitation_spikes,
        silenceDurationMs=round(state.silence_ms_carry, 1),
        volumeLevel=round(state.last_volume, 3),
        isSpeaking=state.last_is_speaking,
    )

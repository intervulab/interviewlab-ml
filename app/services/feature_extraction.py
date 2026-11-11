"""PCM decoding and Librosa MFCC feature extraction."""

from __future__ import annotations

import base64
import struct

import numpy as np

from app.models.schemas import AudioEncoding


def decode_pcm(
    pcm_base64: str,
    encoding: AudioEncoding,
) -> np.ndarray:
    """Decode base64 PCM to mono float32 in [-1, 1]."""
    raw = base64.b64decode(pcm_base64)
    if encoding == AudioEncoding.FLOAT32:
        count = len(raw) // 4
        samples = struct.unpack(f"<{count}f", raw[: count * 4])
        return np.asarray(samples, dtype=np.float32)
    # int16 little-endian
    count = len(raw) // 2
    samples = struct.unpack(f"<{count}h", raw[: count * 2])
    arr = np.asarray(samples, dtype=np.float32)
    return arr / 32768.0


def chunk_duration_secs(num_samples: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return num_samples / float(sample_rate)


def extract_mfcc_deltas(y: np.ndarray, sample_rate: int) -> np.ndarray:
    """Frame-level MFCC delta magnitude (proxy for vocal hesitation / instability)."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for audio analytics") from exc

    if y.size < 64:
        return np.zeros(1, dtype=np.float32)

    # Librosa delta defaults to width=9; very short clips yield <9 MFCC frames and crash scipy.
    min_samples = int(0.35 * sample_rate)
    if y.size < min_samples:
        y = np.pad(y, (0, min_samples - y.size), mode="constant")

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    n_frames = int(mfcc.shape[1])
    if n_frames < 3:
        return np.zeros(1, dtype=np.float32)

    width = 9
    if n_frames < width:
        width = n_frames if n_frames % 2 == 1 else n_frames - 1
    if width < 3:
        return np.linalg.norm(np.diff(mfcc, axis=1), axis=0)

    delta = librosa.feature.delta(mfcc, width=width)
    return np.linalg.norm(delta, axis=0)


def rms_energy(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))

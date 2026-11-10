import base64
import struct

import numpy as np


def _tone_chunk(freq: float = 440.0, duration_secs: float = 0.25, sr: int = 16000) -> str:
    t = np.linspace(0, duration_secs, int(sr * duration_secs), endpoint=False)
    wave = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    payload = struct.pack(f"<{wave.size}f", *wave)
    return base64.b64encode(payload).decode("ascii")


def _silence_chunk(duration_secs: float = 0.25, sr: int = 16000) -> str:
    wave = np.zeros(int(sr * duration_secs), dtype=np.float32)
    payload = struct.pack(f"<{wave.size}f", *wave)
    return base64.b64encode(payload).decode("ascii")


def test_audio_chunk_and_finalize(client):
    session_id = "sess_audio_1"
    for seq in range(4):
        pcm = _tone_chunk() if seq % 2 == 0 else _silence_chunk()
        res = client.post(
            "/api/v1/audio/chunk",
            json={
                "session_id": session_id,
                "sequence": seq,
                "sample_rate": 16000,
                "encoding": "float32",
                "pcm_base64": pcm,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["sequence"] == seq
        assert "metrics" in body

    client.post(
        "/api/v1/transcript/segment",
        json={
            "session_id": session_id,
            "text": "I would use a load balancer um for scale",
            "is_final": True,
            "speaker": "candidate",
        },
    )

    fin = client.post("/api/v1/audio/finalize", json={"session_id": session_id})
    assert fin.status_code == 200
    metrics = fin.json()["audio_metrics"]
    assert metrics["fillerWordCount"] >= 1
    assert "averageWPM" in metrics


def test_idempotent_audio_sequence(client):
    session_id = "sess_audio_2"
    pcm = _tone_chunk()
    payload = {
        "session_id": session_id,
        "sequence": 0,
        "sample_rate": 16000,
        "encoding": "float32",
        "pcm_base64": pcm,
    }
    first = client.post("/api/v1/audio/chunk", json=payload).json()
    second = client.post("/api/v1/audio/chunk", json=payload).json()
    assert first["metrics"] == second["metrics"]

# Component 5 Integration Guide

Component 3 (Node.js gateway) should call this service; the browser never talks to port `8001` directly.

## Service URL

| Environment | Base URL |
|-------------|----------|
| Local | `http://localhost:8001` |
| Docker network | `http://ml-analytics:8001` |

Component 4 (orchestrator) runs on **8000**. This service runs on **8001**.

## Recommended Gateway Flow

```
Browser ──WebSocket binary PCM──► Component 3
Component 3 ──POST /api/v1/audio/chunk──► Component 5
Component 5 ──RealTimeMetrics JSON──► Component 3 ──metrics:update──► Browser

STT (any provider) ──POST /api/v1/transcript/segment──► Component 5

On answer submit:
Component 3 ──GET /api/v1/sessions/{id}/envelope──► Component 5
Component 3 ──POST /api/v1/interview/answer──► Component 4
  (inject `audio_transcript` from envelope)

On session end:
Component 3 ──POST /api/v1/audio/finalize──► Component 5
  (store `audio_metrics` on interview report)
```

## API Summary

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/audio/chunk` | Process one PCM chunk (base64 float32 mono 16kHz) |
| `POST` | `/api/v1/audio/finalize` | Session-level `AudioMetrics` |
| `POST` | `/api/v1/transcript/segment` | Append STT text (WPM + filler stats) |
| `GET` | `/api/v1/sessions/{session_id}/envelope` | Bundle for orchestrator enrichment |
| `POST` | `/api/v1/rubric/grade` | Rubric scores for an answer |
| `DELETE` | `/api/v1/audio/session/{session_id}` | Reset in-memory session state |

### PCM chunk body

```json
{
  "session_id": "session_abc123",
  "sequence": 42,
  "sample_rate": 16000,
  "encoding": "float32",
  "pcm_base64": "<little-endian float32 bytes>"
}
```

Response `metrics` object matches Component 1 `RealTimeMetrics` and WebSocket `metrics:update`.

### Component 4 answer enrichment

```python
envelope = httpx.get(f"{ML_URL}/api/v1/sessions/{session_id}/envelope").json()
payload = {
    "session_id": session_id,
    "question_id": question_id,
    "candidate_answer": typed_answer,
    "audio_transcript": envelope["audio_transcript"],
}
httpx.post(f"{ORCHESTRATOR_URL}/api/v1/interview/answer", json=payload)
```

## Type alignment

Schemas in `app/models/schemas.py` mirror:

- `interviewlab-frontend` → `src/types/api.types.ts` (`AudioMetrics`, `RealTimeMetrics`)
- `interviewlab-agent-orchestrator` → `AnswerRequest.audio_transcript`

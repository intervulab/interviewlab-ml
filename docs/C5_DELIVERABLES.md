# Component 5 — deliverables & acceptance

What C5 owns in the integrated demo, and how to verify it.

## APIs (port 8001)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/audio/chunk` | Live PCM → `RealTimeMetrics` |
| `POST /api/v1/transcript/segment` | Append STT/typed text → WPM & fillers |
| `POST /api/v1/audio/analyze-file` | Uploaded answer WebM/WAV → `AudioMetrics` |
| `GET /api/v1/sessions/{id}/envelope` | Transcript + metrics for C4 |
| `POST /api/v1/audio/finalize` | Session-level `AudioMetrics` |
| `POST /api/v1/rubric/grade` | Text rubric when C4 unavailable |

## What appears in the product (your output)

| Surface | C5 fields |
|---------|-----------|
| Session HUD | `currentWPM`, `fillerWords`, `hesitationSpikes`, `volumeLevel` |
| Answer submit | `audio_metrics` stored on `AudioAnalysis` (via C3) |
| Report — **Audio WPM** tile | `detailedFeedback.audioMetrics.averageWPM` |
| Report — **Communication / audio** | `averageWPM`, `fillerWordCount`, `clarityScore` |
| Report — **Overall** (indirect) | `POST /rubric/grade` → evaluation score |

Not C5: recommendation, technical HR paragraph, interview questions (C4).

## Integration contract (C3 must)

1. Stream PCM to `/audio/chunk` on WebSocket.
2. Forward `transcript:final` to `/transcript/segment` **before** answer submit.
3. On answer with audio: `multipart` upload + `transcript_text` when available.
4. Persist `audio_metrics` from `analyze-file` on each answer.
5. On report: aggregate persisted `AudioAnalysis` rows (do not rely only on in-memory finalize).

## Acceptance commands

```bash
# Health
curl -s http://localhost:8001/health | jq .

# Transcript + envelope
curl -s -X POST http://localhost:8001/api/v1/transcript/segment \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","text":"I would use Redis and token buckets for rate limiting um basically","is_final":true,"speaker":"candidate"}' | jq .

curl -s http://localhost:8001/api/v1/sessions/demo/envelope | jq .

# Rubric
curl -s -X POST http://localhost:8001/api/v1/rubric/grade \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","question_id":"q1","question_text":"Explain rate limiting","answer_text":"Token bucket in Redis with sliding windows"}' | jq .

# Tests
cd intervulab && pytest -q
```

## Known limits (document, do not hide)

- Browser STT (Chrome/Brave) uses Google servers — fails with `network` when offline or blocked.
- **Fallback:** `uv pip install -r requirements-stt.txt` (`faster-whisper` tiny; pin **`av<19`** for PyAV compatibility) and set `ENABLE_LOCAL_STT=true` for transcription on `analyze-file`.
- In-memory session store: restarted ML process loses live state; persisted rows on C3 are source of truth for reports.
- Rubric default mode is heuristic, not a trained transformer unless `requirements-ml.txt` + training are used.

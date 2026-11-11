# InterviewLab — Component 5: ML & Audio Cognitive Analytics

> Part of the [intervulab](https://github.com/intervulab) major project.  
> FastAPI microservice for **MFCC-based hesitation detection**, **live speech metrics**, and **rubric grading** of answer transcripts.

| Peer repo | Role |
|-----------|------|
| [interviewlab-frontend](https://github.com/intervulab/interviewlab-frontend) | Streams 16 kHz PCM chunks (via Component 3) |
| [interviewlab-system-canvas](https://github.com/intervulab/interviewlab-system-canvas) | System design UI |
| [interviewlab-agent-orchestrator](https://github.com/intervulab/interviewlab-agent-orchestrator) | Consumes `audio_transcript` on `/interview/answer` |

## Features

- **Audio pipeline**: decode WebSocket-style PCM chunks → Librosa MFCC deltas → hesitation spikes, volume, pauses
- **Transcript analytics**: WPM, filler-word density (aligned with frontend `AudioMetrics`)
- **Rubric ML**: heuristic scoring by default; optional DistilBERT fine-tuning (`scripts/train_rubric.py`)
- **Gateway-friendly API** for Component 3 to relay `metrics:update` and enrich Component 4

## Quick start

```bash
git clone https://github.com/intervulab/interviewlab-ml.git
cd interviewlab-ml

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload --port 8001
```

- Swagger UI: http://localhost:8001/docs  
- Health: http://localhost:8001/health  

### Optional transformer rubric

```bash
pip install -r requirements-ml.txt
# set RUBRIC_INFERENCE_MODE=transformer in .env
python scripts/train_rubric.py --data data/rubric_samples.jsonl
# set RUBRIC_MODEL_PATH=models/checkpoints/rubric
```

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pip install pytest
pytest
```

## Project layout

```
app/
  main.py
  api/           # audio, transcript, rubric, session routes
  services/      # MFCC analytics, rubric scorer, session store
  models/        # Pydantic schemas (frontend-aligned)
scripts/
  train_rubric.py
docs/
  INTEGRATION.md
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for Component 3 wiring.  
**Deliverables & acceptance:** [docs/C5_DELIVERABLES.md](docs/C5_DELIVERABLES.md).

**Full five-repo local deploy** (humans or LLM agents): [`backend` repo `docs/LOCAL_DEPLOYMENT.md`](https://github.com/intervulab/backend/blob/integration/demo/docs/LOCAL_DEPLOYMENT.md) on branch `integration/demo`.

## License

MIT © InterviewLab Team

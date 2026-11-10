"""
FastAPI entry point — Component 5: ML & Audio Cognitive Analytics.

Run:
  uvicorn app.main:app --reload --port 8001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.audio_routes import router as audio_router
from app.api.rubric_routes import router as rubric_router
from app.api.session_routes import router as session_router
from app.api.transcript_routes import router as transcript_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="InterviewLab ML & Audio Analytics",
    description="""
## Component 5 — Custom ML Fine-Tuning & Audio Cognitive Analytics

Processes streaming PCM audio (MFCC hesitation spikes, pacing, pauses) and grades
answer transcripts against rubric dimensions. Component 3 proxies WebSocket audio here;
results feed the frontend `metrics:update` payload and Component 4 `audio_transcript`.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio_router, prefix="/api/v1")
app.include_router(transcript_router, prefix="/api/v1")
app.include_router(rubric_router, prefix="/api/v1")
app.include_router(session_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "interviewlab-ml",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
        "rubric_mode": settings.RUBRIC_INFERENCE_MODE,
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "InterviewLab ML & Audio Analytics is running",
        "docs": "/docs",
    }

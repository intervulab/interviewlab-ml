import os

import pytest
from fastapi.testclient import TestClient

# Tests always use lightweight heuristics (no HF model downloads).
os.environ.setdefault("RUBRIC_INFERENCE_MODE", "heuristic")

from app.main import app  # noqa: E402
from app.services.session_store import session_store  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_sessions():
    session_store._sessions.clear()
    yield
    session_store._sessions.clear()

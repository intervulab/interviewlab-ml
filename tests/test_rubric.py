def test_rubric_grade(client):
    session_id = "sess_rubric"
    client.post(
        "/api/v1/transcript/segment",
        json={
            "session_id": session_id,
            "text": "First I consider trade-offs, then I pick PostgreSQL because of consistency",
            "is_final": True,
            "speaker": "candidate",
        },
    )

    res = client.post(
        "/api/v1/rubric/grade",
        json={
            "session_id": session_id,
            "question_id": "q1",
            "question_text": "How would you design the database layer?",
            "answer_text": (
                "First I analyze requirements. Second I consider trade-offs between SQL and NoSQL. "
                "Finally I choose PostgreSQL with read replicas for scale."
            ),
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["overall_score"] > 0
    assert len(body["dimension_scores"]) == 4
    assert body["audio_transcript"]


def test_session_envelope(client):
    session_id = "sess_env"
    client.post(
        "/api/v1/transcript/segment",
        json={
            "session_id": session_id,
            "text": "Structured answer with clear steps",
            "speaker": "candidate",
        },
    )
    env = client.get(f"/api/v1/sessions/{session_id}/envelope")
    assert env.status_code == 200
    data = env.json()
    assert data["audio_transcript"]
    assert data["real_time_metrics"]["currentWPM"] >= 0

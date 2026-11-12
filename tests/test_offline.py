"""
Offline tests: a FAKE provider stands in for Mistral/OpenAI, so these run
with no API key and cost nothing. They prove:
  - /chat streams multiple pieces
  - /quiz returns schema-valid JSON (and rejects bad JSON)
  - a 429 makes the app wait and retry
  - switching LLM_PROVIDER works with no code changes
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import pytest
from fastapi.testclient import TestClient

import llm
from main import app

QUIZ = {"topic": "Python", "questions": [{
    "question": "What keyword defines a function?",
    "options": ["def", "fun", "func", "lambda"],
    "answer_index": 0, "explanation": "def starts a function definition."}]}


def make_fake(state, quiz_payload=QUIZ):
    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        state["urls"].append(str(request.url))
        if state["calls"] <= state["fail_first"]:
            return httpx.Response(429, headers={"retry-after": "0"})
        body = json.loads(request.content)
        if body.get("stream"):
            sse = "".join(
                f'data: {json.dumps({"choices": [{"delta": {"content": w}}]})}\n\n'
                for w in ["Hello", " from", " the", " model"]) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode())
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(quiz_payload)}}]})
    return handler


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.01")
    state = {"calls": 0, "fail_first": 0, "urls": []}
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(make_fake(state)))
    yield state
    llm._client = None


@pytest.mark.parametrize("provider,host", [("mistral", "mistral.ai"), ("openai", "openai.com"), ("groq", "groq.com")])
def test_both_providers(setup, monkeypatch, provider, host):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    c = TestClient(app)
    with c.stream("POST", "/chat", json={"message": "hi"}) as r:
        pieces = list(r.iter_text())
        assert r.status_code == 200
    assert "".join(pieces) == "Hello from the model"
    q = c.post("/quiz", json={"topic": "Python", "num_questions": 1})
    assert q.status_code == 200 and q.json()["questions"][0]["answer_index"] == 0
    assert all(host in u for u in setup["urls"])


def test_429_is_retried(setup, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    setup["fail_first"] = 2  # first two calls get 429
    r = TestClient(app).post("/quiz", json={"topic": "Python", "num_questions": 1})
    assert r.status_code == 200
    assert setup["calls"] == 3


def test_429_gives_up(setup, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    setup["fail_first"] = 99
    r = TestClient(app).post("/quiz", json={"topic": "Python", "num_questions": 1})
    assert r.status_code == 429
    assert setup["calls"] == 3


def test_bad_quiz_json_rejected(setup, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    bad = {"topic": "x", "questions": [{"question": "q", "options": ["only", "two"],
                                        "answer_index": 9, "explanation": "e"}]}
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(make_fake(setup, bad)))
    r = TestClient(app).post("/quiz", json={"topic": "x", "num_questions": 1})
    assert r.status_code == 502


def test_unknown_provider(setup, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    r = TestClient(app).get("/health")
    assert r.json()["provider"] == "nope"
    r = TestClient(app).post("/quiz", json={"topic": "x"})
    assert r.status_code == 500

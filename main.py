"""
main.py - the FastAPI app (the "web server" part).

It never mentions Mistral or OpenAI. All provider details live in llm.py.
"""
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

import llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await llm.close_client()  # tidy up when the server shuts down


app = FastAPI(title="AI Study Assistant", lifespan=lifespan)


@app.exception_handler(llm.LLMError)
async def llm_error_handler(request, exc: llm.LLMError):
    """Turn provider problems into clean JSON errors (429, 502, ...)."""
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})


# ------------------------------------------------------------------ models
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = []


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1)
    num_questions: int = Field(default=3, ge=1, le=10)


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion] = Field(min_length=1)


SYSTEM_PROMPT = (
    "You are a friendly study assistant. Explain things clearly and simply, "
    "using short examples. If you are unsure, say so."
)


# --------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "provider": llm.provider_name()}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Streams the answer to the client piece by piece."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    gen = llm.stream_chat(messages)

    # Pull the FIRST piece before replying. If the provider is down or the key
    # is wrong, we can still return a proper error code instead of a broken stream.
    try:
        first = await anext(gen)
    except StopAsyncIteration:
        first = ""

    async def body():
        yield first
        try:
            async for piece in gen:
                yield piece
        except llm.LLMError as exc:  # headers are already sent; add a note instead
            yield f"\n[stream interrupted: {exc}]"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


@app.post("/quiz", response_model=Quiz)
async def quiz(req: QuizRequest):
    """Asks the model for JSON, then validates it with Pydantic."""
    system = (
        "You write multiple-choice quizzes. Reply with ONLY a JSON object shaped like:\n"
        '{"topic": "...", "questions": [{"question": "...", '
        '"options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "..."}]}\n'
        "Each question must have exactly 4 options. answer_index is the 0-based "
        "position of the correct option."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Write {req.num_questions} questions about: {req.topic}. Respond in JSON."},
    ]

    last_problem = ""
    for _ in range(2):  # one try + one "please fix it" retry
        raw = await llm.complete(messages, json_mode=True)
        try:
            result = Quiz.model_validate(json.loads(raw))
            if len(result.questions) != req.num_questions:
                raise ValueError(
                    f"expected {req.num_questions} questions, got {len(result.questions)}"
                )
            return result
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_problem = str(exc)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON was invalid ({last_problem[:300]}). Fix it and reply with JSON only."},
            ]

    raise HTTPException(502, f"Model did not return valid quiz JSON: {last_problem[:300]}")

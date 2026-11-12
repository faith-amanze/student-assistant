# AI Study Assistant (FastAPI)

A small web service that chats with an AI model and generates quizzes.
It works with **Groq**, **Mistral** or **OpenAI**; you switch by changing one environment variable.

| Endpoint | What it does |
|---|---|
| `GET /health` | Shows the server is up and which provider is active |
| `POST /chat` | Streams the AI's answer word-by-word (`StreamingResponse`) |
| `POST /quiz` | Returns a multiple-choice quiz as JSON, validated by Pydantic |

## Project layout

```
study-assistant/
├── main.py            # FastAPI app: endpoints + Pydantic quiz models
├── llm.py             # ALL provider-specific code (URLs, keys, retries, streaming)
├── tests/test_offline.py   # tests using a fake provider (no key needed)
├── requirements.txt
├── .env.example       # template for your secrets
├── .gitignore         # contains .env so your keys never get uploaded
└── README.md
```

## Setup (step by step)

You need Python 3.10 or newer.

```bash
# 1. Go into the project folder
cd study-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the libraries
pip install -r requirements.txt

# 4. Create your secrets file
cp .env.example .env               # Windows: copy .env.example .env
# Now open .env in an editor and paste in your API key(s)

# 5. Start the server
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for an automatic, clickable API page.

## Environment variables

Set these in `.env`.

| Variable | Required? | Default | Meaning |
|---|---|---|---|
| `LLM_PROVIDER` | No | `mistral` | `groq`, `mistral` or `openai` |
| `GROQ_API_KEY` | If provider is groq | none | Get one at console.groq.com/keys |
| `MISTRAL_API_KEY` | If provider is mistral | none | Get one at console.mistral.ai |
| `OPENAI_API_KEY` | If provider is openai | none | Get one at platform.openai.com |
| `GROQ_MODEL` | No | `openai/gpt-oss-20b` | Groq model name |
| `MISTRAL_MODEL` | No | `mistral-small-latest` | Mistral model name |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `LLM_MAX_RETRIES` | No | `5` | How many times to retry after a 429 |
| `LLM_BACKOFF_BASE` | No | `1.0` | Seconds to wait before first retry (doubles each time) |

Model names and rate limits change over time. Check the provider's docs if you get a "model not found" error.

## Try it

**Chat** (the `-N` flag stops curl from buffering, so you see words arrive live):

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain photosynthesis in 3 sentences"}'
```

**Quiz:**

```bash
curl -X POST http://127.0.0.1:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"topic": "Python lists", "num_questions": 3}'
```

Expected shape:

```json
{"topic": "Python lists",
 "questions": [{"question": "...", "options": ["A","B","C","D"],
                "answer_index": 1, "explanation": "..."}]}
```

## Switching providers

1. Stop the server (Ctrl+C).
2. In `.env` change `LLM_PROVIDER` (e.g. `groq` to `mistral` or `openai`) and make sure that provider's API key is set.
3. Start the server again and call `/health`. It should show the new provider name.
4. Run the same `/chat` and `/quiz` commands. No code changes needed.

## How it works (for learners)

- **Streaming:** `/chat` returns a `StreamingResponse`. `llm.stream_chat` reads the provider's
  stream and `yield`s each piece immediately.
- **Async:** every model call uses `async`/`await` (via `httpx.AsyncClient`), so while one
  request waits for the AI, the server can handle others.
- **Structured JSON:** `/quiz` sends `response_format={"type": "json_object"}`, then
  checks the result with the `Quiz` Pydantic model. If the JSON is wrong it asks the model
  to fix it once, then returns a 502 error.
- **Rate limits:** on HTTP 429 the app waits (using the `Retry-After` header if present,
  otherwise 1s, 2s, 4s... with a little random jitter) and retries up to `LLM_MAX_RETRIES` times.
  If it still fails, the client gets a clear 429 JSON error.
- **Provider-agnostic:** `main.py` only calls `llm.stream_chat` and `llm.complete`.
  The `PROVIDERS` table in `llm.py` (groq, mistral, openai) holds each provider's URL, key name and default model.

## Using Groq (free tier)

Groq has a free tier. Sign up at console.groq.com, create a key under API Keys, and set
`LLM_PROVIDER=groq` and `GROQ_API_KEY=...` in `.env`. Free-tier limits are low (roughly 30 requests
per minute and a few thousand tokens per minute, at the time of writing), so you may see 429s.
The app retries automatically. Check console.groq.com/docs/rate-limits for current numbers.

## Tests

```bash
python -m pytest -v
```

These use a fake provider, so they need no API key. They check streaming, quiz validation,
429 retry/give-up behaviour, and both provider settings.

## Evidence checklist (for submission)

After running with a real key, save the terminal output of:
1. the `curl -N` chat command showing text arriving gradually,
2. the `/quiz` command showing valid JSON,
3. both again after switching `LLM_PROVIDER`.

## Security notes

- `.env` is in `.gitignore`. Never paste your API key into code or commit it.
- If a key ever leaks, delete it in the provider's console and make a new one.

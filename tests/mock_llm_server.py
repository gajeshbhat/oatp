"""
Standalone mock OpenAI-compatible LLM server.

Used during integration tests so that crewAI / AutoGen can make real HTTP
calls through the OAT proxy without needing an actual OpenAI API key.

    uvicorn tests.mock_llm_server:app --port 9120
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM Server")


def _build_response(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "gpt-4o")
    messages = body.get("messages", [])

    # Produce a deterministic reply based on the last user message.
    last_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_msg = m.get("content", "")
            break

    reply = f"Mock response to: {last_msg[:80]}"

    if body.get("stream"):
        return _stream(model, reply)

    return JSONResponse(content=_build_response(model, reply))


def _stream(model: str, content: str) -> StreamingResponse:
    async def generate():
        # Single content chunk
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        # Final chunk with finish_reason + usage
        final = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    """Minimal /v1/models endpoint — some frameworks probe this on init."""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "oatp-mock"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "oatp-mock"},
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

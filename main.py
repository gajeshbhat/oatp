"""
OAT Proxy — Open Agent Telemetry interceptor.

Drop-in replacement for an OpenAI-compatible endpoint.  Agents point their
`base_url` here and get automatic distributed tracing + semantic loop
detection for free.

Usage:
    UPSTREAM_BASE_URL=https://api.openai.com  uvicorn main:app --port 9119
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace import SpanKind, StatusCode

from logic.guardrails import SemanticGuardrail
from logic.stitcher import (
    HEADER_TRACE_ID,
    OATP_TRACE_ID_ATTR,
    build_parent_context,
    framework_attributes,
    get_tracer,
    request_attributes,
    resolve_trace_id,
    response_attributes,
    shutdown_tracer,
)

logger = logging.getLogger("oatp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# ── Lifespan ───────────────────────────────────────────────────────────────
_guardrail: SemanticGuardrail | None = None
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _guardrail, _http_client
    _http_client = httpx.AsyncClient(base_url=UPSTREAM_BASE_URL, timeout=120)
    try:
        _guardrail = SemanticGuardrail(qdrant_url=QDRANT_URL)
        logger.info("Semantic guardrail initialised (Qdrant @ %s)", QDRANT_URL)
    except Exception:
        logger.warning("Qdrant unavailable – guardrail disabled", exc_info=True)
        _guardrail = None
    yield
    await _http_client.aclose()
    shutdown_tracer()


app = FastAPI(title="OAT Proxy", version="0.1.0", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_prompt_text(messages: list[dict]) -> str:
    """Concatenate the last few user/assistant messages into a single string."""
    parts: list[str] = []
    for msg in messages[-3:]:
        content = msg.get("content", "")
        if isinstance(content, list):  # vision-style content
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        parts.append(content)
    return "\n".join(parts)


def _forward_headers(request: Request) -> dict[str, str]:
    """Pass through auth and content-type headers to upstream."""
    headers: dict[str, str] = {}
    if auth := request.headers.get("authorization"):
        headers["Authorization"] = auth
    headers["Content-Type"] = "application/json"
    return headers



# ── Main route ─────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    tracer = get_tracer()

    # ── 1. Context Stitching ───────────────────────────────────────────────
    oat_trace_id = resolve_trace_id(request.headers.get(HEADER_TRACE_ID))
    parent_ctx = build_parent_context(oat_trace_id)
    fw_attrs = framework_attributes(dict(request.headers))

    with tracer.start_as_current_span(
        name=f"chat {body.get('model', 'unknown')}",
        kind=SpanKind.SERVER,
        context=parent_ctx,
        attributes={
            OATP_TRACE_ID_ATTR: oat_trace_id,
            **request_attributes(body),
            **fw_attrs,
        },
    ) as server_span:

        # ── 2. Semantic Guardrail ──────────────────────────────────────────
        if _guardrail is not None:
            prompt_text = _extract_prompt_text(body.get("messages", []))
            with tracer.start_as_current_span(
                "semantic_guardrail", kind=SpanKind.INTERNAL
            ) as guard_span:
                result = await _guardrail.check(oat_trace_id, prompt_text)
                guard_span.set_attribute("guardrail.max_similarity", result.max_similarity)
                guard_span.set_attribute("guardrail.checked_count", result.checked_count)
                if result.is_loop:
                    guard_span.add_event("loop_detected", {"similarity": result.max_similarity})
                    server_span.set_status(StatusCode.ERROR, "Semantic loop detected")
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "message": "Potential Semantic Loop Detected",
                                "type": "oatp_loop_detected",
                                "code": "semantic_loop",
                                "similarity": result.max_similarity,
                                "oat_trace_id": oat_trace_id,
                            }
                        },
                        headers={"x-oat-trace-id": oat_trace_id},
                    )

        # ── 3. Forward to upstream LLM ─────────────────────────────────────
        is_stream = body.get("stream", False)
        upstream_headers = _forward_headers(request)

        with tracer.start_as_current_span(
            "upstream_llm_call", kind=SpanKind.CLIENT
        ) as client_span:
            upstream_resp = await _http_client.post(
                "/v1/chat/completions",
                content=json.dumps(body),
                headers=upstream_headers,
            )

            if upstream_resp.status_code != 200:
                client_span.set_status(
                    StatusCode.ERROR, f"Upstream {upstream_resp.status_code}"
                )
                return JSONResponse(
                    status_code=upstream_resp.status_code,
                    content=upstream_resp.json(),
                    headers={"x-oat-trace-id": oat_trace_id},
                )

            if is_stream:
                return _stream_response(upstream_resp, client_span, oat_trace_id)

            resp_body = upstream_resp.json()
            for k, v in response_attributes(resp_body).items():
                client_span.set_attribute(k, v)
            client_span.set_status(StatusCode.OK)

            return JSONResponse(
                content=resp_body,
                headers={"x-oat-trace-id": oat_trace_id},
            )


def _stream_response(
    upstream_resp: httpx.Response, client_span, oat_trace_id: str
) -> StreamingResponse:
    """Wrap upstream SSE chunks, forwarding them to the client transparently."""

    async def _sse_generator() -> AsyncIterator[bytes]:
        async for line in upstream_resp.aiter_lines():
            if not line:
                continue
            yield (line + "\n\n").encode()
            # Try to parse for span enrichment on the final chunk.
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    for choice in chunk.get("choices", []):
                        if fr := choice.get("finish_reason"):
                            client_span.set_attribute(
                                "gen_ai.response.finish_reasons", [fr]
                            )
                    if model := chunk.get("model"):
                        client_span.set_attribute("gen_ai.response.model", model)
                    if usage := chunk.get("usage"):
                        if pt := usage.get("prompt_tokens"):
                            client_span.set_attribute("gen_ai.usage.input_tokens", pt)
                        if ct := usage.get("completion_tokens"):
                            client_span.set_attribute("gen_ai.usage.output_tokens", ct)
                except json.JSONDecodeError:
                    pass
        client_span.set_status(StatusCode.OK)

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"x-oat-trace-id": oat_trace_id},
    )


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "oatp-proxy"}

"""
Integration tests for OAT Proxy.

Requirements:
    docker compose up -d   (Jaeger on :4317/:16686, Qdrant on :6333)

Run:
    pytest tests/test_integration.py -v -m integration

These tests exercise the FULL stack:
  • Real Qdrant for semantic loop detection (embeddings + vector search)
  • Real Jaeger for trace export + verification via its HTTP API
  • Mock upstream LLM (in-process ASGI transport — no OpenAI key needed)
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

# ── Markers ────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.integration


# ── Mock upstream LLM ──────────────────────────────────────────────────────

mock_llm = FastAPI()

MOCK_RESPONSE = {
    "id": "chatcmpl-integration",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Mock LLM response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
}


@mock_llm.post("/v1/chat/completions")
async def mock_chat(request: Request):
    body = await request.json()
    if body.get("stream"):
        return _mock_stream(body)
    return JSONResponse(content=MOCK_RESPONSE)


def _mock_stream(body: dict) -> StreamingResponse:
    """Return a minimal SSE stream mimicking OpenAI."""

    async def generate():
        chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "streamed"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Fixtures ───────────────────────────────────────────────────────────────

def _is_qdrant_up() -> bool:
    try:
        r = httpx.get("http://localhost:6333/healthz", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _is_jaeger_up() -> bool:
    try:
        r = httpx.get("http://localhost:16686/api/services", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


requires_infra = pytest.mark.skipif(
    not (_is_qdrant_up() and _is_jaeger_up()),
    reason="Requires docker compose services (Qdrant + Jaeger)",
)


@pytest.fixture(scope="module")
def proxy_client():
    """
    Boot the OAT proxy with:
      - real Qdrant (localhost:6333)
      - mock upstream LLM (in-process ASGI transport)
      - real OTel export to Jaeger (localhost:4317)
    """
    import main

    with TestClient(main.app, raise_server_exceptions=False) as tc:
        # Replace the upstream HTTP client with one routed to our mock LLM
        main._http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_llm),
            base_url="http://mock-llm",
        )
        yield tc


def _query_jaeger_traces(oat_trace_id: str, max_retries: int = 5) -> list[dict]:
    """Poll Jaeger API until traces for our service appear (batch export is async)."""
    from logic.stitcher import _uuid_to_otel_trace_id

    # Convert our OAT UUID → the hex trace ID Jaeger uses
    otel_tid = _uuid_to_otel_trace_id(oat_trace_id)
    hex_tid = format(otel_tid, "032x")

    for _ in range(max_retries):
        time.sleep(0.5)
        try:
            r = httpx.get(f"http://localhost:16686/api/traces/{hex_tid}", timeout=3)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return data
        except Exception:
            pass
    return []


# ── Test: Agent handoff ────────────────────────────────────────────────────

@requires_infra
class TestAgentHandoffIntegration:
    """Full Agent A → Agent B flow with real infrastructure."""

    def test_agent_a_generates_trace_id(self, proxy_client):
        resp = proxy_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello from Agent A"}]},
        )
        assert resp.status_code == 200
        trace_id = resp.headers.get("x-oat-trace-id")
        assert trace_id is not None
        uuid.UUID(trace_id)  # valid UUID
        assert resp.json()["choices"][0]["message"]["content"] == "Mock LLM response"

    def test_agent_b_stitches_context(self, proxy_client):
        """Agent A gets a trace ID, Agent B reuses it → same Jaeger trace."""
        # Agent A
        resp_a = proxy_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Agent A: plan the task"}]},
        )
        trace_id = resp_a.headers["x-oat-trace-id"]

        # Agent B reuses the trace ID
        resp_b = proxy_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Agent B: execute the plan"}]},
            headers={"x-oat-trace-id": trace_id},
        )
        assert resp_b.status_code == 200
        assert resp_b.headers["x-oat-trace-id"] == trace_id

    def test_traces_land_in_jaeger(self, proxy_client):
        """Verify spans are exported to Jaeger with correct hierarchy."""
        trace_id = str(uuid.uuid4())

        proxy_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Jaeger verification test"}]},
            headers={"x-oat-trace-id": trace_id},
        )

        # Force flush so spans are exported before we query.
        # BatchSpanProcessor needs time to serialize + send over gRPC.
        from logic.stitcher import _provider
        if _provider:
            _provider.force_flush(timeout_millis=5000)
        time.sleep(2)  # Give Jaeger time to index

        traces = _query_jaeger_traces(trace_id, max_retries=8)
        assert len(traces) >= 1, f"Expected trace in Jaeger for {trace_id}"

        spans = traces[0].get("spans", [])
        op_names = {s["operationName"] for s in spans}
        assert "chat gpt-4o" in op_names, f"Missing server span. Got: {op_names}"
        assert "upstream_llm_call" in op_names, f"Missing client span. Got: {op_names}"
        assert "semantic_guardrail" in op_names, f"Missing guardrail span. Got: {op_names}"

    def test_streaming_response(self, proxy_client):
        """SSE streaming works end-to-end through the proxy."""
        resp = proxy_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [{"role": "user", "content": "Stream test"}],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "data: " in body
        assert "[DONE]" in body
        assert "streamed" in body


# ── Test: Semantic loop detection (real Qdrant) ────────────────────────────

@requires_infra
class TestSemanticLoopIntegration:
    """Loop detection with real FastEmbed + Qdrant."""

    def test_identical_prompts_trigger_429(self, proxy_client):
        """Three identical prompts under the same trace → 429 on the third."""
        trace_id = str(uuid.uuid4())
        prompt = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Explain quantum entanglement in simple terms"}]}

        # Call 1 — no history, should pass
        r1 = proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})
        assert r1.status_code == 200, f"Call 1 failed: {r1.text}"

        # Call 2 — one prior, similarity = 1.0 but only 1 match ≥ threshold triggers
        r2 = proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})
        # Depending on threshold, call 2 may already trigger. With similarity=1.0 and threshold=0.96, it will.
        if r2.status_code == 200:
            # Call 3 — definitely should trigger
            r3 = proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})
            assert r3.status_code == 429, f"Expected 429 loop detection, got {r3.status_code}"
            body = r3.json()
        else:
            assert r2.status_code == 429
            body = r2.json()

        assert body["error"]["code"] == "semantic_loop"
        assert body["error"]["similarity"] >= 0.96

    def test_different_prompts_pass_through(self, proxy_client):
        """Semantically different prompts under the same trace should all pass."""
        trace_id = str(uuid.uuid4())
        prompts = [
            "What is the weather like on Mars?",
            "How do you bake sourdough bread?",
            "Explain the history of the Roman Empire",
        ]
        for text in prompts:
            resp = proxy_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": text}]},
                headers={"x-oat-trace-id": trace_id},
            )
            assert resp.status_code == 200, f"Unexpected block for: {text}"

    def test_loop_response_includes_trace_id(self, proxy_client):
        """429 loop response still carries x-oat-trace-id for debugging."""
        trace_id = str(uuid.uuid4())
        prompt = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Repeat this exact prompt for loop test"}]}

        proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})
        r2 = proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})

        # Whether it's call 2 or 3 that triggers, the response must have the trace ID
        if r2.status_code == 200:
            r2 = proxy_client.post("/v1/chat/completions", json=prompt, headers={"x-oat-trace-id": trace_id})

        assert r2.status_code == 429
        assert r2.headers["x-oat-trace-id"] == trace_id
        assert r2.json()["error"]["oat_trace_id"] == trace_id

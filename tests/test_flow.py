"""
Test suite for OAT Proxy — simulates multi-agent handoff.

Scenario:
  1. Agent A calls the proxy WITHOUT a trace ID → proxy generates one.
  2. Agent A extracts the `x-oat-trace-id` from the response.
  3. Agent B calls the proxy WITH that same trace ID → context is stitched.
  4. Repeated identical prompts under the same trace → 429 loop detection.

We mock the upstream LLM and Qdrant so tests run without external services.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from logic.stitcher import (
    OATP_TRACE_ID_ATTR,
    _uuid_to_otel_trace_id,
    resolve_trace_id,
    request_attributes,
    response_attributes,
)


# ── Stitcher unit tests ───────────────────────────────────────────────────

class TestResolveTraceId:
    def test_generates_uuid_when_missing(self):
        tid = resolve_trace_id(None)
        uuid.UUID(tid)  # validates format

    def test_generates_uuid_when_invalid(self):
        tid = resolve_trace_id("not-a-uuid")
        uuid.UUID(tid)
        assert tid != "not-a-uuid"

    def test_passes_through_valid_uuid(self):
        original = str(uuid.uuid4())
        assert resolve_trace_id(original) == original

    def test_deterministic_otel_trace_id(self):
        tid = str(uuid.uuid4())
        assert _uuid_to_otel_trace_id(tid) == _uuid_to_otel_trace_id(tid)

    def test_different_uuids_give_different_trace_ids(self):
        a = _uuid_to_otel_trace_id(str(uuid.uuid4()))
        b = _uuid_to_otel_trace_id(str(uuid.uuid4()))
        assert a != b


class TestAttributeMapping:
    def test_request_attributes_basic(self):
        body = {"model": "gpt-4o", "max_tokens": 100, "temperature": 0.7}
        attrs = request_attributes(body)
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.request.max_tokens"] == 100
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.operation.name"] == "chat"

    def test_request_attributes_with_tools(self):
        body = {"model": "gpt-4o", "tools": [{"type": "function"}]}
        attrs = request_attributes(body)
        assert attrs["oatp.has_tools"] is True

    def test_response_attributes(self):
        body = {
            "model": "gpt-4o-2024-05-13",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop"}],
        }
        attrs = response_attributes(body)
        assert attrs["gen_ai.response.model"] == "gpt-4o-2024-05-13"
        assert attrs["gen_ai.usage.input_tokens"] == 50
        assert attrs["gen_ai.usage.output_tokens"] == 20
        assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


# ── Integration tests (mocked upstream) ───────────────────────────────────

MOCK_UPSTREAM_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from upstream!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _make_mock_response(status_code: int = 200, body: dict | None = None) -> httpx.Response:
    """Build a fake httpx.Response."""
    content = json.dumps(body or MOCK_UPSTREAM_RESPONSE).encode()
    return httpx.Response(status_code=status_code, content=content)




@pytest.fixture()
def client():
    """Create a TestClient with mocked upstream + disabled guardrail."""
    import main

    with TestClient(main.app, raise_server_exceptions=False) as tc:
        # Set mocks AFTER lifespan has run (TestClient triggers lifespan)
        main._guardrail = None

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post = AsyncMock(return_value=_make_mock_response())
        mock_http.aclose = AsyncMock()
        main._http_client = mock_http

        yield tc


class TestAgentHandoff:
    """Simulate the Agent A → Proxy → Agent B flow."""

    def test_agent_a_gets_trace_id(self, client: TestClient):
        """Agent A calls without a trace ID; proxy generates one."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        trace_id = resp.headers.get("x-oat-trace-id")
        assert trace_id is not None
        uuid.UUID(trace_id)  # must be valid UUID

    def test_agent_b_reuses_trace_id(self, client: TestClient):
        """Agent B passes the trace ID from Agent A; proxy stitches context."""
        shared_trace_id = str(uuid.uuid4())
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Continue"}]},
            headers={"x-oat-trace-id": shared_trace_id},
        )
        assert resp.status_code == 200
        assert resp.headers["x-oat-trace-id"] == shared_trace_id

    def test_response_body_forwarded(self, client: TestClient):
        """Upstream response body is forwarded correctly."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        body = resp.json()
        assert body["model"] == "gpt-4o"
        assert body["choices"][0]["message"]["content"] == "Hello from upstream!"

    def test_upstream_error_forwarded(self, client: TestClient):
        """Upstream 4xx/5xx is forwarded to the caller."""
        import main

        error_body = {"error": {"message": "Rate limited", "type": "rate_limit"}}
        main._http_client.post = AsyncMock(
            return_value=_make_mock_response(429, error_body)
        )
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "rate_limit"


class TestLoopDetection:
    """Test the semantic guardrail (with a mock guardrail)."""

    def test_loop_returns_429(self, client: TestClient):
        """When guardrail detects a loop, proxy returns 429."""
        import main
        from logic.guardrails import GuardrailResult

        mock_guard = AsyncMock()
        mock_guard.check = AsyncMock(
            return_value=GuardrailResult(is_loop=True, max_similarity=0.98, checked_count=3)
        )
        main._guardrail = mock_guard

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["code"] == "semantic_loop"
        assert body["error"]["similarity"] == 0.98

    def test_no_loop_passes_through(self, client: TestClient):
        """When guardrail says OK, request proceeds to upstream."""
        import main
        from logic.guardrails import GuardrailResult

        mock_guard = AsyncMock()
        mock_guard.check = AsyncMock(
            return_value=GuardrailResult(is_loop=False, max_similarity=0.5, checked_count=2)
        )
        main._guardrail = mock_guard

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "oatp-proxy"

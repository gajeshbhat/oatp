"""
Cross-framework integration tests for the OAT Proxy.

Proves the paper's core thesis: a crewAI agent and an AutoGen agent can
share a single trace via the OAT Proxy, with spans visible in Jaeger
under one unified trace.

Requirements:
    docker compose up -d                    # Jaeger + Qdrant
    uvicorn tests.mock_llm_server:app --port 9120   # Mock LLM
    UPSTREAM_BASE_URL=http://localhost:9120 uvicorn main:app --port 9119  # Proxy

Run:
    pytest tests/test_cross_framework.py -v -m integration
"""

from __future__ import annotations

import asyncio
import time
import uuid
import warnings

import httpx
import pytest

pytestmark = pytest.mark.integration

PROXY_URL = "http://localhost:9119"
MOCK_LLM_URL = "http://localhost:9120"


# ── Precondition checks ───────────────────────────────────────────────────

def _service_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except Exception:
        return False


requires_servers = pytest.mark.skipif(
    not (_service_up(PROXY_URL) and _service_up(MOCK_LLM_URL)),
    reason="Requires OAT proxy (:9119) and mock LLM (:9120) running",
)

requires_jaeger = pytest.mark.skipif(
    not _service_up("http://localhost:16686"),
    reason="Requires Jaeger (:16686) running",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _query_jaeger_trace(oat_trace_id: str, retries: int = 8) -> list[dict]:
    from logic.stitcher import _uuid_to_otel_trace_id

    hex_tid = format(_uuid_to_otel_trace_id(oat_trace_id), "032x")
    for _ in range(retries):
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


def _get_span_attr(span: dict, key: str):
    """Extract an attribute value from a Jaeger span."""
    for tag in span.get("tags", []):
        if tag["key"] == key:
            return tag["value"]
    return None



# ── Test: crewAI through the proxy ────────────────────────────────────────

@requires_servers
class TestCrewAI:
    """Verify a real crewAI LLM call flows through the OAT proxy."""

    def test_crewai_call_with_trace_id(self):
        from crewai import LLM

        trace_id = str(uuid.uuid4())
        llm = LLM(
            model="openai/gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "crewai",
                "x-oat-agent-name": "researcher",
            },
        )
        result = llm.call("Summarise the latest AI safety research")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Mock response" in result

    @requires_jaeger
    def test_crewai_spans_in_jaeger(self):
        from crewai import LLM

        trace_id = str(uuid.uuid4())
        llm = LLM(
            model="openai/gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "crewai",
                "x-oat-agent-name": "researcher",
                "x-oat-task-id": "task-001",
            },
        )
        llm.call("What are the latest trends in LLM evaluation?")

        # The proxy's BatchSpanProcessor flushes asynchronously.
        time.sleep(2)
        traces = _query_jaeger_trace(trace_id, retries=10)
        assert len(traces) >= 1, f"No trace found in Jaeger for {trace_id}"

        spans = traces[0]["spans"]
        server_span = next((s for s in spans if s["operationName"].startswith("chat ")), None)
        assert server_span is not None

        assert _get_span_attr(server_span, "oatp.framework") == "crewai"
        assert _get_span_attr(server_span, "oatp.agent_name") == "researcher"
        assert _get_span_attr(server_span, "oatp.task_id") == "task-001"


# ── Test: AutoGen through the proxy ───────────────────────────────────────

@requires_servers
class TestAutoGen:
    """Verify a real AutoGen agent call flows through the OAT proxy."""

    def test_autogen_call_with_trace_id(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import UserMessage

        trace_id = str(uuid.uuid4())
        client = OpenAIChatCompletionClient(
            model="gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "autogen",
                "x-oat-agent-name": "coder",
            },
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-4o",
                "structured_output": True,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = asyncio.run(
                client.create([UserMessage(content="Write a Python hello world", source="user")])
            )
        assert "Mock response" in result.content
        assert result.finish_reason == "stop"

    @requires_jaeger
    def test_autogen_spans_in_jaeger(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import UserMessage

        trace_id = str(uuid.uuid4())
        client = OpenAIChatCompletionClient(
            model="gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "autogen",
                "x-oat-agent-name": "coder",
                "x-oat-task-id": "conv-002",
            },
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-4o",
                "structured_output": True,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asyncio.run(client.create([UserMessage(content="Debug this code", source="user")]))

        # The proxy's BatchSpanProcessor flushes asynchronously.
        # Give it time before querying Jaeger.
        time.sleep(2)
        traces = _query_jaeger_trace(trace_id, retries=10)
        assert len(traces) >= 1, f"No trace in Jaeger for {trace_id}"

        spans = traces[0]["spans"]
        server_span = next((s for s in spans if s["operationName"].startswith("chat ")), None)
        assert server_span is not None

        assert _get_span_attr(server_span, "oatp.framework") == "autogen"
        assert _get_span_attr(server_span, "oatp.agent_name") == "coder"
        assert _get_span_attr(server_span, "oatp.task_id") == "conv-002"


# ── Test: LangSmith-wrapped OpenAI client through the proxy ───────────────

@requires_servers
class TestLangSmith:
    """Verify a LangSmith-wrapped OpenAI client flows through the OAT proxy."""

    def test_langsmith_call_with_trace_id(self):
        from openai import OpenAI
        from langsmith.wrappers import wrap_openai

        trace_id = str(uuid.uuid4())
        raw_client = OpenAI(
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "langsmith",
                "x-oat-agent-name": "evaluator",
            },
        )
        client = wrap_openai(raw_client)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Evaluate this agent output"}],
        )
        assert resp.choices[0].message.content is not None
        assert "Mock response" in resp.choices[0].message.content

    @requires_jaeger
    def test_langsmith_spans_in_jaeger(self):
        from openai import OpenAI
        from langsmith.wrappers import wrap_openai

        trace_id = str(uuid.uuid4())
        raw_client = OpenAI(
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": trace_id,
                "x-oat-framework": "langsmith",
                "x-oat-agent-name": "evaluator",
                "x-oat-task-id": "eval-001",
            },
        )
        client = wrap_openai(raw_client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Score this response"}],
        )

        time.sleep(2)
        traces = _query_jaeger_trace(trace_id, retries=10)
        assert len(traces) >= 1, f"No trace in Jaeger for {trace_id}"

        spans = traces[0]["spans"]
        server_span = next((s for s in spans if s["operationName"].startswith("chat ")), None)
        assert server_span is not None

        assert _get_span_attr(server_span, "oatp.framework") == "langsmith"
        assert _get_span_attr(server_span, "oatp.agent_name") == "evaluator"
        assert _get_span_attr(server_span, "oatp.task_id") == "eval-001"


# ── Test: Cross-framework handoff ─────────────────────────────────────────

@requires_servers
@requires_jaeger
class TestCrossFrameworkHandoff:
    """
    The paper's core proof: crewAI agent delegates to AutoGen agent,
    both sharing a single trace ID, producing one unified Jaeger trace
    with framework-specific metadata on each span.
    """

    def test_crewai_to_autogen_handoff(self):
        from crewai import LLM
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_core.models import UserMessage

        # ── Step 1: crewAI "Manager" starts the conversation ──────────
        shared_trace_id = str(uuid.uuid4())

        crewai_llm = LLM(
            model="openai/gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": shared_trace_id,
                "x-oat-framework": "crewai",
                "x-oat-agent-name": "manager",
                "x-oat-task-id": "planning",
                "x-oat-delegation-depth": "0",
            },
        )
        plan = crewai_llm.call("Create a plan to build a REST API")
        assert "Mock response" in plan

        # ── Step 2: AutoGen "Coder" continues with the SAME trace ─────
        autogen_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            base_url=f"{PROXY_URL}/v1",
            api_key="fake-key",
            default_headers={
                "x-oat-trace-id": shared_trace_id,
                "x-oat-framework": "autogen",
                "x-oat-agent-name": "coder",
                "x-oat-task-id": "implementation",
                "x-oat-delegation-depth": "1",
            },
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-4o",
                "structured_output": True,
            },
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = asyncio.run(
                autogen_client.create([
                    UserMessage(content=f"Implement: {plan}", source="user"),
                ])
            )
        assert "Mock response" in result.content

        # ── Step 3: Verify UNIFIED trace in Jaeger ────────────────────
        traces = _query_jaeger_trace(shared_trace_id, retries=10)
        assert len(traces) >= 1, f"No trace in Jaeger for {shared_trace_id}"

        spans = traces[0]["spans"]
        # Should have at least 2 server spans (one per framework call)
        server_spans = [s for s in spans if s["operationName"].startswith("chat ")]
        assert len(server_spans) >= 2, (
            f"Expected ≥2 server spans (crewAI + AutoGen), got {len(server_spans)}"
        )

        # Verify framework-specific metadata
        frameworks_seen = {_get_span_attr(s, "oatp.framework") for s in server_spans}
        assert "crewai" in frameworks_seen, f"Missing crewai span. Saw: {frameworks_seen}"
        assert "autogen" in frameworks_seen, f"Missing autogen span. Saw: {frameworks_seen}"

        # Verify delegation depth
        depths = {_get_span_attr(s, "oatp.delegation_depth") for s in server_spans}
        assert 0 in depths or "0" in depths, f"Missing depth=0 span. Saw: {depths}"

        # Verify all spans share the same Jaeger trace ID
        trace_ids = {s["traceID"] for s in spans}
        assert len(trace_ids) == 1, (
            f"Trace fragmentation detected! Multiple trace IDs: {trace_ids}"
        )

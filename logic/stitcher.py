"""
Context Stitcher — OTel span management for the OAT Proxy.

Responsibilities:
  • Extract or generate a Global Trace ID from `x-oat-trace-id`.
  • Create deterministic 128-bit OTel trace IDs from that UUID so every span
    belonging to the same logical conversation lands in one Jaeger trace.
  • Produce spans that follow the **OTel GenAI Semantic Conventions v1.40+**.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    TraceFlags,
    set_span_in_context,
)

# ── GenAI Semantic Convention attribute keys (v1.40+) ──────────────────────
# These are stable as of OTel semconv 1.40.  We define them as constants to
# avoid hard-coding magic strings everywhere.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

OATP_TRACE_ID_ATTR = "oatp.trace_id"
OATP_HAS_TOOLS = "oatp.has_tools"
OATP_FRAMEWORK = "oatp.framework"
OATP_TASK_ID = "oatp.task_id"
OATP_AGENT_NAME = "oatp.agent_name"
OATP_DELEGATION_DEPTH = "oatp.delegation_depth"

# ── Header constants ──────────────────────────────────────────────────────
HEADER_TRACE_ID = "x-oat-trace-id"
HEADER_FRAMEWORK = "x-oat-framework"
HEADER_TASK_ID = "x-oat-task-id"
HEADER_AGENT_NAME = "x-oat-agent-name"
HEADER_DELEGATION_DEPTH = "x-oat-delegation-depth"

_SERVICE_NAME = "oatp-proxy"

# ── Singleton tracer setup ─────────────────────────────────────────────────
_provider: TracerProvider | None = None


def _get_provider() -> TracerProvider:
    global _provider
    if _provider is None:
        resource = Resource.create({"service.name": _SERVICE_NAME})
        _provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(insecure=True)  # localhost Jaeger
        _provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)
    return _provider


def get_tracer() -> trace.Tracer:
    return _get_provider().get_tracer("oatp.stitcher", "0.1.0")


def shutdown_tracer() -> None:
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


# ── Trace-ID helpers ───────────────────────────────────────────────────────

def resolve_trace_id(header_value: str | None) -> str:
    """Return the canonical OAT trace-id (a UUID string)."""
    if header_value:
        try:
            return str(uuid.UUID(header_value))
        except ValueError:
            pass
    return str(uuid.uuid4())


def _uuid_to_otel_trace_id(oat_id: str) -> int:
    """Deterministically map a UUID string → 128-bit int for OTel."""
    digest = hashlib.md5(oat_id.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest, "big")


def _uuid_to_otel_span_id(oat_id: str) -> int:
    """Deterministically derive a 64-bit span ID from the OAT UUID."""
    digest = hashlib.sha256(oat_id.encode(), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], "big")


def build_parent_context(oat_trace_id: str) -> Context:
    """
    Forge a *remote* SpanContext whose trace-id is derived from the OAT UUID.
    This lets every request that shares the same `x-oat-trace-id` appear in
    the same Jaeger trace.

    We also derive a deterministic span_id because OTel requires a valid
    (non-zero) span_id to treat the SpanContext as valid.
    """
    otel_tid = _uuid_to_otel_trace_id(oat_trace_id)
    otel_sid = _uuid_to_otel_span_id(oat_trace_id)
    span_ctx = SpanContext(
        trace_id=otel_tid,
        span_id=otel_sid,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(span_ctx))


# ── Attribute mapping ─────────────────────────────────────────────────────

def request_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract GenAI semantic-convention attributes from an OpenAI-style request."""
    attrs: dict[str, Any] = {
        GEN_AI_SYSTEM: "openai",
        GEN_AI_OPERATION_NAME: "chat",
    }
    if model := body.get("model"):
        attrs[GEN_AI_REQUEST_MODEL] = model
    if (mt := body.get("max_tokens")) is not None:
        attrs[GEN_AI_REQUEST_MAX_TOKENS] = mt
    if (temp := body.get("temperature")) is not None:
        attrs[GEN_AI_REQUEST_TEMPERATURE] = float(temp)
    if (tp := body.get("top_p")) is not None:
        attrs[GEN_AI_REQUEST_TOP_P] = float(tp)
    if body.get("tools"):
        attrs[OATP_HAS_TOOLS] = True
    return attrs


def framework_attributes(headers: dict[str, str]) -> dict[str, Any]:
    """Extract OAT framework metadata from request headers.

    Normalises framework-specific concepts (crewAI "Tasks", AutoGen
    "Conversations") into a common attribute namespace so that spans
    from different frameworks can be compared side-by-side in Jaeger.
    """
    attrs: dict[str, Any] = {}
    if fw := headers.get(HEADER_FRAMEWORK):
        attrs[OATP_FRAMEWORK] = fw.lower()
    if tid := headers.get(HEADER_TASK_ID):
        attrs[OATP_TASK_ID] = tid
    if name := headers.get(HEADER_AGENT_NAME):
        attrs[OATP_AGENT_NAME] = name
    if depth := headers.get(HEADER_DELEGATION_DEPTH):
        try:
            attrs[OATP_DELEGATION_DEPTH] = int(depth)
        except ValueError:
            pass
    return attrs


def response_attributes(body: dict[str, Any]) -> dict[str, Any]:
    """Extract GenAI semantic-convention attributes from an OpenAI-style response."""
    attrs: dict[str, Any] = {}
    if model := body.get("model"):
        attrs[GEN_AI_RESPONSE_MODEL] = model
    if usage := body.get("usage"):
        if (pt := usage.get("prompt_tokens")) is not None:
            attrs[GEN_AI_USAGE_INPUT_TOKENS] = pt
        if (ct := usage.get("completion_tokens")) is not None:
            attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = ct
    if choices := body.get("choices"):
        reasons = [c.get("finish_reason", "") for c in choices if c.get("finish_reason")]
        if reasons:
            attrs[GEN_AI_RESPONSE_FINISH_REASONS] = reasons
    return attrs

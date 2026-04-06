# OAT Proxy — Open Agent Telemetry

A framework-agnostic middleware for cross-runtime trace stitching and semantic governance in multi-agent systems.

## What It Does

OAT Proxy sits between your AI agents and the LLM provider. Agents point their `base_url` at the proxy — everything else is transparent.

```
┌─────────┐     ┌─────────────────────────────┐     ┌──────────┐
│ crewAI  │────▶│        OAT Proxy :9119      │────▶│ OpenAI   │
│ AutoGen │────▶│  trace stitching + guardrail │────▶│ or any   │
│ LangSmith│───▶│  OTel export to Jaeger       │────▶│ LLM API  │
└─────────┘     └─────────────────────────────┘     └──────────┘
```

**Three capabilities:**

1. **Global Trace Stitching** — Agents share a `x-oat-trace-id` header. The proxy maps it to a deterministic OpenTelemetry trace ID so all spans from a multi-agent conversation appear in one Jaeger trace.

2. **Semantic Circuit Breaker** — Each prompt is embedded locally (FastEmbed) and compared against recent prompts in the same trace (Qdrant). Cosine similarity ≥ 0.96 → blocked with `429` before hitting the LLM.

3. **Schema Normalization** — Framework metadata (`x-oat-framework`, `x-oat-agent-name`, `x-oat-task-id`) is mapped to OTel GenAI Semantic Convention attributes for unified observability.

## Quick Start

```bash
# Install
uv sync --extra dev

# Start infrastructure
docker compose up -d    # Jaeger :16686, Qdrant :6333

# Run the proxy
UPSTREAM_BASE_URL=https://api.openai.com uv run uvicorn main:app --port 9119

# Point any agent at it
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9119/v1", api_key="sk-...")
```

## Testing

```bash
# Unit tests (no infrastructure needed)
uv run pytest tests/test_flow.py -v

# Full suite (requires docker compose up -d + mock server + proxy)
uv run uvicorn tests.mock_llm_server:app --port 9120 &
UPSTREAM_BASE_URL=http://localhost:9120 uv run uvicorn main:app --port 9119 &
uv run pytest tests/ -v
```

**29 tests** covering:
- Trace ID generation and propagation
- crewAI 1.13 → AutoGen 0.7.5 cross-framework handoff (verified in Jaeger)
- LangSmith-wrapped OpenAI client integration
- Semantic loop detection with real Qdrant
- SSE streaming passthrough

## Benchmarks

```bash
python -m benchmarks.latency
```

| Measurement | p50 | p95 |
|---|---|---|
| Direct (no proxy) | 1.19 ms | 2.07 ms |
| Proxy pass-through | 36.54 ms | 52.32 ms |
| Proxy + guardrail | 35.18 ms | 47.50 ms |
| Embedding only | 12.09 ms | 17.74 ms |
| Qdrant query only | 3.11 ms | 5.53 ms |

## Project Structure

```
main.py                     FastAPI proxy (/v1/chat/completions)
logic/
  stitcher.py               OTel trace ID mapping + GenAI attribute extraction
  guardrails.py             FastEmbed + Qdrant semantic loop detection
tests/
  test_flow.py              15 unit tests (fully mocked)
  test_integration.py        7 integration tests (real Qdrant + Jaeger)
  test_cross_framework.py    7 cross-framework tests (crewAI + AutoGen + LangSmith)
  mock_llm_server.py        Standalone mock OpenAI-compatible server
benchmarks/
  latency.py                p50/p95/p99 + concurrent load benchmarks
docker-compose.yaml         Jaeger 2.17.0 + Qdrant v1.17.1
```

## Stack

| Component | Version |
|---|---|
| Python | 3.12+ |
| FastAPI | 0.135 |
| OpenTelemetry | 1.34 |
| crewAI | 1.13.0 |
| AutoGen | 0.7.5 |
| LangSmith | 0.7.25 |
| FastEmbed | 0.8.0 (BAAI/bge-small-en-v1.5) |
| Qdrant | v1.17.1 |
| Jaeger | 2.17.0 |

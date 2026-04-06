"""
OAT Proxy Latency Benchmarks.

Measures the overhead introduced by the proxy for Research Question 2:
"What is the performance impact of semantic analysis (vector similarity)
on the overall agent response time?"

Produces p50 / p95 / p99 latency numbers for:
  1. Baseline: Direct call to mock LLM (no proxy)
  2. Proxy pass-through: Through proxy with guardrail disabled
  3. Proxy + guardrail: Through proxy with semantic analysis active
  4. Embedding only: FastEmbed embedding generation in isolation
  5. Qdrant query only: Vector search in isolation
  6. Concurrent load: N parallel agents through the proxy

Requirements:
    docker compose up -d
    uvicorn tests.mock_llm_server:app --port 9120
    UPSTREAM_BASE_URL=http://localhost:9120 uvicorn main:app --port 9119

Run:
    python -m benchmarks.latency
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from typing import NamedTuple

import httpx

MOCK_LLM_URL = "http://localhost:9120"
PROXY_URL = "http://localhost:9119"
WARMUP_ROUNDS = 3
BENCH_ROUNDS = 30

SAMPLE_PAYLOAD = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}],
}


class Stats(NamedTuple):
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


def _compute_stats(latencies_ms: list[float]) -> Stats:
    s = sorted(latencies_ms)
    n = len(s)
    return Stats(
        p50=s[n // 2],
        p95=s[int(n * 0.95)],
        p99=s[int(n * 0.99)],
        mean=statistics.mean(s),
        min=s[0],
        max=s[-1],
    )


def _print_stats(label: str, stats: Stats) -> None:
    print(f"  {label:30s}  p50={stats.p50:7.2f}ms  p95={stats.p95:7.2f}ms  "
          f"p99={stats.p99:7.2f}ms  mean={stats.mean:7.2f}ms  "
          f"[{stats.min:.2f} – {stats.max:.2f}]")


# ── Benchmark functions ────────────────────────────────────────────────────

def bench_direct(client: httpx.Client) -> list[float]:
    """Baseline: direct call to mock LLM, no proxy."""
    latencies = []
    for i in range(WARMUP_ROUNDS + BENCH_ROUNDS):
        start = time.perf_counter()
        r = client.post(f"{MOCK_LLM_URL}/v1/chat/completions", json=SAMPLE_PAYLOAD)
        elapsed = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        if i >= WARMUP_ROUNDS:
            latencies.append(elapsed)
    return latencies


def bench_proxy_no_guardrail(client: httpx.Client) -> list[float]:
    """Proxy pass-through — each request gets a unique trace to avoid loop detection."""
    latencies = []
    for i in range(WARMUP_ROUNDS + BENCH_ROUNDS):
        trace_id = str(uuid.uuid4())
        start = time.perf_counter()
        r = client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=SAMPLE_PAYLOAD,
            headers={"x-oat-trace-id": trace_id},
        )
        elapsed = (time.perf_counter() - start) * 1000
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
        if i >= WARMUP_ROUNDS:
            latencies.append(elapsed)
    return latencies


_DIVERSE_TOPICS = [
    "quantum entanglement", "sourdough baking", "Roman Empire history",
    "machine learning optimizers", "coral reef ecosystems", "jazz improvisation",
    "spacecraft propulsion", "medieval architecture", "protein folding",
    "graph database design", "Antarctic exploration", "Renaissance painting",
    "compiler optimization", "deep sea creatures", "urban planning",
    "chess endgame strategy", "volcanic geology", "musical harmony theory",
    "distributed consensus", "ancient Egyptian writing", "microbiome research",
    "neural network pruning", "Fibonacci in nature", "coffee roasting science",
    "satellite navigation", "Celtic mythology", "polymer chemistry",
    "game theory basics", "tidal energy systems", "origami mathematics",
    "DNA sequencing methods", "cloud formation", "supply chain logistics",
]


def bench_proxy_with_guardrail(client: httpx.Client) -> list[float]:
    """Proxy + guardrail — unique trace per call, exercises the full embed+query path."""
    latencies = []
    for i in range(WARMUP_ROUNDS + BENCH_ROUNDS):
        trace_id = str(uuid.uuid4())
        topic = _DIVERSE_TOPICS[i % len(_DIVERSE_TOPICS)]
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"Explain {topic} in detail"}],
        }
        start = time.perf_counter()
        r = client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            headers={"x-oat-trace-id": trace_id},
        )
        elapsed = (time.perf_counter() - start) * 1000
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
        if i >= WARMUP_ROUNDS:
            latencies.append(elapsed)
    return latencies


def bench_embedding_only() -> list[float]:
    """FastEmbed embedding generation in isolation."""
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Warmup
    list(embedder.embed(["warmup"]))
    latencies = []
    for _ in range(BENCH_ROUNDS):
        text = f"Explain topic {uuid.uuid4().hex[:8]} in simple terms"
        start = time.perf_counter()
        list(embedder.embed([text]))
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    return latencies


def bench_qdrant_query() -> list[float]:
    """Qdrant vector search in isolation."""
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    qdrant = QdrantClient(url="http://localhost:6333", timeout=5)
    vec = list(embedder.embed(["test query"]))[0].tolist()

    latencies = []
    for _ in range(BENCH_ROUNDS):
        start = time.perf_counter()
        qdrant.query_points(
            collection_name="oatp_prompts",
            query=vec,
            query_filter=Filter(
                must=[FieldCondition(key="trace_id", match=MatchValue(value="bench"))]
            ),
            limit=3,
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
    return latencies


async def _concurrent_worker(
    client: httpx.AsyncClient, results: list[float], worker_id: int
) -> None:
    """Single concurrent worker making BENCH_ROUNDS // concurrency requests."""
    rounds = max(1, BENCH_ROUNDS // 10)
    for i in range(rounds):
        trace_id = str(uuid.uuid4())
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": f"Worker {worker_id} request {i}"}],
        }
        start = time.perf_counter()
        r = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            headers={"x-oat-trace-id": trace_id},
        )
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code == 200:
            results.append(elapsed)


async def bench_concurrent(concurrency: int) -> list[float]:
    """N parallel agents hitting the proxy simultaneously."""
    results: list[float] = []
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [
            _concurrent_worker(client, results, i) for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("OAT Proxy Latency Benchmarks")
    print(f"Warmup: {WARMUP_ROUNDS} rounds, Benchmark: {BENCH_ROUNDS} rounds")
    print("=" * 78)

    # Check services
    for name, url in [("Mock LLM", MOCK_LLM_URL), ("Proxy", PROXY_URL)]:
        try:
            r = httpx.get(f"{url}/health", timeout=2)
            assert r.status_code == 200
        except Exception:
            print(f"ERROR: {name} at {url} is not running. Aborting.")
            return

    client = httpx.Client(timeout=30)

    print("\n── Latency Breakdown ─────────────────────────────────────────")
    _print_stats("1. Direct (no proxy)", _compute_stats(bench_direct(client)))
    _print_stats("2. Proxy (no guardrail)", _compute_stats(bench_proxy_no_guardrail(client)))
    _print_stats("3. Proxy + guardrail", _compute_stats(bench_proxy_with_guardrail(client)))
    _print_stats("4. Embedding only", _compute_stats(bench_embedding_only()))

    try:
        _print_stats("5. Qdrant query only", _compute_stats(bench_qdrant_query()))
    except Exception as e:
        print(f"  5. Qdrant query only: SKIPPED ({e})")

    proxy_lat = _compute_stats(bench_proxy_no_guardrail(client))
    direct_lat = _compute_stats(bench_direct(client))
    overhead = proxy_lat.p50 - direct_lat.p50

    print("\n── Concurrent Load ───────────────────────────────────────────")
    for n in [10, 50, 100]:
        try:
            results = asyncio.run(bench_concurrent(n))
            if results:
                _print_stats(f"  {n} concurrent agents", _compute_stats(results))
            else:
                print(f"    {n} concurrent agents: no successful responses")
        except Exception as e:
            print(f"    {n} concurrent agents: FAILED ({e})")

    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Proxy overhead (p50):        {overhead:+.2f} ms")
    guardrail_lat = _compute_stats(bench_proxy_with_guardrail(client))
    guardrail_overhead = guardrail_lat.p50 - direct_lat.p50
    print(f"  Proxy + guardrail (p50):     {guardrail_overhead:+.2f} ms")
    print(f"  Guardrail-only cost (p50):   {guardrail_overhead - overhead:+.2f} ms")
    print("=" * 78)

    client.close()


if __name__ == "__main__":
    main()

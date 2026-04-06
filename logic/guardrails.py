"""
Semantic Guardrail — Loop Breaker for the OAT Proxy.

Uses FastEmbed for local embedding generation and Qdrant for vector storage.
For each incoming prompt within a trace, we:
  1. Generate an embedding via FastEmbed.
  2. Search Qdrant for the 3 most-recent prompts in the same trace.
  3. If cosine similarity with ANY of those > SIMILARITY_THRESHOLD → loop detected.
  4. Otherwise, store the new embedding and let the request through.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger("oatp.guardrails")

COLLECTION_NAME = "oatp_prompts"
SIMILARITY_THRESHOLD = 0.96
LOOKBACK_COUNT = 3
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@dataclass
class GuardrailResult:
    is_loop: bool
    max_similarity: float
    checked_count: int


class SemanticGuardrail:
    """Stateful loop-breaker backed by Qdrant + FastEmbed."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.threshold = similarity_threshold
        self._qdrant = QdrantClient(url=qdrant_url, timeout=5)
        self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        self._ensure_collection()

    # ── public API ─────────────────────────────────────────────────────────

    async def check(self, trace_id: str, prompt_text: str) -> GuardrailResult:
        """
        Check whether *prompt_text* is semantically too close to recent
        prompts in the same trace.  This method is sync-safe because both
        FastEmbed and Qdrant-client are blocking; we keep the async
        signature so the caller can await it uniformly.
        """
        embedding = self._embed(prompt_text)

        hits = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            query_filter=Filter(
                must=[FieldCondition(key="trace_id", match=MatchValue(value=trace_id))]
            ),
            limit=LOOKBACK_COUNT,
        ).points

        max_sim = max((h.score for h in hits), default=0.0)
        is_loop = max_sim >= self.threshold

        if is_loop:
            logger.warning(
                "Loop detected for trace %s (similarity=%.4f)", trace_id, max_sim
            )
        else:
            # Store the new prompt embedding for future comparisons.
            self._store(trace_id, prompt_text, embedding)

        return GuardrailResult(
            is_loop=is_loop,
            max_similarity=max_sim,
            checked_count=len(hits),
        )

    # ── internals ──────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        # FastEmbed returns a generator; materialise the first (only) vector.
        vectors = list(self._embedder.embed([text]))
        return vectors[0].tolist()

    def _store(
        self, trace_id: str, prompt_text: str, embedding: list[float]
    ) -> None:
        import uuid

        self._qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embedding,
                    payload={
                        "trace_id": trace_id,
                        "prompt_text": prompt_text[:500],  # truncate for storage
                        "ts": time.time(),
                    },
                )
            ],
        )

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection %r", COLLECTION_NAME)

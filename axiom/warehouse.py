"""
AXIOM Data Warehouse
====================
Three-layer storage architecture:

  Layer 1 — Qdrant (vector store)
    Stores paper embeddings (sentence-transformers all-MiniLM-L6-v2).
    Enables semantic similarity search.
    Persistent: data survives restarts.

  Layer 2 — Redis (query cache)
    Caches search results by query+limit+filters.
    TTL: 24 hours. Dramatically reduces re-computation.
    Optional: warehouse works without Redis (just slower).

  Layer 3 — JSONL backup (axiom_warehouse.jsonl)
    Full fidelity backup of every paper stored.
    Human-readable, portable, importable into other tools.

Prerequisites:
    # Qdrant (required for semantic search)
    docker run -d -p 6333:6333 -p 6334:6334 \\
        -v $(pwd)/qdrant_storage:/qdrant/storage \\
        qdrant/qdrant

    # Redis (optional, for caching)
    docker run -d -p 6379:6379 redis:alpine

    # Python deps
    pip install qdrant-client sentence-transformers redis

Usage:
    from axiom.warehouse import PaperWarehouse

    wh = PaperWarehouse()
    wh.connect()                    # connects Qdrant + Redis

    wh.store(papers)                # add papers to the warehouse
    results = wh.search("fraud detection transformers", limit=20)
    stats = wh.stats()

    wh.close()
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from pathlib import Path
from typing import Optional

from .models import Paper
from .normaliser import Normaliser

logger = logging.getLogger(__name__)

# ─── Configuration (override via env vars) ────────────────────────────────────
QDRANT_HOST         = os.getenv("QDRANT_HOST",         "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT",     "6333"))
QDRANT_COLLECTION   = os.getenv("QDRANT_COLLECTION",   "axiom_papers")
REDIS_HOST          = os.getenv("REDIS_HOST",          "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT",      "6379"))
REDIS_TTL           = int(os.getenv("REDIS_TTL",       "86400"))   # 24 hours
EMBED_MODEL         = os.getenv("AXIOM_EMBED_MODEL",   "all-MiniLM-L6-v2")
EMBED_DIM           = 384   # dimension for all-MiniLM-L6-v2
BACKUP_PATH         = os.getenv("AXIOM_WAREHOUSE_PATH", "axiom_warehouse.jsonl")


def _cache_key(query: str, limit: int, quality_only: bool) -> str:
    raw = f"{query.lower().strip()}:{limit}:{quality_only}"
    return "axiom:search:" + hashlib.md5(raw.encode()).hexdigest()


def _paper_to_text(paper: Paper) -> str:
    """Concatenate fields for embedding — title weighted by repetition."""
    parts = [
        paper.title, paper.title,   # title twice → higher semantic weight
        " ".join(paper.authors[:3]),
        paper.venue or "",
        paper.abstract or "",
    ]
    return " ".join(p for p in parts if p).strip()


class PaperWarehouse:
    """
    Semantic vector warehouse for academic papers.

    Backed by Qdrant (vectors) + Redis (cache) + JSONL (backup).
    Degrades gracefully: if Redis is unavailable, caching is skipped.
    If Qdrant is unavailable, falls back to JSONL keyword search.
    """

    def __init__(
        self,
        qdrant_host:   str = QDRANT_HOST,
        qdrant_port:   int = QDRANT_PORT,
        collection:    str = QDRANT_COLLECTION,
        redis_host:    str = REDIS_HOST,
        redis_port:    int = REDIS_PORT,
        redis_ttl:     int = REDIS_TTL,
        embed_model:   str = EMBED_MODEL,
        backup_path:   str = BACKUP_PATH,
    ):
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._collection  = collection
        self._redis_host  = redis_host
        self._redis_port  = redis_port
        self._redis_ttl   = redis_ttl
        self._embed_model = embed_model
        self._backup_path = Path(backup_path)

        self._qdrant = None
        self._redis  = None
        self._model  = None
        self._norm   = Normaliser()
        self._connected = False

        # Fingerprint cache for deduplication (loaded from backup on connect)
        self._fingerprints: set[str] = set()

    # ──────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self) -> "PaperWarehouse":
        """Connect to Qdrant and Redis. Load embedding model."""
        self._connect_qdrant()
        self._connect_redis()
        self._load_embedder()
        self._load_fingerprints()
        self._connected = True
        logger.info(
            f"Warehouse connected — Qdrant={self._qdrant is not None}, "
            f"Redis={self._redis is not None}, "
            f"Embedder={self._model is not None}"
        )
        return self

    def close(self):
        if self._qdrant:
            try:
                self._qdrant.close()
            except Exception:
                pass
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        self._connected = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()

    def _connect_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(host=self._qdrant_host, port=self._qdrant_port, timeout=10)
            # Ensure collection exists
            existing = [c.name for c in client.get_collections().collections]
            if self._collection not in existing:
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Qdrant: created collection '{self._collection}'")
            self._qdrant = client
            logger.info(f"Qdrant: connected at {self._qdrant_host}:{self._qdrant_port}")
        except ImportError:
            logger.warning("qdrant-client not installed. Run: pip install qdrant-client")
        except Exception as e:
            logger.warning(
                f"Qdrant unavailable ({e}). "
                f"Start with: docker run -d -p 6333:6333 qdrant/qdrant"
            )

    def _connect_redis(self):
        try:
            import redis
            r = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            r.ping()
            self._redis = r
            logger.info(f"Redis: connected at {self._redis_host}:{self._redis_port}")
        except ImportError:
            logger.info("redis not installed — caching disabled. Run: pip install redis")
        except Exception as e:
            logger.info(f"Redis unavailable ({e}) — caching disabled.")

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._embed_model)
            logger.info(f"Embedder: loaded '{self._embed_model}'")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — semantic search disabled.\n"
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Embedder load failed: {e}")

    def _load_fingerprints(self):
        """Load fingerprints from JSONL backup for fast dedup."""
        if not self._backup_path.exists():
            return
        with self._backup_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    p = Paper(**data)
                    fp = self._norm.fingerprint(p)
                    self._fingerprints.add(fp)
                except Exception:
                    pass
        logger.info(f"Warehouse: loaded {len(self._fingerprints)} existing fingerprints")

    # ──────────────────────────────────────────────────────────────────────────
    # Write path
    # ──────────────────────────────────────────────────────────────────────────

    def store(self, papers: list[Paper]) -> int:
        """
        Store papers in the warehouse.

        Pipeline: normalise → deduplicate → embed → Qdrant + JSONL

        Returns the number of newly stored papers.
        """
        if not papers:
            return 0

        # Normalise
        normalised = self._norm.normalise_many(papers)

        # Deduplicate against existing warehouse content
        unique, self._fingerprints = self._norm.deduplicate(
            normalised, existing_fingerprints=self._fingerprints
        )

        if not unique:
            return 0

        # Store to JSONL backup first (always works)
        self._append_jsonl(unique)

        # Store to Qdrant if available
        if self._qdrant and self._model:
            self._store_qdrant(unique)
        elif unique:
            logger.debug(f"Warehouse: stored {len(unique)} to JSONL only (Qdrant/embedder unavailable)")

        # Invalidate Redis cache (new data may affect results)
        if self._redis:
            try:
                keys = self._redis.keys("axiom:search:*")
                if keys:
                    self._redis.delete(*keys)
            except Exception:
                pass

        return len(unique)

    def _append_jsonl(self, papers: list[Paper]):
        with self._backup_path.open("a", encoding="utf-8") as f:
            for p in papers:
                f.write(p.model_dump_json() + "\n")

    def _store_qdrant(self, papers: list[Paper]):
        from qdrant_client.models import PointStruct

        texts    = [_paper_to_text(p) for p in papers]
        vectors  = self._model.encode(texts, show_progress_bar=False).tolist()

        points = []
        for i, (paper, vector) in enumerate(zip(papers, vectors)):
            # Qdrant payload stores all paper fields for retrieval
            payload = json.loads(paper.model_dump_json())
            # Qdrant needs a unique integer ID — use hash of fingerprint
            fp  = self._norm.fingerprint(paper)
            uid = int(hashlib.md5(fp.encode()).hexdigest()[:8], 16)
            points.append(PointStruct(id=uid, vector=vector, payload=payload))

        try:
            self._qdrant.upsert(collection_name=self._collection, points=points)
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Read path
    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 20,
        quality_only: bool = False,
        year_range: Optional[str] = None,
    ) -> list[Paper]:
        """
        Semantic search over the warehouse.

        Checks Redis cache first. If miss, searches Qdrant (semantic)
        or falls back to JSONL keyword search.

        Args:
            query:        Search query.
            limit:        Max results.
            quality_only: If True, only return Q1/Q2/top_conference papers.
            year_range:   Optional "YYYY-YYYY" filter.

        Returns:
            List of Paper objects sorted by relevance.
        """
        cache_key = _cache_key(query, limit, quality_only)

        # 1. Try Redis cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            papers = [Paper(**p) for p in cached]
            return self._post_filter(papers, year_range, limit)

        # 2. Semantic search via Qdrant
        if self._qdrant and self._model:
            papers = self._search_qdrant(query, limit=limit * 3)  # over-fetch for filtering
        else:
            # 3. Fallback: JSONL keyword search
            papers = self._search_jsonl(query, limit=limit * 3)

        # Apply quality filter
        if quality_only:
            from .quality import is_high_quality
            papers = [p for p in papers if is_high_quality(p)]

        papers = self._post_filter(papers, year_range, limit)

        # Write to cache
        self._cache_set(cache_key, [json.loads(p.model_dump_json()) for p in papers])

        return papers

    def _search_qdrant(self, query: str, limit: int) -> list[Paper]:
        vector = self._model.encode(query).tolist()
        try:
            hits = self._qdrant.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return self._search_jsonl(query, limit)

        papers: list[Paper] = []
        for hit in hits:
            try:
                p = Paper(**hit.payload)
                p.relevance_score = round(float(hit.score), 4)
                papers.append(p)
            except Exception as e:
                logger.debug(f"Qdrant: failed to parse hit — {e}")
        return papers

    def _search_jsonl(self, query: str, limit: int) -> list[Paper]:
        """Keyword-based fallback search over the JSONL backup."""
        if not self._backup_path.exists():
            return []

        terms = [t for t in query.lower().split() if len(t) > 2]
        scored: list[tuple[float, Paper]] = []

        with self._backup_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = Paper(**json.loads(line))
                except Exception:
                    continue

                text = " ".join([
                    p.title, p.title,
                    p.abstract or "",
                    " ".join(p.authors[:3]),
                    p.venue or "",
                ]).lower()

                hits     = sum(1 for t in terms if t in text)
                title_h  = sum(1 for t in terms if t in p.title.lower())
                score    = hits + title_h * 2

                if score > 0:
                    scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scored[:limit]]
        # Normalise scores to 0-1
        max_score = scored[0][0] if scored else 1
        for score, p in scored[:limit]:
            p.relevance_score = round(score / max_score, 4)
        return results

    def _post_filter(
        self,
        papers: list[Paper],
        year_range: Optional[str],
        limit: int,
    ) -> list[Paper]:
        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2:
                y_min = int(parts[0]) if parts[0].strip() else None
                y_max = int(parts[1]) if parts[1].strip() else None
                papers = [
                    p for p in papers
                    if not (y_min and p.year and p.year < y_min)
                    and not (y_max and p.year and p.year > y_max)
                ]
        return papers[:limit]

    # ──────────────────────────────────────────────────────────────────────────
    # Cache helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[list[dict]]:
        if not self._redis:
            return None
        try:
            raw = self._redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, data: list[dict]):
        if not self._redis:
            return
        try:
            self._redis.setex(key, self._redis_ttl, json.dumps(data))
        except Exception:
            pass

    def invalidate_cache(self):
        """Manually flush all AXIOM search cache entries from Redis."""
        if self._redis:
            try:
                keys = self._redis.keys("axiom:search:*")
                if keys:
                    self._redis.delete(*keys)
                    logger.info(f"Cache: invalidated {len(keys)} key(s)")
            except Exception as e:
                logger.warning(f"Cache flush failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Stats & maintenance
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return warehouse statistics."""
        total_jsonl = 0
        years: list[int] = []
        sources: dict[str, int] = {}
        quality: dict[str, int] = {}

        if self._backup_path.exists():
            with self._backup_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        total_jsonl += 1
                        if data.get("year"):
                            years.append(data["year"])
                        src = data.get("source", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                        rank = data.get("quality_rank") or "unranked"
                        quality[rank] = quality.get(rank, 0) + 1
                    except Exception:
                        pass

        qdrant_count = 0
        if self._qdrant:
            try:
                info = self._qdrant.get_collection(self._collection)
                qdrant_count = info.points_count or 0
            except Exception:
                pass

        cache_keys = 0
        if self._redis:
            try:
                cache_keys = len(self._redis.keys("axiom:search:*"))
            except Exception:
                pass

        return {
            "total_papers":    total_jsonl,
            "qdrant_vectors":  qdrant_count,
            "redis_cached":    cache_keys,
            "year_range":      f"{min(years)}–{max(years)}" if years else "n/a",
            "sources":         sources,
            "quality_ranks":   quality,
            "backup_path":     str(self._backup_path.resolve()),
            "qdrant_online":   self._qdrant is not None,
            "redis_online":    self._redis is not None,
            "embedder_loaded": self._model is not None,
        }

    def reindex(self):
        """
        Rebuild Qdrant index from the JSONL backup.

        Use this if Qdrant data is lost (e.g. after clearing the Docker volume)
        but the JSONL backup still exists.
        """
        if not self._qdrant or not self._model:
            logger.error("Reindex requires Qdrant + embedder to be online.")
            return

        if not self._backup_path.exists():
            logger.error("No JSONL backup found — nothing to reindex.")
            return

        papers: list[Paper] = []
        with self._backup_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    papers.append(Paper(**json.loads(line)))
                except Exception:
                    pass

        if not papers:
            logger.info("Reindex: backup is empty.")
            return

        logger.info(f"Reindex: re-embedding {len(papers)} papers …")
        # Clear collection
        from qdrant_client.models import VectorParams, Distance
        self._qdrant.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self._fingerprints.clear()
        # Re-store without dedup check (backup is already deduplicated)
        batch_size = 100
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            self._store_qdrant(batch)
            logger.info(f"Reindex: {min(i + batch_size, len(papers))}/{len(papers)}")

        logger.info("Reindex complete.")

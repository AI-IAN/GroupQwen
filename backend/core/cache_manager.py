"""
Semantic Cache Manager

Redis-based semantic caching using embeddings for intelligent query matching.
"""

from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass
import hashlib
import logging
import numpy as np

from backend.utils.redis_client import RedisClient
from backend.utils.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    """Represents a cache hit with metadata."""
    response: str
    similarity: float
    cached_at: str
    hit_count: int
    model_used: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hit_rate: float
    total_requests: int
    total_cached: int
    avg_similarity: float
    cache_size_mb: float


class SemanticCacheManager:
    """
    Redis-based semantic caching using embeddings.

    Attributes:
        distance_threshold: Cosine similarity threshold (default 0.10)
        ttl_seconds: Time-to-live for cache entries (default 86400 = 24h)
        embedding_model: sentence-transformers model for embeddings
    """

    def __init__(
        self,
        redis_url: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold: float = 0.10,
        ttl_seconds: int = 86400
    ):
        """
        Initialize semantic cache manager.

        Args:
            redis_url: Redis connection URL
            embedding_model_name: HuggingFace model name for embeddings
            distance_threshold: Minimum cosine similarity for cache hit
            ttl_seconds: Time-to-live for cache entries
        """
        self.redis_client = RedisClient(redis_url)
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.distance_threshold = distance_threshold
        self.ttl_seconds = ttl_seconds

        # Statistics counters
        self.stats_key_prefix = "cache:stats:"
        self._initialize_stats()

    def _initialize_stats(self):
        """Initialize statistics counters in Redis."""
        if not self.redis_client.exists(f"{self.stats_key_prefix}total_requests"):
            self.redis_client.set(f"{self.stats_key_prefix}total_requests", "0")
            self.redis_client.set(f"{self.stats_key_prefix}cache_hits", "0")

    def _get_cache_key(self, prompt_hash: str) -> str:
        """Generate cache key for a prompt."""
        return f"cache:entry:{prompt_hash}"

    def _get_embedding_key(self, prompt_hash: str) -> str:
        """Generate embedding key for a prompt."""
        return f"cache:embedding:{prompt_hash}"

    def _get_index_key(self) -> str:
        """Get key for the index of all cached prompts."""
        return "cache:index"

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def check(
        self,
        prompt: str,
        threshold: Optional[float] = None
    ) -> Optional[CacheHit]:
        """
        Check cache for semantically similar queries.
        Returns cached response if similarity > threshold.

        Args:
            prompt: User query to check
            threshold: Override default similarity threshold

        Returns:
            CacheHit if found, None otherwise
        """
        # Increment total requests
        self.redis_client.increment(f"{self.stats_key_prefix}total_requests")

        threshold = threshold or self.distance_threshold

        # Generate embedding for the prompt
        query_embedding = self.embedding_model.embed(prompt)

        # Get all cached prompt hashes
        cached_hashes = self._get_all_cached_hashes()

        if not cached_hashes:
            return None

        best_match = None
        best_similarity = 0.0

        # Compare with all cached embeddings
        for prompt_hash in cached_hashes:
            embedding_key = self._get_embedding_key(prompt_hash)
            cached_embedding_str = self.redis_client.get(embedding_key)

            if not cached_embedding_str:
                continue

            # Convert stored embedding back to numpy array
            cached_embedding = np.array(eval(cached_embedding_str))

            # Compute similarity
            similarity = self.embedding_model.cosine_similarity(
                query_embedding,
                cached_embedding
            )

            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = prompt_hash

        # If we found a match, retrieve and return it
        if best_match:
            cache_key = self._get_cache_key(best_match)
            cached_data = self.redis_client.get_json(cache_key)

            if cached_data:
                # Increment hit count
                cached_data["hit_count"] = cached_data.get("hit_count", 0) + 1
                self.redis_client.set_json(cache_key, cached_data, self.ttl_seconds)

                # Increment cache hits
                self.redis_client.increment(f"{self.stats_key_prefix}cache_hits")

                logger.info(
                    f"Cache hit! Similarity: {best_similarity:.3f}, "
                    f"Hit count: {cached_data['hit_count']}"
                )

                return CacheHit(
                    response=cached_data["response"],
                    similarity=best_similarity,
                    cached_at=cached_data["timestamp"],
                    hit_count=cached_data["hit_count"],
                    model_used=cached_data.get("model_used"),
                    metadata=cached_data.get("metadata")
                )

        return None

    def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict] = None
    ):
        """
        Store query-response pair with metadata.

        Args:
            prompt: User query
            response: Model response
            metadata: Optional metadata (model used, latency, etc.)
        """
        prompt_hash = self._hash_prompt(prompt)

        # Generate and store embedding
        embedding = self.embedding_model.embed(prompt)
        embedding_key = self._get_embedding_key(prompt_hash)
        self.redis_client.set(
            embedding_key,
            str(embedding.tolist()),
            self.ttl_seconds
        )

        # Store cache entry
        cache_key = self._get_cache_key(prompt_hash)
        cache_data = {
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "hit_count": 0,
            **(metadata or {})
        }
        self.redis_client.set_json(cache_key, cache_data, self.ttl_seconds)

        # Add to index
        self._add_to_index(prompt_hash)

        logger.debug(f"Stored cache entry: {prompt_hash}")

    def _get_all_cached_hashes(self) -> List[str]:
        """Get all cached prompt hashes from index."""
        index_key = self._get_index_key()
        index_data = self.redis_client.get(index_key)

        if not index_data:
            return []

        return index_data.split(",")

    def _add_to_index(self, prompt_hash: str):
        """Add prompt hash to index."""
        index_key = self._get_index_key()
        current_hashes = self._get_all_cached_hashes()

        if prompt_hash not in current_hashes:
            current_hashes.append(prompt_hash)
            self.redis_client.set(index_key, ",".join(current_hashes))

    def invalidate(self, prompt: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            prompt: Prompt to invalidate

        Returns:
            True if invalidated
        """
        prompt_hash = self._hash_prompt(prompt)
        cache_key = self._get_cache_key(prompt_hash)
        embedding_key = self._get_embedding_key(prompt_hash)

        deleted_cache = self.redis_client.delete(cache_key)
        deleted_embedding = self.redis_client.delete(embedding_key)

        return deleted_cache or deleted_embedding

    def clear_all(self) -> bool:
        """
        Clear all cache entries (WARNING: deletes all cached data).

        Returns:
            True if successful
        """
        # Get all hashes
        cached_hashes = self._get_all_cached_hashes()

        # Delete all entries
        for prompt_hash in cached_hashes:
            self.redis_client.delete(self._get_cache_key(prompt_hash))
            self.redis_client.delete(self._get_embedding_key(prompt_hash))

        # Clear index
        self.redis_client.delete(self._get_index_key())

        # Reset stats
        self.redis_client.set(f"{self.stats_key_prefix}total_requests", "0")
        self.redis_client.set(f"{self.stats_key_prefix}cache_hits", "0")

        logger.info("Cleared all cache entries")
        return True

    def get_stats(self) -> CacheStats:
        """
        Return cache performance statistics.

        Returns:
            CacheStats object
        """
        total_requests = int(
            self.redis_client.get(f"{self.stats_key_prefix}total_requests") or "0"
        )
        cache_hits = int(
            self.redis_client.get(f"{self.stats_key_prefix}cache_hits") or "0"
        )

        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

        cached_hashes = self._get_all_cached_hashes()
        total_cached = len(cached_hashes)

        # Estimate cache size (rough approximation)
        cache_size_mb = total_cached * 0.01  # Rough estimate

        return CacheStats(
            hit_rate=hit_rate,
            total_requests=total_requests,
            total_cached=total_cached,
            avg_similarity=0.0,  # Would need to track this separately
            cache_size_mb=cache_size_mb
        )

    def health_check(self) -> bool:
        """
        Check if cache system is healthy.

        Returns:
            True if healthy
        """
        return self.redis_client.ping()

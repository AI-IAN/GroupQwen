"""
Redis Client Wrapper

Provides a simplified interface to Redis for caching and storage.
"""

import redis
from typing import Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client wrapper with connection pooling and error handling.
    """

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0):
        """
        Initialize Redis client.

        Args:
            url: Redis connection URL
            db: Database number
        """
        self.url = url
        self.db = db
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
        return self._client

    def get(self, key: str) -> Optional[str]:
        """
        Get value by key.

        Args:
            key: Cache key

        Returns:
            Value or None if not found
        """
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set key-value pair with optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except redis.RedisError as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    def get_json(self, key: str) -> Optional[dict]:
        """
        Get JSON value by key.

        Args:
            key: Cache key

        Returns:
            Parsed JSON dict or None
        """
        value = self.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            return None

    def set_json(
        self,
        key: str,
        value: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set JSON value.

        Args:
            key: Cache key
            value: Dict to store as JSON
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            json_str = json.dumps(value)
            return self.set(key, json_str, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON encode error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            return self.client.delete(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    def increment(self, key: str) -> Optional[int]:
        """
        Increment counter.

        Args:
            key: Counter key

        Returns:
            New value or None on error
        """
        try:
            return self.client.incr(key)
        except redis.RedisError as e:
            logger.error(f"Redis INCR error for key {key}: {e}")
            return None

    def flush_db(self) -> bool:
        """
        Flush current database (WARNING: deletes all keys).

        Returns:
            True if successful
        """
        try:
            return self.client.flushdb()
        except redis.RedisError as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            return False

    def ping(self) -> bool:
        """
        Check if Redis is available.

        Returns:
            True if connected
        """
        try:
            return self.client.ping()
        except redis.RedisError as e:
            logger.error(f"Redis PING error: {e}")
            return False

    def get_info(self) -> dict:
        """
        Get Redis server info.

        Returns:
            Info dict
        """
        try:
            return self.client.info()
        except redis.RedisError as e:
            logger.error(f"Redis INFO error: {e}")
            return {}

    def close(self):
        """Close Redis connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

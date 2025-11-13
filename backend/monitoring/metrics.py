"""
Metrics Collection Module

Tracks system performance metrics for continuous optimization.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """Represents a single query log entry."""
    timestamp: datetime
    text: str
    model: str
    latency_ms: float
    cache_hit: bool
    user_id: Optional[str]
    success: bool
    complexity_score: float
    tokens_used: int


@dataclass
class DailyStats:
    """Daily aggregated statistics."""
    date: str
    total_queries: int
    cache_hit_rate: float
    avg_latency_ms: float
    model_usage: Dict[str, int]
    success_rate: float


class MetricsCollector:
    """
    Track system performance metrics for continuous optimization.

    Metrics:
    - Cache hit rate (target: 40-60%)
    - Model utilization by tier
    - Average latency by model
    - User satisfaction (implicit from query patterns)
    - Cost per query (relative units)
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            db_url: PostgreSQL database URL for persistence
        """
        self.db_url = db_url
        self._db = None

        # In-memory buffer for recent metrics
        self.query_logs: List[QueryLog] = []
        self.max_buffer_size = 1000

    def log_query(self, query_log: QueryLog):
        """
        Log a query for analysis.

        Args:
            query_log: Query log entry
        """
        # Add to in-memory buffer
        self.query_logs.append(query_log)

        # Trim buffer if too large
        if len(self.query_logs) > self.max_buffer_size:
            self.query_logs = self.query_logs[-self.max_buffer_size:]

        # Persist to database if configured
        if self.db_url:
            self._persist_to_db(query_log)

        logger.debug(
            f"Query logged: model={query_log.model}, "
            f"latency={query_log.latency_ms:.1f}ms, "
            f"cache_hit={query_log.cache_hit}"
        )

    def _persist_to_db(self, query_log: QueryLog):
        """
        Persist query log to PostgreSQL.

        Args:
            query_log: Query to persist
        """
        # Placeholder for database persistence
        # In production, would use SQLAlchemy or psycopg2
        #
        # INSERT INTO query_logs (
        #     timestamp, query, model_used, latency_ms,
        #     cache_hit, user_id, success
        # ) VALUES (...)

        pass

    def get_daily_stats(self, date: Optional[str] = None) -> DailyStats:
        """
        Get aggregated statistics for a specific day.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Daily statistics
        """
        if date is None:
            target_date = datetime.now().date()
        else:
            target_date = datetime.fromisoformat(date).date()

        # Filter logs for the target date
        day_logs = [
            log for log in self.query_logs
            if log.timestamp.date() == target_date
        ]

        if not day_logs:
            return DailyStats(
                date=str(target_date),
                total_queries=0,
                cache_hit_rate=0.0,
                avg_latency_ms=0.0,
                model_usage={},
                success_rate=0.0
            )

        # Calculate statistics
        total_queries = len(day_logs)
        cache_hits = sum(1 for log in day_logs if log.cache_hit)
        cache_hit_rate = cache_hits / total_queries

        total_latency = sum(log.latency_ms for log in day_logs)
        avg_latency = total_latency / total_queries

        # Model usage counts
        model_usage = {}
        for log in day_logs:
            model_usage[log.model] = model_usage.get(log.model, 0) + 1

        # Success rate
        successful = sum(1 for log in day_logs if log.success)
        success_rate = successful / total_queries

        return DailyStats(
            date=str(target_date),
            total_queries=total_queries,
            cache_hit_rate=cache_hit_rate,
            avg_latency_ms=avg_latency,
            model_usage=model_usage,
            success_rate=success_rate
        )

    def get_model_stats(self, model_name: str) -> Dict:
        """
        Get statistics for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model statistics
        """
        model_logs = [log for log in self.query_logs if log.model == model_name]

        if not model_logs:
            return {
                "model": model_name,
                "query_count": 0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0
            }

        total_queries = len(model_logs)
        total_latency = sum(log.latency_ms for log in model_logs)
        avg_latency = total_latency / total_queries

        successful = sum(1 for log in model_logs if log.success)
        success_rate = successful / total_queries

        return {
            "model": model_name,
            "query_count": total_queries,
            "avg_latency_ms": avg_latency,
            "success_rate": success_rate,
            "total_tokens": sum(log.tokens_used for log in model_logs)
        }

    def get_cache_effectiveness(self) -> Dict:
        """
        Analyze cache effectiveness.

        Returns:
            Cache statistics
        """
        if not self.query_logs:
            return {
                "cache_hit_rate": 0.0,
                "avg_latency_cached_ms": 0.0,
                "avg_latency_uncached_ms": 0.0,
                "latency_reduction": "0%"
            }

        total = len(self.query_logs)
        cache_hits = [log for log in self.query_logs if log.cache_hit]
        cache_misses = [log for log in self.query_logs if not log.cache_hit]

        cache_hit_rate = len(cache_hits) / total

        avg_cached_latency = (
            sum(log.latency_ms for log in cache_hits) / len(cache_hits)
            if cache_hits else 0.0
        )

        avg_uncached_latency = (
            sum(log.latency_ms for log in cache_misses) / len(cache_misses)
            if cache_misses else 0.0
        )

        latency_reduction = (
            (avg_uncached_latency - avg_cached_latency) / avg_uncached_latency * 100
            if avg_uncached_latency > 0 else 0.0
        )

        return {
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_cached_ms": avg_cached_latency,
            "avg_latency_uncached_ms": avg_uncached_latency,
            "latency_reduction": f"{latency_reduction:.1f}%"
        }

    def get_summary_stats(self) -> Dict:
        """
        Get overall summary statistics.

        Returns:
            Summary statistics
        """
        if not self.query_logs:
            return {
                "total_queries": 0,
                "cache_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
                "models_used": []
            }

        total_queries = len(self.query_logs)
        cache_hits = sum(1 for log in self.query_logs if log.cache_hit)
        cache_hit_rate = cache_hits / total_queries

        total_latency = sum(log.latency_ms for log in self.query_logs)
        avg_latency = total_latency / total_queries

        # Unique models used
        models_used = list(set(log.model for log in self.query_logs))

        return {
            "total_queries": total_queries,
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": avg_latency,
            "models_used": models_used
        }

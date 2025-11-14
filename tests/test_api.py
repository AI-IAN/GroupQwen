"""
Tests for API Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.api.main import app
from backend.core.router import RouteDecision
from backend.core.cache_manager import CacheResult


@pytest.fixture
def test_client():
    """Create test client with mocked app state."""
    # Mock components
    mock_cache_manager = Mock()
    mock_query_router = Mock()
    mock_metrics_collector = Mock()
    mock_redis_client = Mock()
    mock_model_loader = Mock()

    # Attach to app state
    app.state.cache_manager = mock_cache_manager
    app.state.query_router = mock_query_router
    app.state.metrics_collector = mock_metrics_collector
    app.state.redis_client = mock_redis_client
    app.state.model_loader = mock_model_loader

    # Create test client
    with TestClient(app) as client:
        yield client, app.state


class TestChatCompletionAPI:
    """Tests for chat completion endpoint."""

    def test_chat_completion_success(self, test_client):
        """Test successful chat completion."""
        client, state = test_client

        # Mock router decision
        state.query_router.route_query.return_value = RouteDecision(
            model="qwen3-8b",
            complexity_score=0.3,
            use_cache=False,
            estimated_latency_ms=100,
            reasoning="Simple query"
        )

        # Mock cache (no hit)
        state.cache_manager.check.return_value = None

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert "usage" in data
        assert "metadata" in data

    def test_chat_completion_empty_messages(self, test_client):
        """Test error handling for empty messages."""
        client, state = test_client

        response = client.post(
            "/v1/chat/completions",
            json={"messages": []}
        )

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_chat_completion_cache_hit(self, test_client):
        """Test cache hit scenario."""
        client, state = test_client

        # Mock router decision with cache enabled
        state.query_router.route_query.return_value = RouteDecision(
            model="cache",
            complexity_score=0.2,
            use_cache=True,
            estimated_latency_ms=5,
            reasoning="Cache hit"
        )

        # Mock cache hit
        state.cache_manager.check.return_value = CacheResult(
            response="Cached response",
            model_used="qwen3-8b",
            timestamp=datetime.now(),
            cache_key="test_key"
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["cache_hit"] is True
        assert "Cached response" in data["choices"][0]["message"]["content"]

    def test_chat_completion_cache_miss(self, test_client):
        """Test cache miss scenario."""
        client, state = test_client

        # Mock router decision
        state.query_router.route_query.return_value = RouteDecision(
            model="qwen3-8b",
            complexity_score=0.4,
            use_cache=False,
            estimated_latency_ms=120,
            reasoning="Normal routing"
        )

        # Mock cache miss
        state.cache_manager.check.return_value = None

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is AI?"}]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["cache_hit"] is False

    def test_chat_completion_with_system_message(self, test_client):
        """Test chat completion with system message."""
        client, state = test_client

        state.query_router.route_query.return_value = RouteDecision(
            model="qwen3-8b",
            complexity_score=0.3,
            use_cache=False,
            estimated_latency_ms=100,
            reasoning="Normal"
        )
        state.cache_manager.check.return_value = None

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
            }
        )

        assert response.status_code == 200

    def test_chat_completion_multi_turn(self, test_client):
        """Test multi-turn conversation."""
        client, state = test_client

        state.query_router.route_query.return_value = RouteDecision(
            model="qwen3-8b",
            complexity_score=0.4,
            use_cache=False,
            estimated_latency_ms=150,
            reasoning="Multi-turn"
        )
        state.cache_manager.check.return_value = None

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }
        )

        assert response.status_code == 200

    def test_chat_completion_force_model(self, test_client):
        """Test forcing a specific model."""
        client, state = test_client

        state.query_router.route_query.return_value = RouteDecision(
            model="qwen3-32b",
            complexity_score=0.3,
            use_cache=False,
            estimated_latency_ms=200,
            reasoning="Forced model"
        )
        state.cache_manager.check.return_value = None

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "qwen3-32b"
            }
        )

        assert response.status_code == 200
        # Verify model was passed to router
        state.query_router.route_query.assert_called_once()
        assert state.query_router.route_query.call_args.kwargs.get("force_model") == "qwen3-32b"


class TestHealthAPI:
    """Tests for health check endpoint."""

    def test_health_check_success(self, test_client):
        """Test successful health check."""
        client, state = test_client

        # Mock redis client
        state.redis_client.ping.return_value = True

        with patch('backend.api.routes.HealthChecker') as mock_health_checker:
            mock_checker_instance = Mock()
            mock_health_checker.return_value = mock_checker_instance

            # Mock health status
            from backend.monitoring.health_check import HealthStatus
            mock_checker_instance.check_health.return_value = HealthStatus(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                checks={"redis": True, "models": True},
                details={"message": "All systems operational"}
            )

            response = client.get("/v1/health")

            assert response.status_code == 200
            data = response.json()

            assert "status" in data
            assert "timestamp" in data
            assert "checks" in data

    def test_health_check_degraded(self, test_client):
        """Test health check with degraded status."""
        client, state = test_client

        # Mock redis failure
        state.redis_client = None

        response = client.get("/v1/health")

        # Should still return 200 but with degraded status
        assert response.status_code == 200


class TestMetricsAPI:
    """Tests for metrics endpoint."""

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        client, state = test_client

        # Mock metrics data
        state.metrics_collector.get_summary_stats.return_value = {
            "total_queries": 100,
            "cache_hit_rate": 0.35,
            "avg_latency_ms": 120.5,
            "models_used": ["qwen3-8b", "qwen3-14b"]
        }

        response = client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["total_queries"] == 100
        assert data["cache_hit_rate"] == 0.35
        assert data["avg_latency_ms"] == 120.5
        assert len(data["models_used"]) == 2

    def test_metrics_disabled(self, test_client):
        """Test metrics when collector is None."""
        client, state = test_client

        state.metrics_collector = None

        response = client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["total_queries"] == 0
        assert data["cache_hit_rate"] == 0.0


class TestCacheAPI:
    """Tests for cache management endpoints."""

    def test_cache_stats(self, test_client):
        """Test cache statistics endpoint."""
        client, state = test_client

        # Mock cache stats
        from backend.core.cache_manager import CacheStats
        state.cache_manager.get_stats.return_value = CacheStats(
            hit_rate=0.42,
            total_cached=150,
            total_requests=357,
            cache_size_mb=12.5
        )

        response = client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["hit_rate"] == 0.42
        assert data["total_cached"] == 150
        assert data["total_requests"] == 357
        assert data["cache_size_mb"] == 12.5

    def test_cache_stats_disabled(self, test_client):
        """Test cache stats when cache is disabled."""
        client, state = test_client

        state.cache_manager = None

        response = client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["hit_rate"] == 0.0
        assert data["total_cached"] == 0

    def test_clear_cache(self, test_client):
        """Test cache clearing endpoint."""
        client, state = test_client

        response = client.post("/v1/cache/clear")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        state.cache_manager.clear_all.assert_called_once()


class TestRootEndpoints:
    """Tests for root and utility endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        client, state = test_client

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "Qwen3 Local AI Orchestration System"
        assert data["status"] == "running"
        assert "version" in data

    def test_ping_endpoint(self, test_client):
        """Test ping endpoint."""
        client, state = test_client

        response = client.get("/ping")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        client, state = test_client

        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        client, state = test_client

        response = client.post(
            "/v1/chat/completions",
            json={}  # Missing 'messages' field
        )

        assert response.status_code == 422

    def test_server_error_handling(self, test_client):
        """Test internal server error handling."""
        client, state = test_client

        # Make router raise an exception
        state.query_router.route_query.side_effect = Exception("Test error")

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test"}]
            }
        )

        assert response.status_code == 500
        assert "detail" in response.json()

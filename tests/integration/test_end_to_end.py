"""
Integration Tests - End-to-End Flow

Tests the complete flow from API → Router → Cache → Inference → Response
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from backend.api.main import app
from backend.core.cache_manager import SemanticCacheManager, CacheResult
from backend.core.router import QueryRouter, RouteDecision
from backend.monitoring.metrics import MetricsCollector


@pytest.fixture
def test_app():
    """Setup test app with real components."""
    # Create real components (with mocks for external dependencies)
    mock_redis = Mock()
    mock_redis.ping.return_value = True

    cache_manager = SemanticCacheManager(redis_url="redis://localhost:6379")
    query_router = QueryRouter(cache_manager=None)  # Start without cache
    metrics_collector = MetricsCollector()

    # Attach to app state
    app.state.cache_manager = cache_manager
    app.state.query_router = query_router
    app.state.metrics_collector = metrics_collector
    app.state.redis_client = mock_redis
    app.state.model_loader = Mock()

    yield app

    # Cleanup
    app.state.cache_manager = None
    app.state.query_router = None
    app.state.metrics_collector = None


class TestFullQueryFlow:
    """Test complete query flow through the system."""

    def test_simple_query_flow(self, test_app):
        """Test complete flow: API → Router → Response."""
        client = TestClient(test_app)

        # Make a request
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0.7
            }
        )

        # Verify response structure
        assert response.status_code == 200
        data = response.json()

        # Verify response format
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data
        assert "metadata" in data

        # Verify choices
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

        # Verify metadata
        assert "cache_hit" in data["metadata"]
        assert "complexity_score" in data["metadata"]
        assert "model_selected" in data["metadata"]

    def test_complex_query_flow(self, test_app):
        """Test complex query routing."""
        client = TestClient(test_app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": "Design a distributed system architecture for handling millions of users"
                }]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Complex queries should route to larger models
        assert data["metadata"]["complexity_score"] > 0.4

    def test_conversation_flow(self, test_app):
        """Test multi-turn conversation."""
        client = TestClient(test_app)

        # First message
        response1 = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is Python?"}
                ]
            }
        )
        assert response1.status_code == 200

        # Follow-up message with context
        response2 = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": response1.json()["choices"][0]["message"]["content"]},
                    {"role": "user", "content": "Tell me more about its features"}
                ]
            }
        )
        assert response2.status_code == 200

    def test_device_routing_flow(self, test_app):
        """Test device-specific routing."""
        client = TestClient(test_app)

        # Mobile device
        response_mobile = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "device": "mobile"
            }
        )
        assert response_mobile.status_code == 200

        # Server device
        response_server = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "device": "server"
            }
        )
        assert response_server.status_code == 200


class TestCacheIntegration:
    """Test cache integration with full flow."""

    @patch('backend.core.cache_manager.SemanticCacheManager.check')
    @patch('backend.core.cache_manager.SemanticCacheManager.store')
    def test_cache_miss_then_store(self, mock_store, mock_check, test_app):
        """Test cache miss → store → cache hit."""
        # First request: cache miss
        mock_check.return_value = None

        client = TestClient(test_app)
        response1 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )

        assert response1.status_code == 200
        data1 = response1.json()

        # Verify cache miss
        assert data1["metadata"]["cache_hit"] is False

        # Verify store was called
        assert mock_store.called

    @patch('backend.core.cache_manager.SemanticCacheManager.check')
    def test_cache_hit_fast_response(self, mock_check, test_app):
        """Test cache hit returns cached response."""
        # Mock cache hit
        mock_check.return_value = CacheResult(
            response="Cached response for Hello",
            similarity=0.98,
            model_used="qwen3-8b",
            metadata={"cached_at": datetime.now().isoformat()}
        )

        # Mock router to return cache decision
        mock_router = Mock()
        mock_router.route_query.return_value = RouteDecision(
            model="cache",
            use_cache=True,
            cache_similarity=0.98,
            estimated_latency_ms=5
        )

        with patch.object(test_app.state, 'query_router', mock_router):
            client = TestClient(test_app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert response.status_code == 200
        data = response.json()

        # Verify cache hit
        assert data["metadata"]["cache_hit"] is True

    @patch('backend.core.cache_manager.SemanticCacheManager.check')
    @patch('backend.core.cache_manager.SemanticCacheManager.store')
    def test_similar_queries_cached(self, mock_store, mock_check, test_app):
        """Test similar queries use cache."""
        # First query: miss
        mock_check.return_value = None

        client = TestClient(test_app)

        response1 = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What is the capital of France?"}]}
        )
        assert response1.status_code == 200

        # Second similar query: hit
        mock_check.return_value = CacheResult(
            response="Paris is the capital of France",
            similarity=0.92,
            model_used="qwen3-8b",
            metadata={}
        )

        # Mock router for cache hit
        mock_router = Mock()
        mock_router.route_query.return_value = RouteDecision(
            model="cache",
            use_cache=True,
            cache_similarity=0.92
        )

        with patch.object(test_app.state, 'query_router', mock_router):
            response2 = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "What's France's capital city?"}]}
            )

        assert response2.status_code == 200
        assert response2.json()["metadata"]["cache_hit"] is True


class TestMetricsTracking:
    """Test metrics tracking through the system."""

    def test_metrics_logged_on_request(self, test_app):
        """Test that metrics are logged for each request."""
        mock_metrics = Mock()
        mock_metrics.log_query.return_value = None
        mock_metrics.get_summary_stats.return_value = {
            "total_queries": 1,
            "cache_hit_rate": 0.0,
            "avg_latency_ms": 100.0,
            "models_used": ["qwen3-8b"]
        }

        with patch.object(test_app.state, 'metrics_collector', mock_metrics):
            client = TestClient(test_app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Test"}]}
            )

        assert response.status_code == 200

        # Verify metrics were logged
        assert mock_metrics.log_query.called

        # Verify log_query was called with correct structure
        call_args = mock_metrics.log_query.call_args[0][0]
        assert hasattr(call_args, 'text')
        assert hasattr(call_args, 'model')
        assert hasattr(call_args, 'latency_ms')

    def test_metrics_summary_accessible(self, test_app):
        """Test metrics summary is accessible."""
        client = TestClient(test_app)

        # Make a request first
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test"}]}
        )

        # Get metrics
        metrics_response = client.get("/v1/metrics")

        assert metrics_response.status_code == 200
        data = metrics_response.json()

        assert "total_queries" in data
        assert "cache_hit_rate" in data
        assert "avg_latency_ms" in data


class TestHealthAndStatus:
    """Test health check and status endpoints."""

    @patch('backend.api.routes.HealthChecker')
    def test_health_check_integration(self, mock_health_checker, test_app):
        """Test health check with all components."""
        # Mock health checker
        mock_checker_instance = Mock()
        mock_checker_instance.check_health.return_value = Mock(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            checks={
                "redis": True,
                "models": True,
                "cache": True,
                "router": True
            },
            details={}
        )
        mock_health_checker.return_value = mock_checker_instance

        client = TestClient(test_app)
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "checks" in data

    def test_cache_stats_integration(self, test_app):
        """Test cache stats endpoint."""
        client = TestClient(test_app)
        response = client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()

        assert "hit_rate" in data
        assert "total_cached" in data


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_request_format(self, test_app):
        """Test handling of invalid request format."""
        client = TestClient(test_app)

        # Missing required field
        response = client.post(
            "/v1/chat/completions",
            json={}
        )

        # Should handle gracefully
        assert response.status_code in [400, 422]

    def test_empty_messages_error(self, test_app):
        """Test empty messages error handling."""
        client = TestClient(test_app)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": []}
        )

        assert response.status_code == 400

    def test_backend_error_graceful(self, test_app):
        """Test graceful handling of backend errors."""
        # Mock router to raise exception
        mock_router = Mock()
        mock_router.route_query.side_effect = Exception("Backend error")

        with patch.object(test_app.state, 'query_router', mock_router):
            client = TestClient(test_app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Test"}]}
            )

        assert response.status_code == 500


class TestPerformance:
    """Test performance characteristics."""

    def test_response_time_reasonable(self, test_app):
        """Test that responses are returned in reasonable time."""
        client = TestClient(test_app)

        start_time = time.time()

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]}
        )

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        assert response.status_code == 200

        # Should respond in under 1 second for mocked responses
        assert elapsed_time < 1000

    def test_cache_faster_than_generation(self, test_app):
        """Test that cache hits are faster than generation."""
        # Mock router for cache hit
        mock_router = Mock()
        mock_router.route_query.return_value = RouteDecision(
            model="cache",
            use_cache=True,
            cache_similarity=0.95,
            estimated_latency_ms=5
        )

        # Mock cache hit
        mock_cache = Mock()
        mock_cache.check.return_value = CacheResult(
            response="Cached response",
            similarity=0.95,
            model_used="qwen3-8b",
            metadata={}
        )

        with patch.object(test_app.state, 'query_router', mock_router), \
             patch.object(test_app.state, 'cache_manager', mock_cache):

            client = TestClient(test_app)

            start_time = time.time()
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]}
            )
            cache_time = (time.time() - start_time) * 1000

        assert response.status_code == 200

        # Cache should be very fast (under 100ms including overhead)
        assert cache_time < 100


class TestModelSelection:
    """Test model selection integration."""

    def test_forced_model_used(self, test_app):
        """Test that forced model is actually used."""
        mock_router = Mock()
        mock_router.route_query.return_value = RouteDecision(
            model="qwen3-32b",
            reasoning="Forced model"
        )

        with patch.object(test_app.state, 'query_router', mock_router):
            client = TestClient(test_app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Test"}],
                    "model": "qwen3-32b"
                }
            )

        assert response.status_code == 200
        data = response.json()

        # Verify forced model in routing call
        mock_router.route_query.assert_called()
        call_kwargs = mock_router.route_query.call_args[1]
        assert call_kwargs['force_model'] == "qwen3-32b"

    def test_automatic_model_selection(self, test_app):
        """Test automatic model selection based on complexity."""
        client = TestClient(test_app)

        # Simple query
        response_simple = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]}
        )

        # Complex query
        response_complex = client.post(
            "/v1/chat/completions",
            json={"messages": [{
                "role": "user",
                "content": "Design a complete microservices architecture with detailed implementation"
            }]}
        )

        assert response_simple.status_code == 200
        assert response_complex.status_code == 200

        # Complex query should have higher complexity score
        simple_score = response_simple.json()["metadata"]["complexity_score"]
        complex_score = response_complex.json()["metadata"]["complexity_score"]

        assert complex_score > simple_score

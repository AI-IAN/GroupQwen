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
def mock_app_state():
    """Create mock app state with all components."""
    # Mock router
    mock_router = Mock()
    mock_router.route_query.return_value = RouteDecision(
        model="qwen3-8b",
        complexity_score=0.5,
        use_cache=False,
        estimated_latency_ms=100,
        reasoning="Test routing"
    )

    # Mock cache manager
    mock_cache = Mock()
    mock_cache.check.return_value = None
    mock_cache.store.return_value = None
    mock_cache.get_stats.return_value = Mock(
        hit_rate=0.45,
        total_cached=120,
        total_requests=250,
        cache_size_mb=15.5
    )

    # Mock metrics collector
    mock_metrics = Mock()
    mock_metrics.log_query.return_value = None
    mock_metrics.get_summary_stats.return_value = {
        "total_queries": 100,
        "cache_hit_rate": 0.45,
        "avg_latency_ms": 150.5,
        "models_used": ["qwen3-8b", "qwen3-14b"]
    }

    # Mock redis client
    mock_redis = Mock()

    # Mock model loader
    mock_loader = Mock()

    return {
        "query_router": mock_router,
        "cache_manager": mock_cache,
        "metrics_collector": mock_metrics,
        "redis_client": mock_redis,
        "model_loader": mock_loader
    }


class TestChatCompletionAPI:
    """Test chat completion endpoint."""

    def test_chat_completion_success(self, mock_app_state):
        """Test successful chat completion."""
        # Patch app.state
        with patch.object(app.state, 'query_router', mock_app_state['query_router']), \
             patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert "usage" in data

    def test_chat_completion_empty_messages(self, mock_app_state):
        """Test error handling for empty messages."""
        with patch.object(app.state, 'query_router', mock_app_state['query_router']):
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": []}
            )

        assert response.status_code == 400
        assert "detail" in response.json()

    def test_chat_completion_with_system_message(self, mock_app_state):
        """Test chat completion with system message."""
        with patch.object(app.state, 'query_router', mock_app_state['query_router']), \
             patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                }
            )

        assert response.status_code == 200

    def test_chat_completion_with_conversation(self, mock_app_state):
        """Test chat completion with conversation history."""
        with patch.object(app.state, 'query_router', mock_app_state['query_router']), \
             patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "4"},
                        {"role": "user", "content": "What about 2+3?"}
                    ]
                }
            )

        assert response.status_code == 200

    def test_chat_completion_cache_hit(self, mock_app_state):
        """Test cache hit scenario."""
        # Mock cache hit
        mock_cache = mock_app_state['cache_manager']
        mock_cache.check.return_value = CacheResult(
            response="Cached response",
            similarity=0.95,
            model_used="qwen3-8b",
            metadata={}
        )

        # Mock router to indicate cache use
        mock_router = mock_app_state['query_router']
        mock_router.route_query.return_value = RouteDecision(
            model="cache",
            use_cache=True,
            cache_similarity=0.95,
            estimated_latency_ms=5
        )

        with patch.object(app.state, 'query_router', mock_router), \
             patch.object(app.state, 'cache_manager', mock_cache), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert response.status_code == 200
        data = response.json()

        # Verify cache hit metadata
        assert "metadata" in data
        assert data["metadata"]["cache_hit"] is True

    def test_chat_completion_cache_miss(self, mock_app_state):
        """Test cache miss scenario."""
        # Mock cache miss
        mock_cache = mock_app_state['cache_manager']
        mock_cache.check.return_value = None

        with patch.object(app.state, 'query_router', mock_app_state['query_router']), \
             patch.object(app.state, 'cache_manager', mock_cache), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert response.status_code == 200
        data = response.json()

        # Verify cache miss
        assert "metadata" in data
        assert data["metadata"]["cache_hit"] is False

        # Verify cache.store was called
        mock_cache.store.assert_called_once()

    def test_chat_completion_forced_model(self, mock_app_state):
        """Test forcing a specific model."""
        mock_router = mock_app_state['query_router']
        mock_router.route_query.return_value = RouteDecision(
            model="qwen3-32b",
            reasoning="Forced model"
        )

        with patch.object(app.state, 'query_router', mock_router), \
             patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "model": "qwen3-32b"
                }
            )

        assert response.status_code == 200

    def test_chat_completion_device_parameter(self, mock_app_state):
        """Test device parameter routing."""
        with patch.object(app.state, 'query_router', mock_app_state['query_router']), \
             patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']), \
             patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "device": "mobile"
                }
            )

        assert response.status_code == 200

        # Verify router was called with device parameter
        mock_app_state['query_router'].route_query.assert_called()
        call_kwargs = mock_app_state['query_router'].route_query.call_args[1]
        assert call_kwargs['device'] == "mobile"

    def test_chat_completion_backend_error(self, mock_app_state):
        """Test backend error handling."""
        # Mock router to raise an exception
        mock_router = mock_app_state['query_router']
        mock_router.route_query.side_effect = Exception("Backend error")

        with patch.object(app.state, 'query_router', mock_router):
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]}
            )

        assert response.status_code == 500


class TestHealthAPI:
    """Test health check endpoint."""

    @patch('backend.api.routes.HealthChecker')
    def test_health_check_success(self, mock_health_checker, mock_app_state):
        """Test health check endpoint."""
        # Mock health checker
        mock_checker_instance = Mock()
        mock_checker_instance.check_health.return_value = Mock(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            checks={"redis": True, "models": True},
            details={}
        )
        mock_health_checker.return_value = mock_checker_instance

        with patch.object(app.state, 'redis_client', mock_app_state['redis_client']), \
             patch.object(app.state, 'model_loader', mock_app_state['model_loader']):

            client = TestClient(app)
            response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data

    @patch('backend.api.routes.HealthChecker')
    def test_health_check_error(self, mock_health_checker, mock_app_state):
        """Test health check with error."""
        # Mock health checker to raise exception
        mock_health_checker.side_effect = Exception("Health check failed")

        with patch.object(app.state, 'redis_client', mock_app_state['redis_client']), \
             patch.object(app.state, 'model_loader', mock_app_state['model_loader']):

            client = TestClient(app)
            response = client.get("/v1/health")

        # Should return unhealthy status instead of 500 error
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestMetricsAPI:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self, mock_app_state):
        """Test metrics endpoint."""
        with patch.object(app.state, 'metrics_collector', mock_app_state['metrics_collector']):
            client = TestClient(app)
            response = client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "cache_hit_rate" in data
        assert "avg_latency_ms" in data
        assert "models_used" in data

    def test_metrics_disabled(self):
        """Test metrics when collector is disabled."""
        with patch.object(app.state, 'metrics_collector', None):
            client = TestClient(app)
            response = client.get("/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 0
        assert data["cache_hit_rate"] == 0.0


class TestCacheAPI:
    """Test cache management endpoints."""

    def test_cache_stats(self, mock_app_state):
        """Test cache stats endpoint."""
        with patch.object(app.state, 'cache_manager', mock_app_state['cache_manager']):
            client = TestClient(app)
            response = client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert "hit_rate" in data
        assert "total_cached" in data
        assert "total_requests" in data
        assert "cache_size_mb" in data
        assert data["hit_rate"] == 0.45
        assert data["total_cached"] == 120

    def test_cache_stats_disabled(self):
        """Test cache stats when cache is disabled."""
        with patch.object(app.state, 'cache_manager', None):
            client = TestClient(app)
            response = client.get("/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["hit_rate"] == 0.0
        assert data["total_cached"] == 0

    def test_clear_cache(self, mock_app_state):
        """Test cache clearing."""
        mock_cache = mock_app_state['cache_manager']
        mock_cache.clear_all.return_value = None

        with patch.object(app.state, 'cache_manager', mock_cache):
            client = TestClient(app)
            response = client.post("/v1/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify clear_all was called
        mock_cache.clear_all.assert_called_once()

    def test_clear_cache_disabled(self):
        """Test cache clearing when cache is disabled."""
        with patch.object(app.state, 'cache_manager', None):
            client = TestClient(app)
            response = client.post("/v1/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disabled"


class TestRootEndpoints:
    """Test root and ping endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data

    def test_ping_endpoint(self):
        """Test ping endpoint."""
        client = TestClient(app)
        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestModelsAPI:
    """Test models listing endpoint."""

    def test_list_models(self):
        """Test listing available models."""
        client = TestClient(app)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

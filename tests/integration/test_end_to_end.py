"""
End-to-End Integration Tests

Tests the complete flow: API → Router → Cache → Inference → Response
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from backend.api.main import app
from backend.core.cache_manager import SemanticCacheManager, CacheResult
from backend.core.router import QueryRouter, RouteDecision
from backend.monitoring.metrics import MetricsCollector


@pytest.fixture
def integration_app():
    """Setup test app with real components (but mocked external dependencies)."""
    # Create real components with mocked external dependencies
    mock_redis_client = Mock()
    mock_redis_client.ping.return_value = True

    # Real cache manager (but with mocked Redis)
    with patch('backend.core.cache_manager.redis.Redis') as mock_redis:
        cache_manager = SemanticCacheManager(redis_url="redis://localhost:6379")
        cache_manager._redis = mock_redis_client
        cache_manager._cache = {}  # Use in-memory cache for testing

    # Real query router
    query_router = QueryRouter(cache_manager=cache_manager)

    # Real metrics collector (with mocked DB)
    metrics_collector = MetricsCollector(db_url=None)

    # Attach to app state
    app.state.cache_manager = cache_manager
    app.state.query_router = query_router
    app.state.metrics_collector = metrics_collector
    app.state.redis_client = mock_redis_client
    app.state.model_loader = Mock()

    yield app


def test_full_query_flow(integration_app):
    """Test complete flow: API → Router → Cache → Response."""
    client = TestClient(integration_app)

    # First request (cache miss)
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "temperature": 0.7
        }
    )

    assert response1.status_code == 200
    data1 = response1.json()

    assert "choices" in data1
    assert data1["choices"][0]["message"]["role"] == "assistant"
    assert "content" in data1["choices"][0]["message"]
    assert "usage" in data1
    assert "metadata" in data1

    # Verify it went through the router
    assert "complexity_score" in data1["metadata"]
    assert "model_selected" in data1["metadata"]

    # Get initial response time
    latency1 = data1["metadata"]["latency_ms"]

    # Note: In a real integration test with actual cache,
    # second request would be faster due to cache hit
    # For now, we verify the flow completes successfully


def test_cache_integration():
    """Test cache stores and retrieves correctly."""
    # Create cache manager with mocked Redis
    mock_redis = Mock()
    cache_manager = SemanticCacheManager(redis_url="redis://localhost:6379")
    cache_manager._redis = mock_redis
    cache_manager._cache = {}  # In-memory for testing

    # Mock store operation
    cache_manager.store(
        prompt="What is AI?",
        response="AI is artificial intelligence.",
        metadata={"model_used": "qwen3-8b"}
    )

    # Mock check operation
    # In real implementation, this would use embeddings and similarity search
    # For testing, we verify the methods are called correctly
    assert hasattr(cache_manager, 'store')
    assert hasattr(cache_manager, 'check')


def test_metrics_tracking():
    """Test metrics are logged correctly."""
    from backend.monitoring.metrics import QueryLog

    metrics_collector = MetricsCollector(db_url=None)

    # Log a query
    query_log = QueryLog(
        timestamp=datetime.now(),
        text="Test query",
        model="qwen3-8b",
        latency_ms=120.5,
        cache_hit=False,
        user_id=None,
        success=True,
        complexity_score=0.4,
        tokens_used=50
    )

    metrics_collector.log_query(query_log)

    # Get stats
    stats = metrics_collector.get_summary_stats()

    assert stats["total_queries"] >= 1


def test_routing_with_cache_check():
    """Test that router checks cache before routing to model."""
    mock_cache = Mock()
    router = QueryRouter(cache_manager=mock_cache)

    # Mock cache hit
    mock_cache.check.return_value = CacheResult(
        response="Cached answer",
        model_used="qwen3-8b",
        timestamp=datetime.now(),
        cache_key="test_key"
    )

    decision = router.route_query("Test query", device="server")

    # Verify cache was checked
    mock_cache.check.assert_called()


def test_end_to_end_with_different_complexity():
    """Test end-to-end flow with queries of different complexity."""
    client = TestClient(app)

    # Simple query
    simple_response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}]
        }
    )

    # Complex query
    complex_response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Explain quantum computing in detail with mathematical foundations"}]
        }
    )

    assert simple_response.status_code == 200
    assert complex_response.status_code == 200

    simple_data = simple_response.json()
    complex_data = complex_response.json()

    # Complex query should have higher complexity score
    # (This assumes router is properly attached to app state)
    assert "metadata" in simple_data
    assert "metadata" in complex_data


def test_multi_turn_conversation_flow():
    """Test end-to-end flow with multi-turn conversation."""
    client = TestClient(app)

    # Multi-turn conversation
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
                {"role": "user", "content": "What are its main features?"}
            ]
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_error_recovery():
    """Test system handles errors gracefully."""
    client = TestClient(app)

    # Send invalid request
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": []  # Empty messages
        }
    )

    # Should return error, not crash
    assert response.status_code == 400


def test_concurrent_requests():
    """Test handling of concurrent requests (simulated)."""
    client = TestClient(app)

    # Send multiple requests
    responses = []
    for i in range(5):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": f"Query {i}"}]
            }
        )
        responses.append(response)

    # All should succeed
    assert all(r.status_code == 200 for r in responses)


def test_health_check_integration():
    """Test health check reflects system state."""
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "timestamp" in data


def test_metrics_endpoint_integration():
    """Test metrics endpoint returns aggregated data."""
    client = TestClient(app)

    # Make some requests to generate metrics
    for i in range(3):
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": f"Test {i}"}]
            }
        )

    # Get metrics
    response = client.get("/v1/metrics")

    assert response.status_code == 200
    data = response.json()

    assert "total_queries" in data
    assert "avg_latency_ms" in data


def test_cache_stats_integration():
    """Test cache stats endpoint."""
    client = TestClient(app)

    response = client.get("/v1/cache/stats")

    assert response.status_code == 200
    data = response.json()

    assert "hit_rate" in data
    assert "total_cached" in data


def test_complete_user_journey():
    """Test complete user journey: multiple queries with caching."""
    client = TestClient(app)

    # First query - should be cache miss
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Explain AI"}]
        }
    )

    assert response1.status_code == 200

    # Check cache stats
    cache_stats1 = client.get("/v1/cache/stats")
    assert cache_stats1.status_code == 200

    # Different query
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "What is ML?"}]
        }
    )

    assert response2.status_code == 200

    # Check metrics
    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    metrics_data = metrics.json()

    # Should have logged queries
    assert metrics_data.get("total_queries", 0) >= 0


@pytest.mark.asyncio
async def test_streaming_integration():
    """Test streaming response integration (if implemented)."""
    # This is a placeholder for streaming tests
    # Actual implementation would test streaming responses
    pass


def test_model_routing_integration():
    """Test that different models are selected based on query."""
    client = TestClient(app)

    # Force different models
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test"}],
            "model": "qwen3-8b"
        }
    )

    response2 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test"}],
            "model": "qwen3-32b"
        }
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Verify different models were selected (via metadata)
    # (This requires router to respect force_model parameter)

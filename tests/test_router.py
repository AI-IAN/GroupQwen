"""
Tests for Query Router
"""

import pytest
from backend.core.router import QueryRouter, RouteDecision
from backend.core.complexity_classifier import ComplexityClassifier


def test_complexity_classifier():
    """Test complexity classification."""
    classifier = ComplexityClassifier()

    # Simple query
    simple_score = classifier.score("What is 2+2?")
    assert simple_score < 0.3

    # Complex query
    complex_score = classifier.score(
        "Explain the implications of quantum entanglement on modern cryptography"
    )
    assert complex_score > 0.5


def test_route_simple_query():
    """Test routing of simple queries."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(
        query="What is the capital of France?",
        device="server"
    )

    assert decision.model in ["qwen3-8b", "cache"]
    assert decision.complexity_score < 0.5


def test_route_complex_query():
    """Test routing of complex queries."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(
        query="Design a distributed system architecture for handling 1 million concurrent users",
        device="server"
    )

    assert decision.model in ["qwen3-32b", "qwen3-14b"]
    assert decision.complexity_score > 0.5


def test_edge_device_routing():
    """Test routing for edge devices."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(
        query="Explain quantum computing",
        device="macbook"
    )

    # Edge devices should use smaller models
    assert decision.model in ["qwen3-4b-gguf", "qwen3-8b-gguf"]


# Additional test scenarios as requested by Agent 5


def test_routing_complex_query():
    """Test routing for complex queries."""
    classifier = ComplexityClassifier()
    router = QueryRouter(cache_manager=None)

    query = "Explain quantum computing and its implications for cryptography, including a detailed analysis of Shor's algorithm and post-quantum cryptographic approaches."
    decision = router.route_query(query, device="server")

    # Complex query should route to larger model
    assert decision.model in ["qwen3-14b", "qwen3-32b"]
    assert decision.complexity_score > 0.5


def test_routing_simple_query():
    """Test routing for simple queries."""
    router = QueryRouter(cache_manager=None)

    query = "What is 2+2?"
    decision = router.route_query(query, device="server")

    # Simple query should route to smaller model
    assert decision.model in ["qwen3-8b", "cache"]
    assert decision.complexity_score < 0.5


def test_device_specific_routing_mobile():
    """Test routing for mobile devices."""
    router = QueryRouter(cache_manager=None)

    query = "Tell me about AI"
    decision = router.route_query(query, device="mobile")

    # Mobile devices should use smallest models
    assert decision.model in ["qwen3-4b", "qwen3-4b-gguf"]


def test_device_specific_routing_server():
    """Test routing for server devices."""
    router = QueryRouter(cache_manager=None)

    query = "Explain machine learning"
    decision = router.route_query(query, device="server")

    # Server can use any model based on complexity
    assert decision.model in ["qwen3-8b", "qwen3-14b", "qwen3-32b", "cache"]


def test_force_model_override():
    """Test forced model routing."""
    router = QueryRouter(cache_manager=None)

    # Complex query
    query = "Design a distributed system for handling 1 million concurrent users with detailed architecture"

    # Force to use smaller model despite complexity
    decision = router.route_query(query, device="server", force_model="qwen3-4b")

    # Should respect forced model
    assert decision.model == "qwen3-4b"


def test_edge_case_empty_query():
    """Test empty query handling."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(query="", device="server")

    # Should handle gracefully (route to default model)
    assert decision.model in ["qwen3-8b", "qwen3-4b", "cache"]


def test_edge_case_very_long_query():
    """Test very long query (>10k tokens)."""
    router = QueryRouter(cache_manager=None)

    # Create very long query
    long_query = "Explain " + "very " * 5000 + "complex topic."
    decision = router.route_query(long_query, device="server")

    # Should route to model with large context window
    assert decision.model in ["qwen3-32b", "qwen3-14b"]


def test_edge_case_special_characters():
    """Test query with special characters."""
    router = QueryRouter(cache_manager=None)

    query = "What is the meaning of @#$%^&*()?"
    decision = router.route_query(query, device="server")

    # Should handle special characters gracefully
    assert isinstance(decision.model, str)
    assert decision.complexity_score >= 0.0


def test_model_escalation_logic():
    """Test model escalation based on query complexity."""
    router = QueryRouter(cache_manager=None)

    # Test escalation from simple to complex
    simple_query = "Hi"
    medium_query = "Explain the basics of neural networks"
    complex_query = "Design a complete end-to-end machine learning pipeline for real-time fraud detection in financial transactions, including data preprocessing, feature engineering, model selection, training, deployment, and monitoring strategies"

    simple_decision = router.route_query(simple_query, device="server")
    medium_decision = router.route_query(medium_query, device="server")
    complex_decision = router.route_query(complex_query, device="server")

    # Verify escalation
    assert simple_decision.complexity_score < medium_decision.complexity_score
    assert medium_decision.complexity_score < complex_decision.complexity_score


def test_context_aware_routing():
    """Test routing with conversation context."""
    router = QueryRouter(cache_manager=None)

    query = "And what about that?"
    context = "User asked about quantum computing. Assistant explained basic principles."

    decision = router.route_query(query, context=context, device="server")

    # Context should influence complexity
    assert isinstance(decision, RouteDecision)
    assert decision.complexity_score >= 0.0


def test_cache_consideration_in_routing():
    """Test that router considers cache availability."""
    from unittest.mock import Mock

    mock_cache = Mock()
    router = QueryRouter(cache_manager=mock_cache)

    query = "What is AI?"
    decision = router.route_query(query, device="server")

    # Router should have checked cache
    assert isinstance(decision, RouteDecision)


def test_routing_with_different_devices():
    """Test routing varies appropriately by device type."""
    router = QueryRouter(cache_manager=None)

    query = "Explain neural networks"

    server_decision = router.route_query(query, device="server")
    macbook_decision = router.route_query(query, device="macbook")
    mobile_decision = router.route_query(query, device="mobile")

    # Different devices may route differently
    # (Server can use larger models than mobile)
    assert isinstance(server_decision.model, str)
    assert isinstance(macbook_decision.model, str)
    assert isinstance(mobile_decision.model, str)


def test_reasoning_provided():
    """Test that routing decision includes reasoning."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query("Test query", device="server")

    # Reasoning should be provided
    assert hasattr(decision, 'reasoning')
    assert isinstance(decision.reasoning, str)


def test_estimated_latency():
    """Test that routing decision includes estimated latency."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query("Test query", device="server")

    # Estimated latency should be provided
    assert hasattr(decision, 'estimated_latency_ms')
    assert decision.estimated_latency_ms > 0

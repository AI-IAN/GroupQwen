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

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


# Additional test scenarios for comprehensive coverage

def test_routing_very_simple_query():
    """Test routing for very simple queries."""
    router = QueryRouter(cache_manager=None)

    query = "What is 2+2?"
    decision = router.route_query(query, device="server")

    # Simple arithmetic should use smallest model
    assert decision.model == "qwen3-8b"
    assert decision.complexity_score < 0.3


def test_routing_very_complex_query():
    """Test routing for very complex queries."""
    router = QueryRouter(cache_manager=None)

    query = "Design a distributed microservices architecture for a global e-commerce platform handling millions of transactions, including fault tolerance, data consistency, scalability, security, and detailed implementation of each service."
    decision = router.route_query(query, device="server")

    # Very complex query should use largest model
    assert decision.model in ["qwen3-32b", "qwen3-32b-thinking"]
    assert decision.complexity_score > 0.5


def test_routing_moderate_complexity():
    """Test routing for moderate complexity queries."""
    router = QueryRouter(cache_manager=None)

    query = "Explain the difference between TCP and UDP protocols"
    decision = router.route_query(query, device="server")

    # Moderate complexity should use 8B or 14B
    assert decision.model in ["qwen3-8b", "qwen3-14b"]


def test_device_specific_routing_mobile():
    """Test routing for mobile devices."""
    router = QueryRouter(cache_manager=None)

    query = "Hello, how are you?"
    decision = router.route_query(query, device="mobile")

    # Mobile should use edge models
    assert decision.model in ["qwen3-4b-gguf", "qwen3-8b-gguf"]


def test_device_specific_routing_macbook():
    """Test routing for macbook devices."""
    router = QueryRouter(cache_manager=None)

    query = "Explain machine learning basics"
    decision = router.route_query(query, device="macbook")

    # Macbook should use edge models (GGUF)
    assert decision.model in ["qwen3-4b-gguf", "qwen3-8b-gguf"]


def test_force_model_override():
    """Test forced model routing."""
    router = QueryRouter(cache_manager=None)

    query = "This is a very complex query that would normally route to a large model"
    decision = router.route_query(query, device="server", force_model="qwen3-4b")

    # Forced model should override routing logic
    assert decision.model == "qwen3-4b"
    assert "forced" in decision.reasoning.lower()


def test_force_model_with_cache_check():
    """Test forced model with cache manager."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(
        query="Test query",
        device="server",
        force_model="qwen3-14b"
    )

    # Force model should bypass cache check
    assert decision.model == "qwen3-14b"


def test_edge_case_empty_query():
    """Test empty query handling."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(query="", device="server")

    # Should handle gracefully and route to smallest model
    assert decision.model is not None


def test_edge_case_very_long_query():
    """Test very long query (>10k tokens)."""
    router = QueryRouter(cache_manager=None)

    # Create a very long query
    long_query = "Explain quantum computing. " * 5000
    decision = router.route_query(query=long_query, device="server")

    # Should route to model with large context window
    assert decision.model in ["qwen3-14b", "qwen3-32b"]


def test_edge_case_special_characters():
    """Test query with special characters."""
    router = QueryRouter(cache_manager=None)

    query = "What is the meaning of 你好世界 and émojis like 🚀?"
    decision = router.route_query(query, device="server")

    # Should handle special characters gracefully
    assert decision.model is not None


def test_edge_case_whitespace_only():
    """Test query with only whitespace."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(query="   \n\t  ", device="server")

    # Should handle gracefully
    assert decision.model is not None


def test_edge_case_repeated_words():
    """Test query with many repeated words."""
    router = QueryRouter(cache_manager=None)

    query = "test " * 1000
    decision = router.route_query(query, device="server")

    # Should still route successfully
    assert decision.model is not None


def test_model_escalation_logic():
    """Test model escalation from smaller to larger models."""
    router = QueryRouter(cache_manager=None)

    # Test escalation path
    next_model = router.should_escalate(
        current_model="qwen3-8b",
        response="Test response",
        confidence_score=0.3  # Low confidence
    )

    # Should escalate to next larger model
    assert next_model == "qwen3-14b"


def test_model_escalation_from_14b():
    """Test escalation from 14B model."""
    router = QueryRouter(cache_manager=None)

    next_model = router.should_escalate(
        current_model="qwen3-14b",
        response="Test response",
        confidence_score=0.35  # Low confidence
    )

    # Should escalate to 32B
    assert next_model == "qwen3-32b"


def test_no_escalation_high_confidence():
    """Test no escalation when confidence is high."""
    router = QueryRouter(cache_manager=None)

    next_model = router.should_escalate(
        current_model="qwen3-8b",
        response="Test response",
        confidence_score=0.9  # High confidence
    )

    # Should not escalate
    assert next_model is None


def test_routing_with_context():
    """Test routing with conversation context."""
    router = QueryRouter(cache_manager=None)

    context = "User: What is Python?\nAssistant: Python is a programming language."
    query = "Tell me more about its features"

    decision = router.route_query(query=query, context=context, device="server")

    # Should consider context in routing
    assert decision.model is not None


def test_routing_code_related_query():
    """Test routing for code-related queries."""
    router = QueryRouter(cache_manager=None)

    query = "Write a Python function to implement binary search"
    decision = router.route_query(query, device="server")

    # Code queries might have moderate complexity
    assert decision.model in ["qwen3-8b", "qwen3-14b", "qwen3-32b"]


def test_routing_math_query():
    """Test routing for mathematical queries."""
    router = QueryRouter(cache_manager=None)

    query = "Solve the differential equation dy/dx = 2x"
    decision = router.route_query(query, device="server")

    # Math queries can vary in complexity
    assert decision.model is not None


def test_get_model_for_vision_task():
    """Test getting specialized model for vision tasks."""
    router = QueryRouter(cache_manager=None)

    model = router.get_model_for_task("vision")
    assert model == "qwen3-vl"


def test_get_model_for_translation_task():
    """Test getting specialized model for translation."""
    router = QueryRouter(cache_manager=None)

    model = router.get_model_for_task("translation")
    assert model == "qwen3-mt"


def test_get_model_for_unknown_task():
    """Test getting model for unknown task type."""
    router = QueryRouter(cache_manager=None)

    model = router.get_model_for_task("unknown_task")
    # Should default to largest general model
    assert model == "qwen3-32b"


def test_routing_stats_without_cache():
    """Test getting routing stats without cache manager."""
    router = QueryRouter(cache_manager=None)

    stats = router.get_routing_stats()
    assert isinstance(stats, dict)


def test_decision_metadata():
    """Test that routing decision includes proper metadata."""
    router = QueryRouter(cache_manager=None)

    decision = router.route_query(
        query="What is the capital of France?",
        device="server"
    )

    # Verify decision metadata
    assert hasattr(decision, 'model')
    assert hasattr(decision, 'complexity_score')
    assert hasattr(decision, 'estimated_latency_ms')
    assert hasattr(decision, 'reasoning')
    assert decision.reasoning != ""

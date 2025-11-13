"""
Query Routing Engine

Intelligently routes queries to optimal models based on complexity,
cache availability, and performance requirements.
"""

from typing import Optional, Dict
from dataclasses import dataclass
import yaml
import logging
from pathlib import Path

from backend.core.complexity_classifier import ComplexityClassifier
from backend.core.cache_manager import SemanticCacheManager

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Represents a routing decision with metadata."""
    model: str
    use_cache: bool = False
    cache_similarity: float = 0.0
    complexity_score: float = 0.0
    estimated_latency_ms: float = 0.0
    cost_relative: float = 1.0
    use_thinking_mode: bool = False
    reasoning: str = ""


class QueryRouter:
    """
    Cascade routing strategy combining model selection and escalation.
    Returns optimal model for given query characteristics.

    Process:
    1. Check semantic cache (40-60% hit rate target)
    2. Classify query complexity (0.0-1.0 scale)
    3. Estimate required context length
    4. Select model from Pareto frontier
    5. Set confidence threshold for escalation
    """

    def __init__(
        self,
        cache_manager: Optional[SemanticCacheManager] = None,
        routing_rules_path: Optional[str] = None
    ):
        """
        Initialize query router.

        Args:
            cache_manager: Semantic cache manager instance
            routing_rules_path: Path to routing rules YAML
        """
        self.cache_manager = cache_manager
        self.complexity_classifier = ComplexityClassifier()

        # Load routing rules
        if routing_rules_path is None:
            routing_rules_path = Path(__file__).parent.parent / "config" / "routing_rules.yaml"

        with open(routing_rules_path, 'r') as f:
            self.routing_rules = yaml.safe_load(f)

        self.cache_threshold = self.routing_rules["routing"]["cache_hit_threshold"]
        self.complexity_thresholds = self.routing_rules["routing"]["complexity_thresholds"]
        self.model_specs = self.routing_rules["routing"]["model_selection"]

    def route_query(
        self,
        query: str,
        context: str = "",
        device: str = "server",
        force_model: Optional[str] = None
    ) -> RouteDecision:
        """
        Route query to optimal model.

        Args:
            query: User query
            context: Optional conversation context
            device: Target device ('server', 'macbook', 'mobile')
            force_model: Force specific model (bypass routing)

        Returns:
            RouteDecision with selected model and metadata
        """
        # If model is forced, use it
        if force_model:
            return RouteDecision(
                model=force_model,
                reasoning=f"Model forced: {force_model}"
            )

        # Step 1: Check semantic cache
        if self.cache_manager:
            cache_result = self.cache_manager.check(query)
            if cache_result and cache_result.similarity > (1 - self.cache_threshold):
                logger.info(f"Cache hit! Similarity: {cache_result.similarity:.3f}")
                return RouteDecision(
                    model="cache",
                    use_cache=True,
                    cache_similarity=cache_result.similarity,
                    estimated_latency_ms=10,
                    cost_relative=0.001,
                    reasoning=f"Cache hit with similarity {cache_result.similarity:.3f}"
                )

        # Step 2: Classify query complexity
        complexity_score = self.complexity_classifier.score(query, context)
        logger.info(f"Query complexity: {complexity_score:.3f}")

        # Step 3: Estimate context length
        context_length = self.complexity_classifier.estimate_tokens(query + context)

        # Step 4: Route based on device and complexity
        decision = self._select_model(
            complexity_score=complexity_score,
            context_length=context_length,
            device=device,
            query=query
        )

        return decision

    def _select_model(
        self,
        complexity_score: float,
        context_length: int,
        device: str,
        query: str
    ) -> RouteDecision:
        """
        Select optimal model based on complexity and device.

        Args:
            complexity_score: Complexity score (0-1)
            context_length: Estimated context tokens
            device: Target device
            query: Original query

        Returns:
            RouteDecision
        """
        # Edge device routing (macbook/mobile)
        if device in ["mobile", "macbook"]:
            return self._route_edge_device(complexity_score)

        # Server device: full routing capabilities
        return self._route_server_device(complexity_score, context_length, query)

    def _route_edge_device(self, complexity_score: float) -> RouteDecision:
        """Route for edge devices (limited models)."""
        if complexity_score < self.complexity_thresholds["simple"]:
            return RouteDecision(
                model="qwen3-4b-gguf",
                complexity_score=complexity_score,
                estimated_latency_ms=100,
                cost_relative=1.0,
                reasoning="Simple query on edge device -> Qwen3-4B"
            )
        else:
            return RouteDecision(
                model="qwen3-8b-gguf",
                complexity_score=complexity_score,
                estimated_latency_ms=200,
                cost_relative=1.5,
                reasoning="Moderate/complex query on edge device -> Qwen3-8B"
            )

    def _route_server_device(
        self,
        complexity_score: float,
        context_length: int,
        query: str
    ) -> RouteDecision:
        """Route for server with full model access."""

        # Check if explicit reasoning is required
        requires_reasoning = self.complexity_classifier.requires_reasoning(query)

        if requires_reasoning:
            return RouteDecision(
                model="qwen3-32b-thinking",
                complexity_score=complexity_score,
                estimated_latency_ms=1000,
                cost_relative=4.0,
                use_thinking_mode=True,
                reasoning="Explicit reasoning required -> Qwen3-32B-Thinking"
            )

        # Route based on complexity thresholds
        if complexity_score < self.complexity_thresholds["simple"]:
            # Simple queries -> 8B model
            return RouteDecision(
                model="qwen3-8b",
                complexity_score=complexity_score,
                estimated_latency_ms=self.model_specs["qwen3_8b"]["target_latency_ms"],
                cost_relative=self.model_specs["qwen3_8b"]["cost_relative"],
                reasoning=f"Simple query (score: {complexity_score:.2f}) -> Qwen3-8B"
            )

        elif complexity_score < self.complexity_thresholds["moderate"]:
            # Moderate queries -> 14B model
            return RouteDecision(
                model="qwen3-14b",
                complexity_score=complexity_score,
                estimated_latency_ms=self.model_specs["qwen3_14b"]["target_latency_ms"],
                cost_relative=self.model_specs["qwen3_14b"]["cost_relative"],
                reasoning=f"Moderate query (score: {complexity_score:.2f}) -> Qwen3-14B"
            )

        else:
            # Complex queries -> 32B model
            return RouteDecision(
                model="qwen3-32b",
                complexity_score=complexity_score,
                estimated_latency_ms=self.model_specs["qwen3_32b"]["target_latency_ms"],
                cost_relative=self.model_specs["qwen3_32b"]["cost_relative"],
                reasoning=f"Complex query (score: {complexity_score:.2f}) -> Qwen3-32B"
            )

    def should_escalate(
        self,
        current_model: str,
        response: str,
        confidence_score: float
    ) -> Optional[str]:
        """
        Determine if response should be escalated to a larger model.

        Args:
            current_model: Currently used model
            response: Generated response
            confidence_score: Confidence score from confidence scorer

        Returns:
            Next model to try, or None if no escalation needed
        """
        escalation_config = self.routing_rules.get("escalation", {})

        if not escalation_config.get("escalate_to_larger_model", False):
            return None

        threshold = escalation_config.get("low_confidence_threshold", 0.4)

        if confidence_score < threshold:
            # Define escalation path
            escalation_path = {
                "qwen3-4b": "qwen3-8b",
                "qwen3-8b": "qwen3-14b",
                "qwen3-14b": "qwen3-32b",
                "qwen3-32b": "qwen3-32b-thinking",
            }

            next_model = escalation_path.get(current_model)
            if next_model:
                logger.info(
                    f"Escalating from {current_model} to {next_model} "
                    f"(confidence: {confidence_score:.2f})"
                )
                return next_model

        return None

    def get_model_for_task(self, task_type: str) -> str:
        """
        Get specialized model for specific tasks.

        Args:
            task_type: Type of task ('vision', 'translation', 'code', etc.)

        Returns:
            Model name
        """
        task_models = {
            "vision": "qwen3-vl",
            "translation": "qwen3-mt",
            "image": "qwen3-vl",
            "ocr": "qwen3-vl",
            "screenshot": "qwen3-vl",
        }

        return task_models.get(task_type, "qwen3-32b")

    def get_routing_stats(self) -> Dict:
        """
        Get routing statistics.

        Returns:
            Dict with routing metrics
        """
        stats = {}

        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats["cache"] = {
                "hit_rate": cache_stats.hit_rate,
                "total_cached": cache_stats.total_cached,
                "total_requests": cache_stats.total_requests
            }

        return stats

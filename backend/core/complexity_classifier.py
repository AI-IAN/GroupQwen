"""
Complexity Classification Module

Estimates query complexity (0.0-1.0) to determine which model to use.
Uses multiple signals: query length, structure, reasoning keywords, and context depth.
"""

from typing import List, Optional
import re


class ComplexityClassifier:
    """
    Estimates query complexity using multiple signals:
    - Query length and structure
    - Presence of reasoning keywords
    - Required context depth
    - Task type inference
    """

    def __init__(self):
        # Simple queries - factual, lookup-based
        self.keywords_simple = [
            "what", "how", "list", "find", "search", "show", "display",
            "get", "fetch", "retrieve", "tell me"
        ]

        # Complex queries - analytical, design-oriented
        self.keywords_complex = [
            "explain", "analyze", "design", "optimize", "debate", "compare",
            "evaluate", "assess", "propose", "recommend", "architect",
            "implement", "develop", "create", "build"
        ]

        # Reasoning-heavy queries
        self.keywords_reasoning = [
            "why", "compare", "pros cons", "implications", "consequences",
            "trade-offs", "advantages", "disadvantages", "because", "reasoning",
            "justify", "prove", "demonstrate"
        ]

        # Coding/technical keywords
        self.keywords_technical = [
            "code", "function", "class", "algorithm", "debug", "refactor",
            "api", "database", "query", "optimize", "performance"
        ]

    def score(self, query: str, context: str = "") -> float:
        """
        Compute complexity score (0.0-1.0).

        Factors:
        - Query length (longer = more complex)
        - Context length (deeper context = more complex)
        - Keyword presence (reasoning words increase complexity)
        - Estimated output length
        - Required reasoning steps

        Args:
            query: The user query to classify
            context: Optional context from previous messages

        Returns:
            Float between 0.0 and 1.0 representing complexity
        """

        # Normalized factors (0.0-1.0)
        length_factor = min(len(query) / 200, 1.0)
        context_factor = min(len(context) / 1000, 1.0) if context else 0.0

        # Keyword-based complexity
        keyword_factor = self._compute_keyword_factor(query)

        # Reasoning steps estimation
        reasoning_factor = self._estimate_reasoning_steps(query)

        # Technical complexity
        technical_factor = self._compute_technical_factor(query)

        # Aggregate with weights
        complexity = (
            0.15 * length_factor +
            0.15 * context_factor +
            0.30 * keyword_factor +
            0.25 * reasoning_factor +
            0.15 * technical_factor
        )

        return min(complexity, 1.0)

    def _compute_keyword_factor(self, query: str) -> float:
        """Compute complexity based on keyword presence."""
        query_lower = query.lower()
        factor = 0.0

        # Simple keywords reduce complexity
        simple_count = sum(1 for kw in self.keywords_simple if kw in query_lower)
        if simple_count > 0:
            factor -= 0.1

        # Complex keywords increase complexity
        complex_count = sum(1 for kw in self.keywords_complex if kw in query_lower)
        if complex_count > 0:
            factor += 0.4 * min(complex_count / 2, 1.0)

        # Reasoning keywords significantly increase complexity
        reasoning_count = sum(1 for kw in self.keywords_reasoning if kw in query_lower)
        if reasoning_count > 0:
            factor += 0.3 * min(reasoning_count / 2, 1.0)

        return max(0.0, min(factor, 1.0))

    def _estimate_reasoning_steps(self, query: str) -> float:
        """
        Estimate number of reasoning steps required.

        Heuristics:
        - Count conjunctions (and, or, but)
        - Count commas (multiple sub-questions)
        - Presence of multi-step phrases
        - Question marks (multiple questions)
        """
        reasoning_indicators = 0

        # Count conjunctions
        reasoning_indicators += query.count(" and ")
        reasoning_indicators += query.count(" or ")
        reasoning_indicators += query.count(" but ")

        # Count commas (suggests enumeration or multi-part)
        reasoning_indicators += min(query.count(","), 3)

        # Count question marks (multiple questions)
        reasoning_indicators += max(query.count("?") - 1, 0)

        # Multi-step phrases
        multi_step_phrases = [
            "step by step", "first.*then", "after that",
            "next", "finally", "in order to"
        ]
        for phrase in multi_step_phrases:
            if re.search(phrase, query.lower()):
                reasoning_indicators += 1

        # Normalize to 0-1 scale
        return min(reasoning_indicators / 5, 1.0)

    def _compute_technical_factor(self, query: str) -> float:
        """Compute technical complexity factor."""
        query_lower = query.lower()

        technical_count = sum(1 for kw in self.keywords_technical if kw in query_lower)

        # Code blocks or technical syntax
        if "```" in query or "def " in query or "class " in query:
            technical_count += 2

        # File paths or imports
        if "/" in query or "import " in query:
            technical_count += 1

        return min(technical_count / 5, 1.0)

    def classify_category(self, query: str, context: str = "") -> str:
        """
        Classify query into a category based on complexity score.

        Categories:
        - simple: 0.0 - 0.2
        - moderate: 0.2 - 0.5
        - complex: 0.5 - 0.8
        - advanced: 0.8 - 1.0

        Args:
            query: The user query
            context: Optional context

        Returns:
            Category string
        """
        score = self.score(query, context)

        if score < 0.2:
            return "simple"
        elif score < 0.5:
            return "moderate"
        elif score < 0.8:
            return "complex"
        else:
            return "advanced"

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count.
        Approximation: ~4 characters per token for English.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def requires_reasoning(self, query: str) -> bool:
        """
        Check if query explicitly requires reasoning capabilities.

        Args:
            query: User query

        Returns:
            True if reasoning is likely needed
        """
        query_lower = query.lower()

        # Explicit reasoning requests
        reasoning_phrases = [
            "think through", "reasoning", "step by step",
            "explain why", "walk me through", "show your work"
        ]

        for phrase in reasoning_phrases:
            if phrase in query_lower:
                return True

        # High complexity automatically suggests reasoning
        if self.score(query) > 0.7:
            return True

        return False

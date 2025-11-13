"""
Confidence Scoring Module

Estimates confidence of model responses to determine if escalation is needed.
"""

from typing import Dict, List, Optional
import numpy as np


class ConfidenceScorer:
    """
    Response confidence estimation using token logits and heuristics.

    Methods:
    - Token probability analysis
    - Response coherence checking
    - Hedge word detection
    - Length-based confidence
    """

    def __init__(self, low_confidence_threshold: float = 0.4):
        """
        Initialize confidence scorer.

        Args:
            low_confidence_threshold: Threshold below which to escalate
        """
        self.low_confidence_threshold = low_confidence_threshold

        # Hedge words that indicate uncertainty
        self.hedge_words = [
            "maybe", "perhaps", "possibly", "might", "could", "may",
            "uncertain", "unclear", "not sure", "don't know", "unsure",
            "i think", "probably", "seems like", "appears to"
        ]

        # Explicit uncertainty phrases
        self.uncertainty_phrases = [
            "i don't have", "i'm not sure", "i cannot", "i can't",
            "i don't know", "no information", "unable to", "insufficient",
            "as an ai", "i apologize", "i'm sorry"
        ]

    def score_response(
        self,
        response: str,
        logits: Optional[List[float]] = None,
        token_probs: Optional[List[float]] = None
    ) -> float:
        """
        Compute overall confidence score for a response.

        Args:
            response: Generated response text
            logits: Optional token logits from the model
            token_probs: Optional token probabilities

        Returns:
            Confidence score (0.0-1.0)
        """
        scores = []

        # 1. Token probability-based confidence (if available)
        if token_probs is not None:
            prob_score = self._score_from_token_probs(token_probs)
            scores.append((prob_score, 0.4))  # Weight: 0.4

        # 2. Linguistic confidence (hedge words, uncertainty)
        linguistic_score = self._score_linguistic_confidence(response)
        scores.append((linguistic_score, 0.3))  # Weight: 0.3

        # 3. Response completeness
        completeness_score = self._score_completeness(response)
        scores.append((completeness_score, 0.2))  # Weight: 0.2

        # 4. Response length (too short = low confidence)
        length_score = self._score_response_length(response)
        scores.append((length_score, 0.1))  # Weight: 0.1

        # Weighted average
        if not scores:
            return 0.5  # Default neutral confidence

        weighted_sum = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)

        return min(weighted_sum / total_weight, 1.0)

    def _score_from_token_probs(self, token_probs: List[float]) -> float:
        """
        Score confidence based on token probabilities.

        High average probability = high confidence
        Low variance = high confidence

        Args:
            token_probs: List of token probabilities

        Returns:
            Confidence score
        """
        if not token_probs:
            return 0.5

        probs_array = np.array(token_probs)

        # Average probability (higher = more confident)
        avg_prob = np.mean(probs_array)

        # Variance (lower = more confident, model is more certain)
        variance = np.var(probs_array)

        # Combine: high average, low variance = high confidence
        confidence = avg_prob * (1 - min(variance, 0.5))

        return float(confidence)

    def _score_linguistic_confidence(self, response: str) -> float:
        """
        Score based on linguistic markers of uncertainty.

        Args:
            response: Response text

        Returns:
            Confidence score
        """
        response_lower = response.lower()
        confidence = 1.0

        # Check for hedge words
        hedge_count = sum(1 for word in self.hedge_words if word in response_lower)
        confidence -= hedge_count * 0.05  # Each hedge reduces confidence by 5%

        # Check for explicit uncertainty phrases (severe penalty)
        uncertainty_count = sum(
            1 for phrase in self.uncertainty_phrases
            if phrase in response_lower
        )
        confidence -= uncertainty_count * 0.3  # Each phrase reduces by 30%

        return max(confidence, 0.0)

    def _score_completeness(self, response: str) -> float:
        """
        Score based on response completeness.

        Incomplete responses often indicate low confidence or truncation.

        Args:
            response: Response text

        Returns:
            Confidence score
        """
        # Check if response ends properly
        proper_endings = [".", "!", "?", ":", "```"]
        ends_properly = any(response.rstrip().endswith(end) for end in proper_endings)

        if not ends_properly:
            return 0.5  # Likely truncated or incomplete

        # Check for very short responses (< 20 chars)
        if len(response) < 20:
            return 0.3

        return 1.0

    def _score_response_length(self, response: str) -> float:
        """
        Score based on response length.

        Very short responses often indicate low confidence or inability to answer.

        Args:
            response: Response text

        Returns:
            Confidence score
        """
        length = len(response)

        if length < 10:
            return 0.2
        elif length < 50:
            return 0.6
        elif length < 100:
            return 0.8
        else:
            return 1.0

    def should_escalate(self, confidence: float) -> bool:
        """
        Determine if response should be escalated to a larger model.

        Args:
            confidence: Confidence score

        Returns:
            True if should escalate
        """
        return confidence < self.low_confidence_threshold

    def get_escalation_reason(self, response: str, confidence: float) -> str:
        """
        Get human-readable reason for escalation.

        Args:
            response: Response text
            confidence: Confidence score

        Returns:
            Reason string
        """
        response_lower = response.lower()

        if any(phrase in response_lower for phrase in self.uncertainty_phrases):
            return "Response contains explicit uncertainty"

        if len(response) < 20:
            return "Response too short"

        if confidence < self.low_confidence_threshold:
            return f"Low confidence score: {confidence:.2f}"

        return "No escalation needed"

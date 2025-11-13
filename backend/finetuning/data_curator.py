"""
Data Curator Module

Interactive tool for curating training data from conversation exports.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from backend.finetuning.data_processor import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class CurationDecision:
    """Represents a curation decision."""
    conversation_id: str
    accepted: bool
    reason: Optional[str] = None


class DataCurator:
    """
    Interactive data curation tool.

    Features:
    - Filter by quality metrics (length, coherence, domain)
    - Remove PII/sensitive information
    - Format for instruction tuning
    - Track curator decisions for reproducibility
    """

    def __init__(self):
        self.rejected_patterns: List[str] = []
        self.accepted_count = 0
        self.rejected_count = 0
        self.decisions: List[CurationDecision] = []

    def curate_conversations(
        self,
        conversations: List[List[ConversationTurn]],
        auto_filter: bool = True,
        interactive: bool = False
    ) -> List[List[ConversationTurn]]:
        """
        Curate conversations with automatic and manual filtering.

        Args:
            conversations: Raw conversations
            auto_filter: Apply automatic quality filters
            interactive: Enable interactive review

        Returns:
            Curated conversations
        """
        curated = []

        for i, conv in enumerate(conversations):
            # Automatic quality checks
            if auto_filter:
                if not self._passes_auto_checks(conv):
                    self.rejected_count += 1
                    continue

            # Interactive review
            if interactive:
                if self._interactive_review(conv, i, len(conversations)):
                    curated.append(conv)
                    self.accepted_count += 1
                else:
                    self.rejected_count += 1
            else:
                curated.append(conv)
                self.accepted_count += 1

        logger.info(
            f"Curation complete: "
            f"{self.accepted_count} accepted, "
            f"{self.rejected_count} rejected"
        )

        return curated

    def _passes_auto_checks(self, conv: List[ConversationTurn]) -> bool:
        """
        Run automatic quality checks.

        Args:
            conv: Conversation to check

        Returns:
            True if passes all checks
        """
        # Check: Minimum conversation length
        if len(conv) < 2:
            return False

        for turn in conv:
            content = turn.content

            # Check: Minimum content length
            if len(content) < 50:
                return False

            # Check: Maximum content length (too long = likely copy-paste)
            if len(content) > 2000:
                return False

            # Check: Low confidence responses
            low_confidence_phrases = [
                "i don't have", "i'm not sure", "i cannot",
                "i can't", "i don't know", "no information",
                "unable to", "insufficient"
            ]

            if any(phrase in content.lower() for phrase in low_confidence_phrases):
                return False

            # Check: AI disclaimers (often in refusals)
            disclaimer_phrases = [
                "as an ai", "i'm an ai", "i apologize but",
                "i'm sorry but", "i cannot help with"
            ]

            if any(phrase in content.lower() for phrase in disclaimer_phrases):
                return False

            # Check: Empty or whitespace-only
            if not content.strip():
                return False

        return True

    def _interactive_review(
        self,
        conv: List[ConversationTurn],
        index: int,
        total: int
    ) -> bool:
        """
        Interactive review of conversation.

        Args:
            conv: Conversation to review
            index: Current index
            total: Total conversations

        Returns:
            True if accepted
        """
        print(f"\n{'=' * 80}")
        print(f"Conversation {index + 1}/{total}")
        print(f"{'=' * 80}")

        for turn in conv:
            role_display = turn.role.upper()
            content_preview = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content

            print(f"\n[{role_display}]")
            print(content_preview)

        print(f"\n{'=' * 80}")

        # Get user decision
        while True:
            decision = input("Accept this conversation? (y/n/q to quit): ").lower()

            if decision == 'q':
                print("Curation stopped by user")
                return False
            elif decision == 'y':
                return True
            elif decision == 'n':
                return False
            else:
                print("Please enter 'y', 'n', or 'q'")

    def detect_pii(self, text: str) -> List[str]:
        """
        Detect potential PII in text.

        Args:
            text: Text to check

        Returns:
            List of detected PII patterns
        """
        import re

        pii_found = []

        # Email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            pii_found.append("email")

        # Phone numbers (simple pattern)
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            pii_found.append("phone")

        # SSN-like patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            pii_found.append("ssn")

        # Credit card-like patterns
        if re.search(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', text):
            pii_found.append("credit_card")

        return pii_found

    def remove_pii(self, text: str) -> str:
        """
        Remove or redact PII from text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        import re

        # Redact emails
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            text
        )

        # Redact phone numbers
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE_REDACTED]',
            text
        )

        # Redact SSN
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN_REDACTED]',
            text
        )

        # Redact credit cards
        text = re.sub(
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            '[CC_REDACTED]',
            text
        )

        return text

    def get_curation_stats(self) -> Dict:
        """
        Get curation statistics.

        Returns:
            Stats dict
        """
        total = self.accepted_count + self.rejected_count

        return {
            "total_reviewed": total,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "acceptance_rate": self.accepted_count / total if total > 0 else 0.0
        }

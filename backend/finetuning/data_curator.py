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

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.conversations: List[Dict] = []
        self.rejected_patterns: List[str] = []
        self.accepted_count = 0
        self.rejected_count = 0
        self.decisions: List[CurationDecision] = []
        self.pii_removed_count = 0

    def load(self) -> int:
        """Load conversations from file. Returns count."""
        if not self.dataset_path:
            logger.error("No dataset path specified")
            return 0

        try:
            import json
            from pathlib import Path

            path = Path(self.dataset_path)
            if not path.exists():
                logger.error(f"Dataset file not found: {self.dataset_path}")
                return 0

            # Load JSON or JSONL file
            self.conversations = []

            if path.suffix == '.jsonl':
                # JSONL format
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.conversations.append(data)
            elif path.suffix == '.json':
                # JSON format
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.conversations = data
                    else:
                        self.conversations = [data]
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return 0

            logger.info(f"Loaded {len(self.conversations)} conversations from {self.dataset_path}")
            return len(self.conversations)

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return 0

    def filter_quality(self, min_length: int = 10, max_length: int = 10000) -> int:
        """
        Filter low-quality conversations.

        Removes:
        - Too short (< min_length chars)
        - Too long (> max_length chars)
        - All caps messages
        - Repetitive content
        - Single-turn conversations

        Returns: Number of conversations removed
        """
        initial_count = len(self.conversations)
        filtered = []

        for conv in self.conversations:
            # Get messages
            messages = conv.get("messages", [])

            # Check: Minimum conversation length (must have at least 2 messages)
            if len(messages) < 2:
                continue

            # Check each message
            valid = True
            for msg in messages:
                content = msg.get("content", "")

                # Check: Minimum content length
                if len(content) < min_length:
                    valid = False
                    break

                # Check: Maximum content length
                if len(content) > max_length:
                    valid = False
                    break

                # Check: All caps (more than 50% uppercase)
                if content.isupper() and len(content) > 10:
                    valid = False
                    break

                # Check: Repetitive content (same character repeated)
                if len(set(content.replace(' ', ''))) < 5 and len(content) > 10:
                    valid = False
                    break

            if valid:
                filtered.append(conv)

        removed_count = initial_count - len(filtered)
        self.conversations = filtered

        logger.info(f"Quality filter: removed {removed_count} conversations ({len(filtered)} remaining)")
        return removed_count

    def remove_pii(self) -> int:
        """
        Detect and remove PII (Personally Identifiable Information).

        Detects:
        - Email addresses: name@domain.com
        - Phone numbers: (123) 456-7890, 123-456-7890
        - SSNs: 123-45-6789
        - Physical addresses (best effort)
        - Credit card numbers

        Replaces with [REDACTED]

        Returns: Number of PII instances removed
        """
        import re

        pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }

        total_pii_removed = 0

        for conv in self.conversations:
            messages = conv.get("messages", [])

            for msg in messages:
                content = msg.get("content", "")
                original_content = content

                # Apply each PII pattern
                for pii_type, pattern in pii_patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        total_pii_removed += len(matches)
                        content = re.sub(pattern, '[REDACTED]', content)

                # Update message content if changed
                if content != original_content:
                    msg["content"] = content

        self.pii_removed_count = total_pii_removed
        logger.info(f"PII removal: redacted {total_pii_removed} instances")
        return total_pii_removed

    def export(self, output_path: str, format: str = 'jsonl') -> bool:
        """Export curated dataset to file."""
        try:
            import json
            from pathlib import Path

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == 'jsonl':
                # JSONL format (one JSON object per line)
                with open(output_path, 'w') as f:
                    for conv in self.conversations:
                        f.write(json.dumps(conv) + '\n')

            elif format == 'json':
                # JSON format (single array)
                with open(output_path, 'w') as f:
                    json.dump(self.conversations, f, indent=2)

            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported {len(self.conversations)} conversations to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return False

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

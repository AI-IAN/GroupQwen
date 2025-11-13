"""
Data Processor Module

Handles dataset preparation and formatting for fine-tuning.
"""

from typing import List, Dict, Optional
import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[str] = None


class DataProcessor:
    """
    Processes raw conversation data into instruction-tuning format.

    Supports:
    - ChatGPT export format
    - Claude export format
    - Perplexity export format
    - Custom JSON formats
    """

    def __init__(self):
        self.conversations: List[List[ConversationTurn]] = []

    def load_chatgpt_export(self, zip_path: str) -> List[List[ConversationTurn]]:
        """
        Load conversations from ChatGPT export ZIP.

        Args:
            zip_path: Path to ChatGPT export ZIP

        Returns:
            List of conversations
        """
        import zipfile

        logger.info(f"Loading ChatGPT export: {zip_path}")

        conversations = []

        try:
            with zipfile.ZipFile(zip_path) as z:
                with z.open("conversations.json") as f:
                    data = json.load(f)

            for conv in data:
                conv_turns = []

                # Parse conversation tree structure
                for node_id, node in conv.get("mapping", {}).items():
                    message = node.get("message")
                    if not message or not message.get("content", {}).get("parts"):
                        continue

                    role = message["author"]["role"]
                    content = message["content"]["parts"][0]
                    timestamp = message.get("create_time")

                    # Map roles
                    if role == "system":
                        role = "system"
                    elif role in ["user", "human"]:
                        role = "user"
                    elif role in ["assistant", "ai"]:
                        role = "assistant"

                    conv_turns.append(
                        ConversationTurn(
                            role=role,
                            content=content,
                            timestamp=timestamp
                        )
                    )

                if conv_turns:
                    conversations.append(conv_turns)

            logger.info(f"Loaded {len(conversations)} conversations from ChatGPT export")

        except Exception as e:
            logger.error(f"Error loading ChatGPT export: {e}")

        return conversations

    def load_claude_export(self, json_path: str) -> List[List[ConversationTurn]]:
        """
        Load conversations from Claude export JSON.

        Args:
            json_path: Path to Claude export JSON

        Returns:
            List of conversations
        """
        logger.info(f"Loading Claude export: {json_path}")

        conversations = []

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            for conv in data:
                conv_turns = []

                for message in conv.get("messages", []):
                    role = message.get("role", "user")
                    content = message.get("content", "")

                    conv_turns.append(
                        ConversationTurn(
                            role=role,
                            content=content
                        )
                    )

                if conv_turns:
                    conversations.append(conv_turns)

            logger.info(f"Loaded {len(conversations)} conversations from Claude export")

        except Exception as e:
            logger.error(f"Error loading Claude export: {e}")

        return conversations

    def format_for_training(
        self,
        conversations: List[List[ConversationTurn]],
        format_type: str = "instruction"
    ) -> List[Dict]:
        """
        Format conversations for training.

        Args:
            conversations: List of conversation turns
            format_type: Format type ('instruction', 'chat', 'completion')

        Returns:
            Formatted training examples
        """
        training_data = []

        for conv in conversations:
            if format_type == "instruction":
                # Instruction-tuning format
                for i in range(len(conv) - 1):
                    if conv[i].role == "user" and conv[i + 1].role == "assistant":
                        training_data.append({
                            "instruction": conv[i].content,
                            "input": "",
                            "output": conv[i + 1].content
                        })

            elif format_type == "chat":
                # Chat format (messages array)
                messages = [
                    {"role": turn.role, "content": turn.content}
                    for turn in conv
                ]
                training_data.append({"messages": messages})

            elif format_type == "completion":
                # Simple completion format
                prompt = ""
                completion = ""

                for turn in conv:
                    if turn.role == "user":
                        prompt += f"User: {turn.content}\n"
                    elif turn.role == "assistant":
                        completion = turn.content
                        break

                if prompt and completion:
                    training_data.append({
                        "prompt": prompt.strip(),
                        "completion": completion
                    })

        logger.info(f"Formatted {len(training_data)} training examples")

        return training_data

    def save_training_data(self, data: List[Dict], output_path: str):
        """
        Save training data to JSONL file.

        Args:
            data: Training data
            output_path: Output file path
        """
        logger.info(f"Saving training data to: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Saved {len(data)} examples")

    def filter_by_quality(
        self,
        conversations: List[List[ConversationTurn]],
        min_length: int = 50,
        max_length: int = 2000
    ) -> List[List[ConversationTurn]]:
        """
        Filter conversations by quality metrics.

        Args:
            conversations: Input conversations
            min_length: Minimum content length
            max_length: Maximum content length

        Returns:
            Filtered conversations
        """
        filtered = []

        for conv in conversations:
            # Skip very short conversations
            if len(conv) < 2:
                continue

            # Check length constraints
            valid = True
            for turn in conv:
                content_len = len(turn.content)
                if content_len < min_length or content_len > max_length:
                    valid = False
                    break

                # Skip low-confidence responses
                if any(phrase in turn.content.lower() for phrase in [
                    "i don't have", "i'm not sure", "i cannot",
                    "as an ai", "i apologize"
                ]):
                    valid = False
                    break

            if valid:
                filtered.append(conv)

        logger.info(
            f"Filtered conversations: {len(conversations)} -> {len(filtered)}"
        )

        return filtered

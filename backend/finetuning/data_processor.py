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

    @staticmethod
    def convert_to_training_format(conversations: List[Dict]) -> str:
        """
        Convert conversations to JSONL training format.

        Input: List of conversations, each with messages
        Output: JSONL string, one conversation per line

        Format:
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]}
        """
        jsonl_lines = []

        for conv in conversations:
            # Handle different conversation formats
            if "messages" in conv:
                # Already in messages format
                messages = conv["messages"]
            elif isinstance(conv, list):
                # List of ConversationTurn objects or dicts
                messages = []
                for turn in conv:
                    if hasattr(turn, 'role') and hasattr(turn, 'content'):
                        # ConversationTurn object
                        messages.append({
                            "role": turn.role,
                            "content": turn.content
                        })
                    elif isinstance(turn, dict) and "role" in turn and "content" in turn:
                        # Already a dict
                        messages.append({
                            "role": turn["role"],
                            "content": turn["content"]
                        })
            else:
                logger.warning(f"Skipping conversation with unknown format: {type(conv)}")
                continue

            # Create JSONL entry
            if messages:
                jsonl_lines.append(json.dumps({"messages": messages}))

        return "\n".join(jsonl_lines)

    @staticmethod
    def validate_dataset(dataset_path: str) -> bool:
        """
        Validate JSONL dataset format.

        Checks:
        - File exists and is valid JSON lines
        - Each line has "messages" key
        - Messages have "role" and "content"
        - Roles are valid (user/assistant/system)
        """
        try:
            if not Path(dataset_path).exists():
                logger.error(f"Dataset file not found: {dataset_path}")
                return False

            valid_roles = {"user", "assistant", "system"}
            line_count = 0

            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse JSON
                        data = json.loads(line)
                        line_count += 1

                        # Check for messages key
                        if "messages" not in data:
                            logger.error(f"Line {line_num}: Missing 'messages' key")
                            return False

                        messages = data["messages"]
                        if not isinstance(messages, list):
                            logger.error(f"Line {line_num}: 'messages' must be a list")
                            return False

                        # Validate each message
                        for msg_idx, msg in enumerate(messages):
                            if not isinstance(msg, dict):
                                logger.error(f"Line {line_num}, message {msg_idx}: Must be a dict")
                                return False

                            # Check required keys
                            if "role" not in msg:
                                logger.error(f"Line {line_num}, message {msg_idx}: Missing 'role'")
                                return False
                            if "content" not in msg:
                                logger.error(f"Line {line_num}, message {msg_idx}: Missing 'content'")
                                return False

                            # Validate role
                            if msg["role"] not in valid_roles:
                                logger.error(
                                    f"Line {line_num}, message {msg_idx}: "
                                    f"Invalid role '{msg['role']}'. Must be one of {valid_roles}"
                                )
                                return False

                            # Validate content is string
                            if not isinstance(msg["content"], str):
                                logger.error(
                                    f"Line {line_num}, message {msg_idx}: "
                                    f"Content must be a string"
                                )
                                return False

                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: Invalid JSON - {e}")
                        return False

            logger.info(f"Dataset validation successful: {line_count} conversations validated")
            return True

        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False

    @staticmethod
    def load_from_various_formats(file_path: str) -> List[Dict]:
        """
        Load conversations from ChatGPT, Claude, or plain text format.

        Supports:
        - ChatGPT: conversations.json from ZIP
        - Claude: JSON export
        - Plain text: simple Q&A format
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        try:
            # Detect format by extension and content
            if file_path.suffix == '.zip':
                # ChatGPT ZIP export
                return DataProcessor._load_chatgpt_zip(str(file_path))
            elif file_path.suffix == '.json':
                # Could be ChatGPT or Claude JSON
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Try to detect format
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if "mapping" in first_item:
                        # ChatGPT format
                        return DataProcessor._parse_chatgpt_conversations(data)
                    elif "messages" in first_item:
                        # Claude or pre-formatted format
                        return data
                elif isinstance(data, dict) and "mapping" in data:
                    # Single ChatGPT conversation
                    return DataProcessor._parse_chatgpt_conversations([data])
            elif file_path.suffix == '.txt':
                # Plain text Q&A format
                return DataProcessor._load_plain_text(str(file_path))
            else:
                logger.warning(f"Unknown file format: {file_path.suffix}")
                return []

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    @staticmethod
    def _load_chatgpt_zip(zip_path: str) -> List[Dict]:
        """Load from ChatGPT ZIP export."""
        import zipfile

        try:
            with zipfile.ZipFile(zip_path) as z:
                # Find conversations.json
                conv_file = None
                for name in z.namelist():
                    if name.endswith('conversations.json'):
                        conv_file = name
                        break

                if not conv_file:
                    logger.error("conversations.json not found in ZIP")
                    return []

                with z.open(conv_file) as f:
                    data = json.load(f)

                return DataProcessor._parse_chatgpt_conversations(data)

        except Exception as e:
            logger.error(f"Error loading ChatGPT ZIP: {e}")
            return []

    @staticmethod
    def _parse_chatgpt_conversations(data: List[Dict]) -> List[Dict]:
        """Parse ChatGPT conversation format to standard format."""
        conversations = []

        for conv in data:
            messages = []

            # Parse conversation tree structure
            mapping = conv.get("mapping", {})

            # Build conversation flow (simplified - takes first path)
            for node_id, node in mapping.items():
                message = node.get("message")
                if not message:
                    continue

                content_parts = message.get("content", {}).get("parts", [])
                if not content_parts or not content_parts[0]:
                    continue

                role = message.get("author", {}).get("role", "user")
                content = content_parts[0]

                # Map roles
                if role in ["system"]:
                    role = "system"
                elif role in ["user", "human"]:
                    role = "user"
                elif role in ["assistant", "ai"]:
                    role = "assistant"
                else:
                    continue

                messages.append({
                    "role": role,
                    "content": content
                })

            if messages:
                conversations.append({"messages": messages})

        return conversations

    @staticmethod
    def _load_plain_text(file_path: str) -> List[Dict]:
        """Load plain text Q&A format."""
        conversations = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Simple parser for Q: ... A: ... format
            import re

            # Split by Q: markers
            qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)'
            matches = re.findall(qa_pattern, content, re.DOTALL)

            for question, answer in matches:
                question = question.strip()
                answer = answer.strip()

                if question and answer:
                    conversations.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    })

        except Exception as e:
            logger.error(f"Error loading plain text: {e}")

        return conversations

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

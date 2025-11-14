#!/usr/bin/env python3
"""
Export chat history from various sources to normalized format.

Usage:
    python scripts/export_chat_history.py --source chatgpt --input chatgpt_export.zip --output data/exported.json
    python scripts/export_chat_history.py --source claude --input claude_export.json --output data/exported.json
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Dict
import sys
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.finetuning.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_chatgpt(zip_path: str) -> List[Dict]:
    """
    Export from ChatGPT ZIP format.

    ChatGPT exports contain conversations.json in ZIP.
    Format: {"id": "...", "mapping": {...}, ...}
    """
    logger.info(f"Exporting from ChatGPT ZIP: {zip_path}")

    try:
        conversations = []

        with zipfile.ZipFile(zip_path) as z:
            # Find conversations.json
            conv_file = None
            for name in z.namelist():
                if name.endswith('conversations.json'):
                    conv_file = name
                    break

            if not conv_file:
                logger.error("conversations.json not found in ZIP archive")
                return []

            with z.open(conv_file) as f:
                data = json.load(f)

        # Parse ChatGPT format
        for conv_data in data:
            messages = []

            # Parse conversation tree structure
            mapping = conv_data.get("mapping", {})

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

                # Map roles to standard format
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

        logger.info(f"Exported {len(conversations)} conversations from ChatGPT")
        return conversations

    except Exception as e:
        logger.error(f"Error exporting ChatGPT data: {e}", exc_info=True)
        return []


def export_claude(json_path: str) -> List[Dict]:
    """Export from Claude JSON format."""
    logger.info(f"Exporting from Claude JSON: {json_path}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        conversations = []

        # Claude export format varies, try to handle different formats
        if isinstance(data, list):
            # List of conversations
            for conv in data:
                if "messages" in conv:
                    # Already in correct format
                    conversations.append(conv)
                elif "conversation" in conv:
                    # Nested conversation
                    messages = []
                    for msg in conv["conversation"]:
                        role = msg.get("role", "user")
                        content = msg.get("content", "") or msg.get("text", "")
                        messages.append({"role": role, "content": content})
                    if messages:
                        conversations.append({"messages": messages})
        elif isinstance(data, dict):
            # Single conversation or wrapped format
            if "messages" in data:
                conversations.append(data)
            elif "conversations" in data:
                conversations = data["conversations"]

        logger.info(f"Exported {len(conversations)} conversations from Claude")
        return conversations

    except Exception as e:
        logger.error(f"Error exporting Claude data: {e}", exc_info=True)
        return []


def export_perplexity(json_path: str) -> List[Dict]:
    """
    Export from Perplexity format (if available).

    Note: Perplexity export format is not well-documented.
    This is a best-effort implementation.
    """
    logger.info(f"Exporting from Perplexity JSON: {json_path}")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        conversations = []

        # Try to parse Perplexity format (hypothetical)
        if isinstance(data, list):
            for item in data:
                if "query" in item and "answer" in item:
                    # Query-answer format
                    messages = [
                        {"role": "user", "content": item["query"]},
                        {"role": "assistant", "content": item["answer"]}
                    ]
                    conversations.append({"messages": messages})

        logger.info(f"Exported {len(conversations)} conversations from Perplexity")
        return conversations

    except Exception as e:
        logger.error(f"Error exporting Perplexity data: {e}", exc_info=True)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Export chat history from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from ChatGPT ZIP
  python scripts/export_chat_history.py --source chatgpt --input chatgpt_export.zip --output data/exported.json

  # Export from Claude JSON
  python scripts/export_chat_history.py --source claude --input claude_export.json --output data/exported.json

  # Export from Perplexity
  python scripts/export_chat_history.py --source perplexity --input perplexity_export.json --output data/exported.json
        """
    )

    parser.add_argument(
        '--source',
        choices=['chatgpt', 'claude', 'perplexity'],
        required=True,
        help='Source platform for chat export'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input file path (ZIP for ChatGPT, JSON for others)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Call appropriate export function
    conversations = []

    if args.source == 'chatgpt':
        conversations = export_chatgpt(args.input)
    elif args.source == 'claude':
        conversations = export_claude(args.input)
    elif args.source == 'perplexity':
        conversations = export_perplexity(args.input)

    if not conversations:
        logger.error("No conversations exported. Check input file format.")
        sys.exit(1)

    # Save normalized conversations to output
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)

        logger.info(f"Successfully exported {len(conversations)} conversations to {args.output}")

        # Print summary
        total_messages = sum(len(conv.get("messages", [])) for conv in conversations)
        avg_messages = total_messages / len(conversations) if conversations else 0

        print("\n" + "=" * 60)
        print("Export Summary")
        print("=" * 60)
        print(f"Source:             {args.source}")
        print(f"Input:              {args.input}")
        print(f"Output:             {args.output}")
        print(f"Conversations:      {len(conversations)}")
        print(f"Total messages:     {total_messages}")
        print(f"Avg messages/conv:  {avg_messages:.1f}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error saving output: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

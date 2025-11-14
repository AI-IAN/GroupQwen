#!/usr/bin/env python3
"""
Interactive dataset curation tool.

Usage:
    # Interactive mode
    python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --interactive

    # Auto mode (apply filters automatically)
    python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --auto
"""

import argparse
from pathlib import Path
import sys
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.finetuning.data_curator import DataCurator

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def interactive_curation(curator: DataCurator):
    """Show each conversation and prompt for keep/skip/edit."""
    print("\n" + "=" * 80)
    print("Interactive Curation Mode")
    print("=" * 80)
    print("Instructions:")
    print("  [k] Keep conversation")
    print("  [s] Skip conversation")
    print("  [e] Edit conversation (advanced)")
    print("  [q] Quit and save progress")
    print("=" * 80 + "\n")

    if not curator.conversations:
        print("No conversations loaded.")
        return

    curated = []
    total = len(curator.conversations)

    for idx, conv in enumerate(curator.conversations):
        print(f"\n{'=' * 80}")
        print(f"Conversation {idx + 1}/{total}")
        print(f"{'=' * 80}")

        messages = conv.get("messages", [])

        # Display conversation
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Truncate long messages for display
            if len(content) > 300:
                content_display = content[:300] + "..."
            else:
                content_display = content

            print(f"\n[{role}]")
            print(content_display)

        print(f"\n{'=' * 80}")

        # Get user decision
        while True:
            decision = input(f"Action? [k]eep, [s]kip, [e]dit, [q]uit [{idx + 1}/{total}]: ").lower().strip()

            if decision == 'k':
                # Keep conversation
                curated.append(conv)
                curator.accepted_count += 1
                print("✓ Kept")
                break
            elif decision == 's':
                # Skip conversation
                curator.rejected_count += 1
                print("✗ Skipped")
                break
            elif decision == 'e':
                # Edit conversation (simplified - just show option)
                print("Edit mode not yet implemented. Use [k] or [s].")
                continue
            elif decision == 'q':
                # Quit and save progress
                print(f"\nQuitting. Curated {len(curated)} conversations so far.")
                curator.conversations = curated
                return
            else:
                print("Invalid input. Use 'k', 's', 'e', or 'q'.")

    # Update curator conversations with curated list
    curator.conversations = curated
    print(f"\n✓ Curation complete: {len(curated)} conversations kept, {curator.rejected_count} skipped")


def auto_curation(curator: DataCurator, min_length: int = 10, max_length: int = 10000):
    """Apply filters automatically."""
    print("\n" + "=" * 80)
    print("Automatic Curation Mode")
    print("=" * 80)

    initial_count = len(curator.conversations)
    print(f"Initial conversations: {initial_count}")

    # Apply quality filters
    print("\nApplying quality filters...")
    removed_quality = curator.filter_quality(min_length=min_length, max_length=max_length)
    print(f"  - Removed {removed_quality} low-quality conversations")
    print(f"  - Remaining: {len(curator.conversations)}")

    # Apply PII removal
    print("\nRemoving PII...")
    pii_removed = curator.remove_pii()
    print(f"  - Redacted {pii_removed} PII instances")

    print("\n" + "=" * 80)
    print("Auto-curation Statistics")
    print("=" * 80)
    print(f"Initial conversations:     {initial_count}")
    print(f"After quality filtering:   {len(curator.conversations)}")
    print(f"PII instances removed:     {pii_removed}")
    print(f"Final conversations:       {len(curator.conversations)}")
    print(f"Retention rate:            {len(curator.conversations) / initial_count * 100:.1f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Curate dataset for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (review each conversation)
  python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --interactive

  # Auto mode (apply filters automatically)
  python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --auto

  # Auto mode with custom length thresholds
  python scripts/curate_dataset.py --input data/exported.json --output data/curated.jsonl --auto --min-length 20 --max-length 5000
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input file (JSON/JSONL)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive curation mode'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatic curation mode'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum message length (default: 10)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=10000,
        help='Maximum message length (default: 10000)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.auto:
        parser.error("Must specify either --interactive or --auto mode")

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    curator = DataCurator(args.input)
    count = curator.load()

    if count == 0:
        logger.error("No conversations loaded. Check input file format.")
        sys.exit(1)

    print(f"Loaded {count} conversations")

    # Run curation
    if args.interactive:
        interactive_curation(curator)
    elif args.auto:
        auto_curation(curator, min_length=args.min_length, max_length=args.max_length)

    # Export curated dataset
    if len(curator.conversations) == 0:
        print("\nWarning: No conversations remaining after curation!")
        sys.exit(0)

    print(f"\nExporting curated dataset to {args.output}...")
    success = curator.export(args.output, format='jsonl')

    if success:
        print(f"✓ Successfully exported {len(curator.conversations)} curated conversations")
        print(f"  Output: {args.output}")

        # Print final summary
        print("\n" + "=" * 80)
        print("Final Summary")
        print("=" * 80)
        print(f"Input file:           {args.input}")
        print(f"Output file:          {args.output}")
        print(f"Conversations kept:   {len(curator.conversations)}")
        print(f"Mode:                 {'Interactive' if args.interactive else 'Automatic'}")
        print("=" * 80)
    else:
        logger.error("Failed to export curated dataset")
        sys.exit(1)


if __name__ == '__main__':
    main()

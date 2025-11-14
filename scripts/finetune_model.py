#!/usr/bin/env python3
"""
Fine-tune Qwen3 models on custom datasets.

Usage:
    python scripts/finetune_model.py --model qwen3-8b --dataset data/curated.jsonl --epochs 3
"""

import argparse
import signal
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.finetuning.trainer import Trainer, TrainingConfig
from backend.finetuning.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global trainer for signal handling
trainer = None
checkpoint_manager = None
interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C by saving checkpoint."""
    global interrupted

    if interrupted:
        # Second Ctrl+C, force exit
        logger.warning("\nForce exiting...")
        sys.exit(1)

    interrupted = True
    print("\n\n=== Training Interrupted ===")
    print("Saving checkpoint before exit...")

    if trainer and checkpoint_manager and trainer.model is not None:
        try:
            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=trainer.model,
                tokenizer=trainer.tokenizer,
                epoch=trainer.current_epoch,
                step=trainer.current_step,
                loss=trainer.current_loss,
                metadata={
                    'run_id': trainer.run_id,
                    'interrupted': True,
                    'learning_rate': trainer.config.learning_rate
                }
            )
            print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
            print("\nYou can resume training with:")
            print(f"  python scripts/finetune_model.py --resume {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    print("\nExiting...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 models with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune qwen3-8b for 3 epochs
  python scripts/finetune_model.py --model qwen3-8b --dataset data/curated.jsonl --epochs 3

  # Custom batch size and learning rate
  python scripts/finetune_model.py --model qwen3-14b --dataset data/curated.jsonl \\
      --epochs 5 --batch-size 8 --learning-rate 1e-4

  # Resume from checkpoint
  python scripts/finetune_model.py --resume ./checkpoints/checkpoint-epoch1-step100
        """
    )

    parser.add_argument(
        '--model',
        help='Base model: qwen3-8b, qwen3-14b, qwen3-32b'
    )
    parser.add_argument(
        '--dataset',
        help='Path to JSONL dataset (from Agent 2A)'
    )
    parser.add_argument(
        '--output',
        default='./models/finetuned',
        help='Output directory (default: ./models/finetuned)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=16,
        help='LoRA rank (default: 16)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=100,
        help='Save checkpoint every N steps (default: 100)'
    )
    parser.add_argument(
        '--resume',
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='./checkpoints',
        help='Checkpoint directory (default: ./checkpoints)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.resume:
        if not args.model:
            parser.error("--model is required (unless using --resume)")
        if not args.dataset:
            parser.error("--dataset is required (unless using --resume)")

    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("GroupQwen Fine-Tuning Pipeline")
    print("=" * 70)

    # Create training config
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.resume:
        # TODO: Load config from checkpoint
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        print("   (Resume functionality: load model, optimizer, and training state)")
        logger.warning("Resume from checkpoint not fully implemented yet")
        # For now, just exit
        return

    config = TrainingConfig(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps
    )

    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Model:         {args.model}")
    print(f"   Dataset:       {args.dataset}")
    print(f"   Output:        {args.output}")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch Size:    {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   LoRA Rank:     {args.lora_rank}")
    print(f"   LoRA Alpha:    {args.lora_alpha}")
    print(f"   Save Steps:    {args.save_steps}")

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        print(f"\n✗ Error: Dataset not found: {args.dataset}")
        print("\nPlease run data curation first:")
        print("  python -m backend.finetuning.data_curator")
        sys.exit(1)

    # Initialize checkpoint manager
    global checkpoint_manager, trainer
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_best_n=3
    )

    # Initialize trainer
    trainer = Trainer(config, checkpoint_manager)
    trainer.run_id = run_id

    # Start training
    print(f"\n🚀 Starting fine-tuning (Run ID: {run_id})")
    print("   Press Ctrl+C to save checkpoint and exit\n")

    result = trainer.train()

    # Print results
    print("\n" + "=" * 70)
    if result.success:
        print("✓ Training Complete!")
        print("=" * 70)
        print(f"\n📊 Results:")
        print(f"   Output:         {result.output_dir}")
        print(f"   Final Loss:     {result.final_loss:.4f}")
        print(f"   Epochs:         {result.epochs_completed}")
        print(f"   Steps:          {result.steps_completed}")

        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            print(f"\n📁 Saved Checkpoints ({len(checkpoints)}):")
            for ckpt in checkpoints[-3:]:  # Show last 3
                print(f"   - {ckpt.name}: loss={ckpt.loss:.4f}, epoch={ckpt.epoch}")

        print(f"\n✨ Fine-tuned model ready at: {result.output_dir}")
        print("\nNext steps:")
        print("  1. Test the model: python -m backend.inference.test_model")
        print("  2. Deploy via API: Update model_loader.py")
        print("  3. Quantize for production: Use scripts/quantize_model.py")

    else:
        print("✗ Training Failed")
        print("=" * 70)
        print(f"\n❌ Error: {result.error}")
        print(f"\n📊 Progress:")
        print(f"   Epochs:  {result.epochs_completed}")
        print(f"   Steps:   {result.steps_completed}")
        sys.exit(1)


if __name__ == '__main__':
    main()

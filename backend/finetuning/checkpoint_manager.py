"""
Checkpoint Manager Module

Manages model checkpoints during and after fine-tuning.
"""

from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a model checkpoint."""
    name: str
    path: str
    epoch: int
    step: int
    loss: float
    created_at: str
    size_mb: float
    metadata: Optional[Dict] = None


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Save checkpoints at regular intervals
    - Keep best N checkpoints based on loss
    - Clean up old checkpoints
    - Resume from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        keep_every_n_epochs: int = 1
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_best_n: Number of best checkpoints to keep
            keep_every_n_epochs: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best_n = keep_best_n
        self.keep_every_n_epochs = keep_every_n_epochs

        self.checkpoints: List[Checkpoint] = []
        self._load_checkpoint_index()

    def _load_checkpoint_index(self):
        """Load checkpoint index from disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)

            self.checkpoints = [
                Checkpoint(**ckpt) for ckpt in data.get("checkpoints", [])
            ]

            logger.info(f"Loaded {len(self.checkpoints)} checkpoints from index")

    def _save_checkpoint_index(self):
        """Save checkpoint index to disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"

        data = {
            "checkpoints": [
                {
                    "name": ckpt.name,
                    "path": ckpt.path,
                    "epoch": ckpt.epoch,
                    "step": ckpt.step,
                    "loss": ckpt.loss,
                    "created_at": ckpt.created_at,
                    "size_mb": ckpt.size_mb,
                    "metadata": ckpt.metadata
                }
                for ckpt in self.checkpoints
            ]
        }

        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        model,
        tokenizer,
        epoch: int,
        step: int,
        loss: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            epoch: Current epoch
            step: Current step
            loss: Current loss
            metadata: Optional metadata

        Returns:
            Checkpoint path
        """
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        logger.info(f"Saving checkpoint: {checkpoint_name}")

        # Create checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Placeholder for actual model saving
        # model.save_pretrained(str(checkpoint_path))
        # tokenizer.save_pretrained(str(checkpoint_path))

        # Calculate size (placeholder)
        size_mb = 100.0  # Placeholder

        # Create checkpoint record
        checkpoint = Checkpoint(
            name=checkpoint_name,
            path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            loss=loss,
            created_at=datetime.now().isoformat(),
            size_mb=size_mb,
            metadata=metadata
        )

        self.checkpoints.append(checkpoint)
        self._save_checkpoint_index()

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        return str(checkpoint_path)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on policy."""
        if len(self.checkpoints) <= self.keep_best_n:
            return

        # Sort by loss (ascending)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.loss)

        # Keep best N
        to_keep = set(ckpt.name for ckpt in sorted_checkpoints[:self.keep_best_n])

        # Keep epoch checkpoints
        for ckpt in self.checkpoints:
            if ckpt.epoch % self.keep_every_n_epochs == 0:
                to_keep.add(ckpt.name)

        # Remove checkpoints not in keep set
        to_remove = [ckpt for ckpt in self.checkpoints if ckpt.name not in to_keep]

        for ckpt in to_remove:
            logger.info(f"Removing checkpoint: {ckpt.name}")

            # Delete checkpoint directory (placeholder)
            # import shutil
            # shutil.rmtree(ckpt.path, ignore_errors=True)

            self.checkpoints.remove(ckpt)

        self._save_checkpoint_index()

    def get_best_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the best checkpoint (lowest loss).

        Returns:
            Best checkpoint or None
        """
        if not self.checkpoints:
            return None

        return min(self.checkpoints, key=lambda x: x.loss)

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint.

        Returns:
            Latest checkpoint or None
        """
        if not self.checkpoints:
            return None

        return max(self.checkpoints, key=lambda x: x.created_at)

    def load_checkpoint(self, checkpoint_name: str):
        """
        Load a specific checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to load

        Returns:
            Loaded model and tokenizer (placeholder)
        """
        checkpoint = next(
            (ckpt for ckpt in self.checkpoints if ckpt.name == checkpoint_name),
            None
        )

        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")

        logger.info(f"Loading checkpoint: {checkpoint_name}")

        # Placeholder for actual loading
        # model = AutoModel.from_pretrained(checkpoint.path)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint.path)

        return None, None  # Placeholder

    def list_checkpoints(self) -> List[Checkpoint]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoints
        """
        return self.checkpoints.copy()

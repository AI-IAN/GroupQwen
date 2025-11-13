"""
Model Loader Module

Handles loading and unloading of models with resource management.
"""

from typing import Dict, Optional, Any
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    framework: str
    vram_gb: float
    status: str  # 'loaded', 'idle', 'loading', 'error'
    config: Dict[str, Any]


class ModelLoader:
    """
    Manages loading and unloading of models.

    Handles:
    - Resource allocation (VRAM management)
    - Model configuration loading
    - Framework-specific initialization
    - Model swapping for memory optimization
    """

    def __init__(self, model_config_path: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            model_config_path: Path to model configuration YAML
        """
        if model_config_path is None:
            model_config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"

        with open(model_config_path, 'r') as f:
            self.model_configs = yaml.safe_load(f)["models"]

        self.loaded_models: Dict[str, ModelInfo] = {}

    def get_model_config(self, model_key: str) -> Optional[Dict]:
        """
        Get configuration for a specific model.

        Args:
            model_key: Model key (e.g., 'qwen3_8b')

        Returns:
            Model configuration dict
        """
        return self.model_configs.get(model_key)

    def load_model(self, model_key: str) -> bool:
        """
        Load a model into memory.

        Args:
            model_key: Model key to load

        Returns:
            True if successfully loaded
        """
        config = self.get_model_config(model_key)
        if not config:
            logger.error(f"Model config not found: {model_key}")
            return False

        logger.info(f"Loading model: {model_key} ({config['name']})")

        # Check if already loaded
        if model_key in self.loaded_models:
            logger.info(f"Model already loaded: {model_key}")
            return True

        # Create model info
        model_info = ModelInfo(
            name=config["name"],
            framework=config["framework"],
            vram_gb=config["vram_gb"],
            status="loading",
            config=config
        )

        self.loaded_models[model_key] = model_info

        # Actual loading would happen here via framework-specific handlers
        # This is a placeholder for the actual implementation

        logger.info(f"Model loaded successfully: {model_key}")
        model_info.status = "loaded"

        return True

    def unload_model(self, model_key: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_key: Model key to unload

        Returns:
            True if successfully unloaded
        """
        if model_key not in self.loaded_models:
            logger.warning(f"Model not loaded: {model_key}")
            return False

        logger.info(f"Unloading model: {model_key}")

        # Actual unloading logic would be framework-specific
        del self.loaded_models[model_key]

        logger.info(f"Model unloaded: {model_key}")
        return True

    def get_loaded_models(self) -> Dict[str, ModelInfo]:
        """
        Get information about all loaded models.

        Returns:
            Dict of loaded models
        """
        return self.loaded_models.copy()

    def is_model_loaded(self, model_key: str) -> bool:
        """
        Check if a model is loaded.

        Args:
            model_key: Model key

        Returns:
            True if loaded
        """
        return model_key in self.loaded_models

    def get_total_vram_used(self) -> float:
        """
        Calculate total VRAM used by loaded models.

        Returns:
            Total VRAM in GB
        """
        return sum(
            model.vram_gb
            for model in self.loaded_models.values()
            if model.status == "loaded"
        )

    def can_load_model(self, model_key: str, available_vram_gb: float) -> bool:
        """
        Check if a model can be loaded given available VRAM.

        Args:
            model_key: Model key
            available_vram_gb: Available VRAM in GB

        Returns:
            True if can be loaded
        """
        config = self.get_model_config(model_key)
        if not config:
            return False

        required_vram = config["vram_gb"]
        current_usage = self.get_total_vram_used()

        return (current_usage + required_vram) <= available_vram_gb

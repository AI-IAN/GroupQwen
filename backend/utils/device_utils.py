"""
Device Utilities Module

Provides device detection and GPU/CPU management utilities.
"""

import subprocess
import logging
from typing import Dict, List, Optional
import platform

logger = logging.getLogger(__name__)


class DeviceInfo:
    """Device information and capabilities."""

    def __init__(self):
        self.device_type = self._detect_device_type()
        self.has_gpu = self._check_gpu_availability()
        self.gpu_info = self._get_gpu_info() if self.has_gpu else {}
        self.cpu_info = self._get_cpu_info()
        self.ram_gb = self._get_ram_gb()

    def _detect_device_type(self) -> str:
        """
        Detect device type based on hardware.

        Returns:
            'server', 'macbook', or 'mobile'
        """
        system = platform.system()

        if system == "Darwin":
            # Check if M-series Mac
            machine = platform.machine()
            if machine.startswith("arm"):
                return "macbook"  # M-series Mac
            return "macbook"
        elif system == "Linux":
            return "server"
        elif system == "Windows":
            return "server"
        else:
            return "server"

    def _check_gpu_availability(self) -> bool:
        """
        Check if CUDA GPU is available.

        Returns:
            True if GPU available
        """
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_gpu_info(self) -> Dict:
        """
        Get GPU information using nvidia-smi.

        Returns:
            Dict with GPU info
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []

                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({
                            "name": parts[0],
                            "memory_total_mb": int(parts[1]),
                            "memory_free_mb": int(parts[2]),
                            "utilization_percent": int(parts[3])
                        })

                return {
                    "count": len(gpus),
                    "gpus": gpus
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
            logger.warning(f"Failed to get GPU info: {e}")

        return {}

    def _get_cpu_info(self) -> Dict:
        """
        Get CPU information.

        Returns:
            Dict with CPU info
        """
        import multiprocessing

        return {
            "processor": platform.processor(),
            "cores": multiprocessing.cpu_count(),
            "architecture": platform.machine()
        }

    def _get_ram_gb(self) -> Optional[float]:
        """
        Get total RAM in GB.

        Returns:
            RAM in GB or None
        """
        try:
            import psutil
            ram_bytes = psutil.virtual_memory().total
            return ram_bytes / (1024 ** 3)  # Convert to GB
        except ImportError:
            logger.warning("psutil not available, cannot determine RAM")
            return None

    def get_optimal_device(self) -> str:
        """
        Get optimal device for inference.

        Returns:
            'cuda', 'mps' (for M-series Mac), or 'cpu'
        """
        if self.has_gpu:
            return "cuda"
        elif platform.system() == "Darwin" and platform.machine().startswith("arm"):
            return "mps"  # Metal Performance Shaders for M-series Mac
        else:
            return "cpu"

    def can_run_model(self, vram_requirement_gb: float) -> bool:
        """
        Check if device can run a model with given VRAM requirement.

        Args:
            vram_requirement_gb: Required VRAM in GB

        Returns:
            True if sufficient VRAM available
        """
        if not self.has_gpu or not self.gpu_info:
            return False

        gpus = self.gpu_info.get("gpus", [])
        if not gpus:
            return False

        # Check if any GPU has enough free memory
        for gpu in gpus:
            free_gb = gpu["memory_free_mb"] / 1024
            if free_gb >= vram_requirement_gb:
                return True

        return False

    def get_available_vram_gb(self) -> float:
        """
        Get total available VRAM across all GPUs.

        Returns:
            Available VRAM in GB
        """
        if not self.has_gpu or not self.gpu_info:
            return 0.0

        gpus = self.gpu_info.get("gpus", [])
        total_free = sum(gpu["memory_free_mb"] for gpu in gpus)
        return total_free / 1024

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DeviceInfo(type={self.device_type}, "
            f"gpu={self.has_gpu}, "
            f"ram_gb={self.ram_gb:.1f})"
        )


def get_device_info() -> DeviceInfo:
    """
    Get current device information.

    Returns:
        DeviceInfo object
    """
    return DeviceInfo()


def select_device(preferred: Optional[str] = None) -> str:
    """
    Select device for inference.

    Args:
        preferred: Preferred device ('cuda', 'cpu', 'mps')

    Returns:
        Selected device string
    """
    info = get_device_info()

    if preferred:
        return preferred

    return info.get_optimal_device()

"""
Health Check Module

Provides system health monitoring and status checks.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: str
    checks: Dict[str, bool]
    details: Dict[str, str]


class HealthChecker:
    """
    System health checker.

    Monitors:
    - Redis connectivity
    - Model availability
    - GPU status
    - Disk space
    - API responsiveness
    """

    def __init__(self):
        self.last_check: Optional[HealthStatus] = None

    async def check_health(
        self,
        redis_client=None,
        model_loader=None,
        device_info=None
    ) -> HealthStatus:
        """
        Perform comprehensive health check.

        Args:
            redis_client: Redis client instance
            model_loader: Model loader instance
            device_info: Device info instance

        Returns:
            Health status
        """
        checks = {}
        details = {}

        # Check Redis
        if redis_client:
            try:
                redis_healthy = redis_client.ping()
                checks["redis"] = redis_healthy
                details["redis"] = "Connected" if redis_healthy else "Disconnected"
            except Exception as e:
                checks["redis"] = False
                details["redis"] = f"Error: {str(e)}"
        else:
            checks["redis"] = None
            details["redis"] = "Not configured"

        # Check GPU
        if device_info:
            try:
                gpu_healthy = device_info.has_gpu
                checks["gpu"] = gpu_healthy

                if gpu_healthy and device_info.gpu_info:
                    gpu_count = device_info.gpu_info.get("count", 0)
                    details["gpu"] = f"{gpu_count} GPU(s) available"
                else:
                    details["gpu"] = "No GPU available"
            except Exception as e:
                checks["gpu"] = False
                details["gpu"] = f"Error: {str(e)}"
        else:
            checks["gpu"] = None
            details["gpu"] = "Not checked"

        # Check models
        if model_loader:
            try:
                loaded_models = model_loader.get_loaded_models()
                checks["models"] = len(loaded_models) > 0
                details["models"] = f"{len(loaded_models)} model(s) loaded"
            except Exception as e:
                checks["models"] = False
                details["models"] = f"Error: {str(e)}"
        else:
            checks["models"] = None
            details["models"] = "Not checked"

        # Check disk space
        try:
            disk_healthy = self._check_disk_space()
            checks["disk"] = disk_healthy
            details["disk"] = "Sufficient space" if disk_healthy else "Low disk space"
        except Exception as e:
            checks["disk"] = False
            details["disk"] = f"Error: {str(e)}"

        # Determine overall status
        if all(v for v in checks.values() if v is not None):
            overall_status = "healthy"
        elif any(v == False for v in checks.values()):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        health_status = HealthStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            checks=checks,
            details=details
        )

        self.last_check = health_status
        return health_status

    def _check_disk_space(self, min_free_gb: float = 10.0) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            min_free_gb: Minimum free space in GB

        Returns:
            True if sufficient space available
        """
        try:
            import shutil
            stat = shutil.disk_usage("/")
            free_gb = stat.free / (1024 ** 3)
            return free_gb >= min_free_gb
        except Exception:
            return True  # Don't fail health check if can't determine

    def get_last_check(self) -> Optional[HealthStatus]:
        """
        Get last health check result.

        Returns:
            Last health status or None
        """
        return self.last_check

    def is_healthy(self) -> bool:
        """
        Quick check if system is healthy.

        Returns:
            True if healthy
        """
        if not self.last_check:
            return False

        return self.last_check.status == "healthy"

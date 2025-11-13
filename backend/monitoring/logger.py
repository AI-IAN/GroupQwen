"""
Logging Configuration Module

Configures structured logging for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "detailed"
) -> logging.Logger:
    """
    Setup logger with specified configuration.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        log_format: Format style ('simple', 'detailed', 'json')

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Choose format
    if log_format == "simple":
        formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
    elif log_format == "json":
        # JSON format for structured logging
        formatter = JSONFormatter()
    else:  # detailed
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted string
        """
        import json

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get or create logger with default configuration.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Application-wide logger configuration
def configure_app_logging(log_level: str = "INFO", log_dir: Optional[str] = None):
    """
    Configure application-wide logging.

    Args:
        log_level: Default log level
        log_dir: Directory for log files
    """
    # Main application logger
    app_logger = setup_logger(
        "backend",
        log_level=log_level,
        log_file=f"{log_dir}/app.log" if log_dir else None,
        log_format="detailed"
    )

    # API logger
    api_logger = setup_logger(
        "backend.api",
        log_level=log_level,
        log_file=f"{log_dir}/api.log" if log_dir else None,
        log_format="detailed"
    )

    # Inference logger
    inference_logger = setup_logger(
        "backend.inference",
        log_level=log_level,
        log_file=f"{log_dir}/inference.log" if log_dir else None,
        log_format="detailed"
    )

    return {
        "app": app_logger,
        "api": api_logger,
        "inference": inference_logger
    }

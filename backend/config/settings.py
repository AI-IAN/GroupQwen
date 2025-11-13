"""
Configuration settings for Qwen3 Local AI System.
Loads from environment variables with sensible defaults.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model Configuration
    primary_model: str = Field(default="qwen3-32b-awq", env="PRIMARY_MODEL")
    secondary_model: str = Field(default="qwen3-14b-awq", env="SECONDARY_MODEL")
    cache_model: str = Field(default="qwen3-8b-q4", env="CACHE_MODEL")

    # Hardware
    gpu_memory_fraction: float = Field(default=0.95, env="GPU_MEMORY_FRACTION")
    num_gpu_layers: int = Field(default=32, env="NUM_GPU_LAYERS")
    batch_size: int = Field(default=1, env="BATCH_SIZE")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(default=86400, env="CACHE_TTL")  # 24 hours

    # Cache Configuration
    cache_distance_threshold: float = Field(default=0.10, env="CACHE_DISTANCE_THRESHOLD")
    cache_ttl_general: int = Field(default=86400, env="CACHE_TTL_GENERAL")
    cache_ttl_timesensitive: int = Field(default=3600, env="CACHE_TTL_TIMESENSITIVE")
    cache_max_size_mb: int = Field(default=10000, env="CACHE_MAX_SIZE_MB")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )

    # Fine-tuning
    lora_rank: int = Field(default=128, env="LORA_RANK")
    lora_alpha: int = Field(default=32, env="LORA_ALPHA")
    training_batch_size: int = Field(default=2, env="TRAINING_BATCH_SIZE")
    gradient_accumulation_steps: int = Field(default=4, env="GRADIENT_ACCUMULATION_STEPS")

    # API Configuration
    api_port: int = Field(default=8000, env="API_PORT")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")

    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    metrics_db_url: Optional[str] = Field(
        default="postgresql://user:pass@localhost/metrics",
        env="METRICS_DB_URL"
    )

    # Device Configuration
    device_type: str = Field(default="server", env="DEVICE_TYPE")  # server | macbook | mobile
    tailscale_ip: Optional[str] = Field(default=None, env="TAILSCALE_IP")

    # Model Paths
    models_dir: str = Field(default="./models", env="MODELS_DIR")
    checkpoints_dir: str = Field(default="./checkpoints", env="CHECKPOINTS_DIR")

    # vLLM Configuration
    vllm_tensor_parallel_size: int = Field(default=1, env="VLLM_TENSOR_PARALLEL_SIZE")
    vllm_gpu_memory_utilization: float = Field(default=0.90, env="VLLM_GPU_MEMORY_UTILIZATION")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Cache configuration dictionary
CACHE_CONFIG = {
    "redis_url": settings.redis_url,
    "embedding_model": settings.embedding_model,
    "distance_threshold": settings.cache_distance_threshold,
    "ttl_general": settings.cache_ttl_general,
    "ttl_timesensitive": settings.cache_ttl_timesensitive,
    "max_cache_size_mb": settings.cache_max_size_mb,
}

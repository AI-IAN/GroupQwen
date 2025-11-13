"""
Pydantic Models for API Request/Response Schemas
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


# Chat Completion Models
class Message(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    messages: List[Message]
    model: Optional[str] = None  # If None, router will select
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    device: Optional[str] = "server"  # server, macbook, mobile


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None


# Vision Models
class VisionAnalysisRequest(BaseModel):
    """Request for vision analysis."""
    image: str  # Base64 or URL
    prompt: str
    return_bboxes: bool = False
    max_tokens: int = 1024


class VisionAnalysisResponse(BaseModel):
    """Response from vision analysis."""
    content: str
    bounding_boxes: Optional[List[Dict]] = None
    latency_ms: float


# Translation Models
class TranslationRequest(BaseModel):
    """Request for translation."""
    text: str
    source_lang: Optional[str] = None
    target_lang: str = "en"
    preserve_formatting: bool = True


class TranslationResponse(BaseModel):
    """Response from translation."""
    translated_text: str
    source_lang: str
    target_lang: str
    latency_ms: float


# Fine-tuning Models
class FinetuneRequest(BaseModel):
    """Request to start fine-tuning."""
    base_model: str
    dataset_path: str
    epochs: int = Field(default=3, ge=1, le=10)
    lora_rank: int = Field(default=128, ge=8, le=256)
    learning_rate: float = Field(default=2e-4, ge=1e-5, le=1e-3)
    output_dir: str


class FinetuneResponse(BaseModel):
    """Response from fine-tuning request."""
    job_id: str
    status: str
    message: str


class FinetuneStatusResponse(BaseModel):
    """Status of fine-tuning job."""
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0.0 - 1.0
    current_epoch: Optional[int] = None
    current_loss: Optional[float] = None
    eta_seconds: Optional[int] = None


# System Models
class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    status: str  # running, idle, loading, error
    vram_used: Optional[float] = None
    vram_total: Optional[float] = None


class ModelsListResponse(BaseModel):
    """List of available models."""
    models: List[ModelInfo]


class CacheStatsResponse(BaseModel):
    """Cache statistics."""
    hit_rate: float
    total_cached: int
    total_requests: int
    cache_size_mb: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    checks: Dict[str, Optional[bool]]
    details: Dict[str, str]


class MetricsResponse(BaseModel):
    """System metrics."""
    total_queries: int
    cache_hit_rate: float
    avg_latency_ms: float
    models_used: List[str]
    daily_stats: Optional[Dict] = None

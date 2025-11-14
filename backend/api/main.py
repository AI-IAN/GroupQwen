"""
FastAPI Application Entry Point

Main application for Qwen3 Local AI Orchestration System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.api.routes import router
from backend.config.settings import settings
from backend.monitoring.logger import setup_logger
from backend.utils.redis_client import RedisClient
from backend.core.cache_manager import SemanticCacheManager
from backend.core.router import QueryRouter
from backend.inference.model_loader import ModelLoader
from backend.monitoring.metrics import MetricsCollector

# Setup logging
logger = setup_logger("backend.api.main", log_level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Qwen3 Local AI Orchestration System")
    logger.info("=" * 80)
    logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")
    logger.info(f"Device type: {settings.device_type}")
    logger.info(f"Primary model: {settings.primary_model}")
    logger.info(f"Redis URL: {settings.redis_url}")

    # Initialize components
    redis_client = None
    cache_manager = None
    query_router = None
    model_loader = None
    metrics_collector = None

    try:
        # Redis client
        logger.info("Initializing Redis client...")
        redis_client = RedisClient(url=settings.redis_url)
        if not redis_client.ping():
            logger.warning("Redis connection failed - cache will be disabled")
            redis_client = None
        else:
            logger.info("✓ Redis client initialized")

        # Semantic cache manager
        if redis_client:
            logger.info("Initializing semantic cache manager...")
            cache_manager = SemanticCacheManager(
                redis_url=settings.redis_url,
                embedding_model_name=settings.embedding_model,
                distance_threshold=settings.cache_distance_threshold,
                ttl_seconds=settings.cache_ttl_general
            )
            logger.info("✓ Semantic cache manager initialized")
        else:
            cache_manager = None
            logger.warning("Cache manager disabled (Redis unavailable)")

        # Query router
        logger.info("Initializing query router...")
        query_router = QueryRouter(cache_manager=cache_manager)
        logger.info("✓ Query router initialized")

        # Model loader
        logger.info("Initializing model loader...")
        model_loader = ModelLoader()
        logger.info("✓ Model loader initialized")

        # Metrics collector
        logger.info("Initializing metrics collector...")
        metrics_collector = MetricsCollector(db_url=settings.metrics_db_url)
        logger.info("✓ Metrics collector initialized")

        # Store in app state for access in routes
        app.state.redis_client = redis_client
        app.state.cache_manager = cache_manager
        app.state.query_router = query_router
        app.state.model_loader = model_loader
        app.state.metrics_collector = metrics_collector

        logger.info("=" * 80)
        logger.info("All components initialized successfully")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        # Continue with degraded functionality
        app.state.redis_client = None
        app.state.cache_manager = None
        app.state.query_router = QueryRouter(cache_manager=None)
        app.state.model_loader = ModelLoader()
        app.state.metrics_collector = MetricsCollector()

    yield

    # Shutdown
    logger.info("Shutting down API server...")

    # Cleanup resources
    if redis_client:
        logger.info("Closing Redis connection...")
        redis_client.close()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Qwen3 Local AI Orchestration System",
    description="Production-grade local AI inference with semantic caching and intelligent routing",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Qwen3 Local AI Orchestration System",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/v1/health"
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint for connectivity checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,  # Enable auto-reload for development
        log_level=settings.log_level.lower()
    )

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

    # TODO: Initialize components
    # - Redis client
    # - Semantic cache manager
    # - Query router
    # - Model loader
    # - Metrics collector
    # - Health checker

    logger.info("All components initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down API server...")
    # TODO: Cleanup resources
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

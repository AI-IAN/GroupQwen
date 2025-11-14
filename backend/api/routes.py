"""
API Routes for Qwen3 Local System
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Optional
import time
import uuid
import logging
from datetime import datetime

from backend.api.models import (
    ChatCompletionRequest, ChatCompletionResponse,
    VisionAnalysisRequest, VisionAnalysisResponse,
    TranslationRequest, TranslationResponse,
    FinetuneRequest, FinetuneResponse, FinetuneStatusResponse,
    ModelsListResponse, ModelInfo,
    CacheStatsResponse, HealthCheckResponse, MetricsResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Chat Completion Endpoint
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completion endpoint.

    Routes queries intelligently based on complexity and caches responses.
    """
    try:
        # Extract last message
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        last_message = request.messages[-1]
        query = last_message.content

        # Prepare context from previous messages
        context = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in request.messages[:-1]
        )

        # Get components from app state
        query_router = req.app.state.query_router
        cache_manager = req.app.state.cache_manager
        metrics_collector = req.app.state.metrics_collector

        # Route query to appropriate model
        route_decision = query_router.route_query(
            query=query,
            context=context,
            device=request.device or "server",
            force_model=request.model
        )

        logger.info(
            f"Query routed: model={route_decision.model}, "
            f"complexity={route_decision.complexity_score:.2f}, "
            f"cache_hit={route_decision.use_cache}"
        )

        # Handle cache hit
        if route_decision.use_cache and cache_manager:
            cache_result = cache_manager.check(query)
            if cache_result:
                response_text = cache_result.response
                model_used = cache_result.model_used or "cache"
                cache_hit = True
                latency_ms = 5.0  # Cache latency
            else:
                # Cache check in router, but didn't actually hit
                cache_hit = False
                response_text = _generate_mock_response(query, route_decision.model)
                model_used = route_decision.model
                latency_ms = route_decision.estimated_latency_ms
        else:
            # Generate new response
            cache_hit = False
            # TODO: Call actual inference handler based on model
            response_text = _generate_mock_response(query, route_decision.model)
            model_used = route_decision.model
            latency_ms = route_decision.estimated_latency_ms

            # Store in cache for future use
            if cache_manager and not route_decision.use_cache:
                cache_manager.store(
                    prompt=query,
                    response=response_text,
                    metadata={
                        "model_used": model_used,
                        "complexity_score": route_decision.complexity_score
                    }
                )

        # Log metrics
        if metrics_collector:
            from backend.monitoring.metrics import QueryLog

            metrics_collector.log_query(QueryLog(
                timestamp=datetime.now(),
                text=query[:500],
                model=model_used,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
                user_id=None,
                success=True,
                complexity_score=route_decision.complexity_score,
                tokens_used=int(len(query.split()) * 1.3 + len(response_text.split()) * 1.3)
            ))

        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model_used,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": int(len(query.split()) * 1.3),
                "completion_tokens": int(len(response_text.split()) * 1.3),
                "total_tokens": int(len((query + response_text).split()) * 1.3)
            },
            metadata={
                "cache_hit": cache_hit,
                "complexity_score": route_decision.complexity_score,
                "model_selected": model_used,
                "latency_ms": latency_ms,
                "reasoning": route_decision.reasoning
            }
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _generate_mock_response(query: str, model: str) -> str:
    """Generate a mock response for testing (placeholder for actual inference)."""
    return (
        f"[Mock Response from {model}]\n\n"
        f"This is a placeholder response to demonstrate the routing and caching system.\n\n"
        f"Your query: '{query[:200]}{'...' if len(query) > 200 else ''}'\n\n"
        f"In production, this would be replaced with actual inference from {model}. "
        f"The system has successfully:\n"
        f"- Analyzed query complexity\n"
        f"- Routed to the appropriate model\n"
        f"- Prepared for caching\n"
        f"- Logged metrics\n\n"
        f"Next step: Implement actual model inference!"
    )


# Vision Analysis Endpoint
@router.post("/v1/vision/analyze", response_model=VisionAnalysisResponse)
async def analyze_vision(request: VisionAnalysisRequest):
    """
    Analyze images with Qwen3-VL.

    Supports: image understanding, OCR, GUI automation, screenshot analysis.
    """
    try:
        # TODO: Use VisionHandler for actual analysis
        # vision_handler.analyze(request)

        response = VisionAnalysisResponse(
            content=f"Vision analysis: {request.prompt[:50]}...",
            bounding_boxes=[] if not request.return_bboxes else [
                {"label": "example", "bbox": [0, 0, 100, 100], "confidence": 0.9}
            ],
            latency_ms=100.0
        )

        return response

    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Translation Endpoint
@router.post("/v1/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text using Qwen3-MT (92 languages supported).
    """
    try:
        # TODO: Use TranslationHandler for actual translation
        # translation_handler.translate(request)

        response = TranslationResponse(
            translated_text=f"[Translated: {request.text[:30]}...]",
            source_lang=request.source_lang or "en",
            target_lang=request.target_lang,
            latency_ms=150.0
        )

        return response

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Fine-tuning Job Registry (in-memory, use database in production)
finetune_jobs = {}


def run_finetuning_background(job_id: str, request: FinetuneRequest):
    """Background task for running fine-tuning."""
    try:
        from backend.finetuning.trainer import Trainer, TrainingConfig
        from backend.finetuning.checkpoint_manager import CheckpointManager

        # Update status
        finetune_jobs[job_id]['status'] = 'running'
        finetune_jobs[job_id]['started_at'] = datetime.now().isoformat()

        # Create config
        config = TrainingConfig(
            model_name=request.base_model,
            dataset_path=request.dataset_path,
            output_dir=f"./models/finetuned/{job_id}",
            epochs=request.epochs or 3,
            batch_size=request.batch_size or 4,
            learning_rate=request.learning_rate or 2e-4,
            lora_rank=request.lora_rank or 16,
            lora_alpha=request.lora_alpha or 32,
            save_steps=request.save_steps or 100
        )

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=f"./checkpoints/{job_id}"
        )

        # Initialize trainer
        trainer = Trainer(config, checkpoint_manager)

        # Set progress callback
        def update_progress(metrics):
            if job_id in finetune_jobs:
                finetune_jobs[job_id]['progress'] = metrics['step'] / (config.epochs * 100)  # Estimate
                finetune_jobs[job_id]['current_epoch'] = metrics['epoch']
                finetune_jobs[job_id]['current_loss'] = metrics['loss']
                finetune_jobs[job_id]['learning_rate'] = metrics['lr']

        trainer.set_progress_callback(update_progress)

        # Run training
        result = trainer.train()

        # Update job with result
        if result.success:
            finetune_jobs[job_id]['status'] = 'completed'
            finetune_jobs[job_id]['output_dir'] = result.output_dir
            finetune_jobs[job_id]['final_loss'] = result.final_loss
            finetune_jobs[job_id]['epochs_completed'] = result.epochs_completed
            finetune_jobs[job_id]['steps_completed'] = result.steps_completed
        else:
            finetune_jobs[job_id]['status'] = 'failed'
            finetune_jobs[job_id]['error'] = result.error

        finetune_jobs[job_id]['completed_at'] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Fine-tuning background task error: {e}", exc_info=True)
        finetune_jobs[job_id]['status'] = 'failed'
        finetune_jobs[job_id]['error'] = str(e)


# Fine-tuning Endpoints
@router.post("/v1/finetune/start", response_model=FinetuneResponse)
async def start_finetuning(
    request: FinetuneRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a fine-tuning job on specified model.

    Runs training in background, allowing API to continue serving requests.
    """
    try:
        job_id = str(uuid.uuid4())

        # Initialize job in registry
        finetune_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'base_model': request.base_model,
            'dataset_path': request.dataset_path,
            'created_at': datetime.now().isoformat(),
            'progress': 0.0,
            'current_epoch': 0,
            'current_loss': 0.0,
            'learning_rate': request.learning_rate or 2e-4
        }

        # Start training in background
        background_tasks.add_task(run_finetuning_background, job_id, request)

        logger.info(f"Fine-tuning job {job_id} queued for model {request.base_model}")

        return FinetuneResponse(
            job_id=job_id,
            status="queued",
            message=f"Fine-tuning job {job_id} queued for {request.base_model}"
        )

    except Exception as e:
        logger.error(f"Fine-tuning start error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/finetune/status/{job_id}", response_model=FinetuneStatusResponse)
async def get_finetuning_status(job_id: str):
    """
    Get status of a fine-tuning job.

    Returns current progress, loss, and ETA.
    """
    try:
        if job_id not in finetune_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job = finetune_jobs[job_id]

        # Calculate ETA (rough estimate)
        eta_seconds = None
        if job['status'] == 'running' and job.get('progress', 0) > 0:
            # Estimate based on progress
            started_at = datetime.fromisoformat(job['started_at'])
            elapsed = (datetime.now() - started_at).total_seconds()
            if job['progress'] > 0:
                estimated_total = elapsed / job['progress']
                eta_seconds = int(estimated_total - elapsed)

        return FinetuneStatusResponse(
            job_id=job_id,
            status=job['status'],
            progress=job.get('progress', 0.0),
            current_epoch=job.get('current_epoch', 0),
            current_loss=job.get('current_loss', 0.0),
            eta_seconds=eta_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fine-tuning status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/finetune/cancel/{job_id}")
async def cancel_finetuning(job_id: str):
    """
    Cancel a fine-tuning job.

    Note: Background tasks cannot be truly cancelled in FastAPI,
    but we mark the job as cancelled in our registry.
    """
    try:
        if job_id not in finetune_jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job = finetune_jobs[job_id]

        if job['status'] in ['completed', 'failed', 'cancelled']:
            return {
                "message": f"Job {job_id} already {job['status']}",
                "status": job['status']
            }

        # Mark as cancelled
        finetune_jobs[job_id]['status'] = 'cancelled'
        finetune_jobs[job_id]['cancelled_at'] = datetime.now().isoformat()

        logger.info(f"Fine-tuning job {job_id} cancelled")

        return {
            "message": f"Job {job_id} cancelled",
            "status": "cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fine-tuning cancel error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Model Management Endpoints
@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """
    List available models and their status.
    """
    try:
        # TODO: Get actual model status from ModelLoader

        models = [
            ModelInfo(name="qwen3-32b", status="running", vram_used=20.0, vram_total=32.0),
            ModelInfo(name="qwen3-14b", status="running", vram_used=11.0, vram_total=32.0),
            ModelInfo(name="qwen3-8b", status="idle"),
            ModelInfo(name="qwen3-vl", status="idle"),
        ]

        return ModelsListResponse(models=models)

    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache Management Endpoints
@router.get("/v1/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(req: Request):
    """
    Get semantic cache performance statistics.
    """
    try:
        cache_manager = req.app.state.cache_manager

        if cache_manager:
            stats = cache_manager.get_stats()
            return CacheStatsResponse(
                hit_rate=stats.hit_rate,
                total_cached=stats.total_cached,
                total_requests=stats.total_requests,
                cache_size_mb=stats.cache_size_mb
            )
        else:
            # Cache disabled
            return CacheStatsResponse(
                hit_rate=0.0,
                total_cached=0,
                total_requests=0,
                cache_size_mb=0.0
            )

    except Exception as e:
        logger.error(f"Cache stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/cache/clear")
async def clear_cache(req: Request):
    """
    Clear all cached responses.
    """
    try:
        cache_manager = req.app.state.cache_manager

        if cache_manager:
            cache_manager.clear_all()
            return {"message": "Cache cleared successfully", "status": "success"}
        else:
            return {"message": "Cache is disabled", "status": "disabled"}

    except Exception as e:
        logger.error(f"Clear cache error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# System Health and Metrics
@router.get("/v1/health", response_model=HealthCheckResponse)
async def health_check(req: Request):
    """
    System health check endpoint.
    """
    try:
        from backend.monitoring.health_check import HealthChecker

        # Get components from app state
        redis_client = req.app.state.redis_client
        model_loader = req.app.state.model_loader

        # Perform health check
        health_checker = HealthChecker()
        health_status = await health_checker.check_health(
            redis_client=redis_client,
            model_loader=model_loader,
            device_info=None  # TODO: Add device_info
        )

        return HealthCheckResponse(
            status=health_status.status,
            timestamp=health_status.timestamp,
            checks=health_status.checks,
            details=health_status.details
        )

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        # Return unhealthy status instead of error
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            checks={"error": False},
            details={"error": str(e)}
        )


@router.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics(req: Request):
    """
    Get system performance metrics.
    """
    try:
        metrics_collector = req.app.state.metrics_collector

        if metrics_collector:
            stats = metrics_collector.get_summary_stats()
            return MetricsResponse(
                total_queries=stats["total_queries"],
                cache_hit_rate=stats["cache_hit_rate"],
                avg_latency_ms=stats["avg_latency_ms"],
                models_used=stats["models_used"]
            )
        else:
            # Metrics disabled
            return MetricsResponse(
                total_queries=0,
                cache_hit_rate=0.0,
                avg_latency_ms=0.0,
                models_used=[]
            )

    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

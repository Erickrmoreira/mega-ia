import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.deps import (
    require_api_key_factory,
    require_metrics_api_key_factory,
    require_model_info_api_key_factory,
    require_train_api_key_factory,
)
from api.middlewares.rate_limit_middleware import register_rate_limit_middleware
from api.middlewares.request_context import register_request_context_middleware
from api.rate_limit_backend import build_rate_limiter
from api.routes import (
    build_generate_router,
    build_health_router,
    build_metrics_router,
    build_models_router,
)
from api.telemetry import InMemoryTelemetry
from core.config import load_settings
from core.training_jobs import TrainingJobManager

app = FastAPI(
    title="Loteria Intelligent Generator API",
    description="API para geracao de apostas inteligente - Mega-Sena, Quina e Lotofacil",
    version="3.2.0",
)
logger = logging.getLogger("mega_ia.api")
settings = load_settings()
app.state.settings = settings

if settings.require_api_key and not settings.api_key:
    raise RuntimeError("MEGA_IA_API_KEY obrigatoria neste ambiente.")
if settings.require_artifact_hmac and not settings.artifact_hmac_key:
    raise RuntimeError("MEGA_IA_ARTIFACT_HMAC_KEY obrigatoria neste ambiente.")
if settings.app_env == "prod" and settings.rate_limit_backend != "redis":
    raise RuntimeError("Em producao, MEGA_IA_RATE_LIMIT_BACKEND deve ser 'redis'.")
if (
    settings.app_env == "prod"
    and settings.trust_proxy_headers
    and not settings.trusted_proxy_ips
):
    raise RuntimeError(
        "Em producao, MEGA_IA_TRUSTED_PROXY_IPS deve ser definido quando TRUST_PROXY_HEADERS=true."
    )

rate_limiter = build_rate_limiter(
    backend=settings.rate_limit_backend,
    redis_url=settings.redis_url or None,
    strict_redis=settings.require_redis_rate_limit,
)

app.state.output_dir = Path("output")
app.state.telemetry = InMemoryTelemetry()
app.state.train_jobs = TrainingJobManager(
    max_workers=1,
    redis_url=settings.redis_url or None,
    redis_prefix=settings.train_redis_prefix,
    strict_redis_lock=settings.require_redis_train_lock,
    job_ttl_seconds=settings.train_job_ttl_seconds,
    retry_count=settings.train_job_retry_count,
    retry_backoff_seconds=settings.train_job_retry_backoff_seconds,
    max_queue_size=settings.train_max_queue_size,
)

register_request_context_middleware(app=app, logger=logger)
register_rate_limit_middleware(app=app, settings=settings, rate_limiter=rate_limiter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
)

require_api_key = require_api_key_factory(settings)
require_train_api_key = require_train_api_key_factory(settings)
require_model_info_api_key = require_model_info_api_key_factory(settings)
require_metrics_api_key = require_metrics_api_key_factory(settings)

app.include_router(
    build_generate_router(
        require_api_key_dep=require_api_key,
        settings=settings,
        logger=logger,
    )
)
app.include_router(
    build_models_router(
        require_train_api_key_dep=require_train_api_key,
        require_model_info_api_key_dep=require_model_info_api_key,
        settings=settings,
    )
)
app.include_router(
    build_metrics_router(
        require_metrics_api_key_dep=require_metrics_api_key, settings=settings
    )
)
app.include_router(build_health_router(rate_limiter=rate_limiter, settings=settings))

app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")

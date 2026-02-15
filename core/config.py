import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Representa contrato único de configuração operacional da aplicação."""

    app_env: str
    require_api_key: bool
    require_artifact_hmac: bool
    require_redis_rate_limit: bool
    require_redis_train_lock: bool

    api_key: str
    train_api_key: str
    model_info_api_key: str
    artifact_hmac_key: str

    rate_limit_window_seconds: int
    rate_limit_max_requests: int
    train_rate_limit_window_seconds: int
    train_rate_limit_max_requests: int
    rate_limit_backend: str
    redis_url: str
    train_redis_prefix: str

    max_candidates: int
    max_request_cost: int
    train_cooldown_seconds: int
    train_job_ttl_seconds: int
    train_job_retry_count: int
    train_job_retry_backoff_seconds: int
    train_max_queue_size: int
    cors_allowed_origins: list[str]
    enable_metrics_endpoint: bool
    metrics_api_key: str
    trust_proxy_headers: bool
    trusted_proxy_ips: list[str]


def load_settings() -> Settings:
    """
    Carrega configuração de ambiente, normaliza tipos e aplica defaults seguros.

    Efeito colateral: consolida políticas de segurança e limites em um único objeto.
    """
    app_env = os.getenv("MEGA_IA_ENV", "dev").strip().lower()
    api_key = os.getenv("MEGA_IA_API_KEY", "").strip()
    train_api_key = os.getenv("MEGA_IA_TRAIN_API_KEY", api_key).strip()
    model_info_api_key = os.getenv(
        "MEGA_IA_MODEL_INFO_API_KEY", train_api_key or api_key
    ).strip()
    metrics_api_key = os.getenv(
        "MEGA_IA_METRICS_API_KEY", model_info_api_key or train_api_key or api_key
    ).strip()
    artifact_hmac_key = os.getenv("MEGA_IA_ARTIFACT_HMAC_KEY", "").strip()

    require_api_key = _as_bool(
        os.getenv("MEGA_IA_REQUIRE_API_KEY"), default=(app_env == "prod")
    )
    require_artifact_hmac = _as_bool(
        os.getenv("MEGA_IA_REQUIRE_ARTIFACT_HMAC"), default=(app_env == "prod")
    )
    require_redis_rate_limit = _as_bool(
        os.getenv("MEGA_IA_REQUIRE_REDIS_RATE_LIMIT"), default=(app_env == "prod")
    ) or (app_env == "prod")
    require_redis_train_lock = _as_bool(
        os.getenv("MEGA_IA_REQUIRE_REDIS_TRAIN_LOCK"), default=(app_env == "prod")
    ) or (app_env == "prod")
    trust_proxy_headers = _as_bool(
        os.getenv("MEGA_IA_TRUST_PROXY_HEADERS"), default=False
    )
    trusted_proxy_ips = [
        ip.strip()
        for ip in os.getenv("MEGA_IA_TRUSTED_PROXY_IPS", "").split(",")
        if ip.strip()
    ]

    cors_allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "MEGA_IA_CORS_ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]

    return Settings(
        app_env=app_env,
        require_api_key=require_api_key,
        require_artifact_hmac=require_artifact_hmac,
        require_redis_rate_limit=require_redis_rate_limit,
        require_redis_train_lock=require_redis_train_lock,
        api_key=api_key,
        train_api_key=train_api_key,
        model_info_api_key=model_info_api_key,
        artifact_hmac_key=artifact_hmac_key,
        rate_limit_window_seconds=int(
            os.getenv("MEGA_IA_RATE_LIMIT_WINDOW_SECONDS", "60")
        ),
        rate_limit_max_requests=int(os.getenv("MEGA_IA_RATE_LIMIT_MAX_REQUESTS", "30")),
        train_rate_limit_window_seconds=int(
            os.getenv("MEGA_IA_TRAIN_RATE_LIMIT_WINDOW_SECONDS", "300")
        ),
        train_rate_limit_max_requests=int(
            os.getenv("MEGA_IA_TRAIN_RATE_LIMIT_MAX_REQUESTS", "3")
        ),
        rate_limit_backend=os.getenv("MEGA_IA_RATE_LIMIT_BACKEND", "memory")
        .strip()
        .lower(),
        redis_url=os.getenv("MEGA_IA_REDIS_URL", "").strip(),
        train_redis_prefix=os.getenv("MEGA_IA_TRAIN_REDIS_PREFIX", "mega_ia").strip()
        or "mega_ia",
        max_candidates=int(os.getenv("MEGA_IA_MAX_CANDIDATES", "20000")),
        max_request_cost=int(os.getenv("MEGA_IA_MAX_REQUEST_COST", "600000")),
        train_cooldown_seconds=max(
            1, int(os.getenv("MEGA_IA_TRAIN_COOLDOWN_SECONDS", "120"))
        ),
        train_job_ttl_seconds=int(os.getenv("MEGA_IA_TRAIN_JOB_TTL_SECONDS", "86400")),
        train_job_retry_count=int(os.getenv("MEGA_IA_TRAIN_JOB_RETRY_COUNT", "1")),
        train_job_retry_backoff_seconds=int(
            os.getenv("MEGA_IA_TRAIN_JOB_RETRY_BACKOFF_SECONDS", "2")
        ),
        train_max_queue_size=max(
            1, int(os.getenv("MEGA_IA_TRAIN_MAX_QUEUE_SIZE", "10"))
        ),
        cors_allowed_origins=cors_allowed_origins,
        enable_metrics_endpoint=_as_bool(
            os.getenv("MEGA_IA_ENABLE_METRICS_ENDPOINT"), default=True
        ),
        metrics_api_key=metrics_api_key,
        trust_proxy_headers=trust_proxy_headers,
        trusted_proxy_ips=trusted_proxy_ips,
    )

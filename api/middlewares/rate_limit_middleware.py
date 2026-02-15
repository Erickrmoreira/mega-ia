import hashlib
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.deps import client_ip
from api.rate_limit_backend import MemoryRateLimiter


def register_rate_limit_middleware(app: FastAPI, settings, rate_limiter) -> None:
    """
    Registra middleware HTTP de controle de taxa por escopo funcional.

    Responsabilidade:
    - isolar quota de geração e treino para evitar bloqueio cruzado;
    - aplicar limite por origem/fingerprint de credencial;
    - retornar 429 com `Retry-After` quando houver excesso.
    """
    ip_lock = threading.Lock()

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        protected_paths = ("/generate/", "/models/train/")
        if request.url.path.startswith(protected_paths):
            ip_key = client_ip(request)
            api_key = request.headers.get("x-api-key", "").strip()
            scope = (
                "train" if request.url.path.startswith("/models/train/") else "generate"
            )

            if api_key:
                key_suffix = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
                key = f"{scope}:{ip_key}:{key_suffix}"
            else:
                key = f"{scope}:{ip_key}:-"

            if scope == "train":
                window_seconds = settings.train_rate_limit_window_seconds
                max_requests = settings.train_rate_limit_max_requests
            else:
                window_seconds = settings.rate_limit_window_seconds
                max_requests = settings.rate_limit_max_requests

            if isinstance(rate_limiter, MemoryRateLimiter):
                with ip_lock:
                    result = rate_limiter.allow(
                        key=key,
                        window_seconds=window_seconds,
                        max_requests=max_requests,
                    )
            else:
                result = rate_limiter.allow(
                    key=key,
                    window_seconds=window_seconds,
                    max_requests=max_requests,
                )
            if not result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            "Limite de requisicoes excedido. "
                            f"Maximo: {max_requests} por {window_seconds}s."
                        )
                    },
                    headers={
                        "Retry-After": str(result.retry_after_seconds),
                        "X-Request-ID": getattr(request.state, "request_id", "-"),
                    },
                )
        return await call_next(request)

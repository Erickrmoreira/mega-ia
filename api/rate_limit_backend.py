import importlib
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger("mega_ia.rate_limit")


@dataclass
class RateLimitResult:
    """Representa decisão de throttling e prazo de retry para uma chave de cliente."""

    allowed: bool
    retry_after_seconds: int = 0


class BaseRateLimiter:
    """Define contrato de autorização e healthcheck para backends de rate limit."""

    def allow(
        self, key: str, window_seconds: int, max_requests: int
    ) -> RateLimitResult:
        raise NotImplementedError

    def healthcheck(self) -> tuple[bool, str]:
        raise NotImplementedError


class MemoryRateLimiter(BaseRateLimiter):
    """Backend local de janela deslizante para execução single-node e desenvolvimento."""

    def __init__(self):
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._last_seen: dict[str, float] = {}
        self._max_buckets = int(os.getenv("MEGA_IA_RATE_LIMIT_MAX_BUCKETS", "50000"))

    def _evict_if_needed(self, now: float) -> None:
        if len(self._buckets) <= self._max_buckets:
            return
        # Contém crescimento de memória sob cardinalidade alta de chaves.
        overflow = len(self._buckets) - self._max_buckets
        oldest = sorted(self._last_seen.items(), key=lambda item: item[1])[:overflow]
        for key, _ in oldest:
            self._buckets.pop(key, None)
            self._last_seen.pop(key, None)

    def allow(
        self, key: str, window_seconds: int, max_requests: int
    ) -> RateLimitResult:
        now = time.time()
        cutoff = now - window_seconds
        bucket = self._buckets[key]

        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - bucket[0])))
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

        bucket.append(now)
        self._last_seen[key] = now
        self._evict_if_needed(now)
        return RateLimitResult(allowed=True)

    def healthcheck(self) -> tuple[bool, str]:
        return True, "memory_ok"


class RedisRateLimiter(BaseRateLimiter):
    """Backend distribuído com Redis para consistência de quota entre réplicas."""

    def __init__(self, redis_url: str):
        try:
            redis_module = importlib.import_module("redis")
            Redis = getattr(redis_module, "Redis")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Pacote redis nao disponivel.") from exc

        self._client = Redis.from_url(redis_url, decode_responses=False)
        self._client.ping()

    def allow(
        self, key: str, window_seconds: int, max_requests: int
    ) -> RateLimitResult:
        now = time.time()
        key_name = f"rl:{key}"
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(key_name, 0, now - window_seconds)
        pipe.zcard(key_name)
        _, current = pipe.execute()
        current = int(current)

        if current >= max_requests:
            oldest = self._client.zrange(key_name, 0, 0, withscores=True)
            retry_after = 1
            if oldest:
                _, ts = oldest[0]
                retry_after = max(1, int(window_seconds - (now - float(ts))))
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

        member = f"{now}:{key}"
        pipe = self._client.pipeline()
        pipe.zadd(key_name, {member: now})
        pipe.expire(key_name, window_seconds)
        pipe.execute()
        return RateLimitResult(allowed=True)

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self._client.ping()
            return True, "redis_ok"
        except Exception as exc:
            return False, f"redis_unavailable: {exc}"


def build_rate_limiter(
    backend: str, redis_url: str | None, strict_redis: bool = False
) -> BaseRateLimiter:
    """
    Resolve backend de rate limit conforme configuração e política de fallback.

    Efeito colateral: pode falhar startup em modo estrito para evitar operação sem Redis.
    """
    selected = backend.strip().lower()
    if selected == "redis":
        if not redis_url:
            if strict_redis:
                raise RuntimeError(
                    "MEGA_IA_REDIS_URL obrigatoria para rate limit redis."
                )
            logger.warning(
                "Rate limit Redis selecionado sem REDIS_URL. Usando memoria."
            )
            return MemoryRateLimiter()
        try:
            limiter = RedisRateLimiter(redis_url)
            logger.info("Rate limiter Redis ativo.")
            return limiter
        except Exception as exc:
            if strict_redis:
                raise RuntimeError(
                    f"Falha ao inicializar rate limiter Redis: {exc}"
                ) from exc
            logger.warning(
                "Falha ao inicializar rate limiter Redis: %s. Usando memoria.", exc
            )
            return MemoryRateLimiter()

    if strict_redis:
        raise RuntimeError(
            "Em modo estrito, MEGA_IA_RATE_LIMIT_BACKEND deve ser 'redis'."
        )

    logger.info("Rate limiter em memoria ativo.")
    return MemoryRateLimiter()

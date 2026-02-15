from __future__ import annotations

import importlib
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("mega_ia.training_jobs")


def _utc_now() -> str:
    """Gera timestamp UTC padronizado para metadados operacionais de jobs."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingJob:
    """Representa estado persistível de execução de um job de treino."""

    job_id: str
    lottery: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    max_retries: int = 0


class TrainingJobManager:
    """Orquestra jobs de treino com lock por loteria, retries e limite de backlog.

    Decisoes de projeto:
    - Evita dois treinos simultaneos da mesma modalidade.
    - Permite lock distribuido via Redis quando disponivel.
    - Impoe limite de fila para proteger CPU/memoria sob abuso.
    """

    def __init__(
        self,
        max_workers: int = 1,
        redis_url: str | None = None,
        redis_prefix: str = "mega_ia",
        strict_redis_lock: bool = False,
        job_ttl_seconds: int = 86400,
        retry_count: int = 1,
        retry_backoff_seconds: int = 2,
        max_queue_size: int = 10,
    ):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="train-job"
        )
        self._lock = threading.Lock()
        self._jobs: dict[str, TrainingJob] = {}
        self._running_by_lottery: dict[str, str] = {}
        self._futures: dict[str, Future] = {}
        self._last_train_by_lottery: dict[str, float] = {}
        self._job_lock_tokens: dict[str, str] = {}
        self._redis = None
        self._redis_prefix = (redis_prefix or "mega_ia").strip()
        self._strict_redis_lock = bool(strict_redis_lock)
        self._job_ttl_seconds = max(60, int(job_ttl_seconds))
        self._retry_count = max(0, int(retry_count))
        self._retry_backoff_seconds = max(0, int(retry_backoff_seconds))
        self._max_queue_size = max(1, int(max_queue_size))
        if redis_url:
            try:
                redis_module = importlib.import_module("redis")
                Redis = getattr(redis_module, "Redis")
                self._redis = Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("TrainingJobManager Redis ativo.")
            except Exception as exc:
                if self._strict_redis_lock:
                    raise RuntimeError(
                        f"Falha ao inicializar Redis para lock de treino: {exc}"
                    ) from exc
                logger.warning("Falha ao inicializar Redis para treino: %s", exc)
                self._redis = None

    def _lock_key(self, lottery: str) -> str:
        """Deriva chave de lock distribuído por modalidade."""
        return f"{self._redis_prefix}:train:lock:{lottery}"

    def _cooldown_key(self, lottery: str) -> str:
        """Deriva chave de cooldown por modalidade."""
        return f"{self._redis_prefix}:train:cooldown:{lottery}"

    def remaining_cooldown(self, lottery: str, cooldown_seconds: int) -> int:
        """Informa janela restante de cooldown antes de aceitar novo treino."""
        if cooldown_seconds <= 0:
            return 0
        if self._redis is not None:
            try:
                ttl = int(self._redis.ttl(self._cooldown_key(lottery)))
                return max(0, ttl) if ttl > 0 else 0
            except Exception as exc:
                logger.warning("Falha ao consultar cooldown no Redis: %s", exc)

        with self._lock:
            last = self._last_train_by_lottery.get(lottery, 0.0)
            remaining = int(cooldown_seconds - (_now_ts() - last))
            return max(0, remaining)

    def mark_trained(self, lottery: str, cooldown_seconds: int) -> None:
        """Registra conclusão de treino e inicia bloqueio temporal da modalidade."""
        now = _now_ts()
        if self._redis is not None and cooldown_seconds > 0:
            try:
                self._redis.set(
                    self._cooldown_key(lottery), str(int(now)), ex=cooldown_seconds
                )
            except Exception as exc:
                logger.warning("Falha ao registrar cooldown no Redis: %s", exc)
        with self._lock:
            self._last_train_by_lottery[lottery] = now

    def submit(self, lottery: str, fn: Callable[[], dict[str, Any]]) -> TrainingJob:
        """Agenda job de treino com garantias de exclusão por modalidade e limite de fila."""
        self.cleanup_expired_jobs()
        with self._lock:
            queued_or_running = sum(
                1 for job in self._jobs.values() if job.status in {"queued", "running"}
            )
            if queued_or_running >= self._max_queue_size:
                raise RuntimeError(
                    f"Fila de treino lotada (max={self._max_queue_size})."
                )
            if lottery in self._running_by_lottery:
                running_job_id = self._running_by_lottery[lottery]
                return self._jobs[running_job_id]

            lock_token = uuid.uuid4().hex
            if self._redis is not None:
                try:
                    acquired = bool(
                        self._redis.set(
                            self._lock_key(lottery), lock_token, nx=True, ex=3600
                        )
                    )
                except Exception as exc:
                    logger.error("Falha ao adquirir lock de treino no Redis: %s", exc)
                    raise RuntimeError(
                        "Falha ao adquirir lock distribuido de treino."
                    ) from exc
                if not acquired:
                    raise RuntimeError(
                        f"Ja existe treino em andamento para '{lottery}'."
                    )
            else:
                lock_token = ""

            job_id = uuid.uuid4().hex
            job = TrainingJob(
                job_id=job_id,
                lottery=lottery,
                status="queued",
                created_at=_utc_now(),
                max_retries=self._retry_count,
            )
            self._jobs[job_id] = job
            self._running_by_lottery[lottery] = job_id
            self._job_lock_tokens[job_id] = lock_token

        future = self._executor.submit(self._run_job, job_id, lottery, fn)
        with self._lock:
            self._futures[job_id] = future
        return job

    def _release_distributed_lock(self, lottery: str, job_id: str) -> None:
        """Libera lock distribuído apenas quando o token ainda pertence ao job atual."""
        if self._redis is None:
            return
        token = self._job_lock_tokens.get(job_id)
        if not token:
            return
        key = self._lock_key(lottery)
        try:
            current = self._redis.get(key)
            if current == token:
                self._redis.delete(key)
        except Exception as exc:
            logger.warning("Falha ao liberar lock de treino no Redis: %s", exc)

    def _run_job(
        self, job_id: str, lottery: str, fn: Callable[[], dict[str, Any]]
    ) -> None:
        """Executa workload com retry/backoff e publica estado final consistente."""
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()
        logger.info("Job de treino iniciado job_id=%s lottery=%s", job_id, lottery)

        try:
            result: dict[str, Any] | None = None
            last_exc: Exception | None = None
            total_attempts = self._retry_count + 1
            for attempt in range(1, total_attempts + 1):
                with self._lock:
                    job.attempts = attempt
                try:
                    result = fn()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= total_attempts:
                        break
                    if self._retry_backoff_seconds > 0:
                        time.sleep(self._retry_backoff_seconds * attempt)

            with self._lock:
                if last_exc is None and result is not None:
                    job.status = "succeeded"
                    job.result = result
                    logger.info(
                        "Job de treino concluido job_id=%s lottery=%s attempts=%s",
                        job_id,
                        lottery,
                        job.attempts,
                    )
                else:
                    job.status = "failed"
                    job.error = (
                        str(last_exc)
                        if last_exc is not None
                        else "Falha no job de treino."
                    )
                    logger.exception(
                        "Job de treino falhou job_id=%s lottery=%s attempts=%s",
                        job_id,
                        lottery,
                        job.attempts,
                        exc_info=last_exc,
                    )
                job.finished_at = _utc_now()
        finally:
            self._release_distributed_lock(lottery, job_id)
            with self._lock:
                self._running_by_lottery.pop(lottery, None)
                self._job_lock_tokens.pop(job_id, None)

    def get(self, job_id: str) -> TrainingJob | None:
        """Consulta job por identificador após higienização de registros expirados."""
        self.cleanup_expired_jobs()
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup_expired_jobs(self) -> int:
        """Remove jobs finalizados que ultrapassaram o TTL de retenção."""
        now = _now_ts()
        removed = 0
        with self._lock:
            stale_job_ids: list[str] = []
            for job_id, job in self._jobs.items():
                if job.status in {"queued", "running"}:
                    continue
                if not job.finished_at:
                    continue
                try:
                    finished_ts = datetime.fromisoformat(job.finished_at).timestamp()
                except Exception:
                    finished_ts = now
                if now - finished_ts >= self._job_ttl_seconds:
                    stale_job_ids.append(job_id)

            for job_id in stale_job_ids:
                self._jobs.pop(job_id, None)
                future = self._futures.pop(job_id, None)
                if future is not None and future.done():
                    del future
                removed += 1
        return removed

    def snapshot_metrics(self) -> dict[str, Any]:
        """Extrai fotografia operacional da fila para telemetria e diagnóstico."""
        with self._lock:
            queued = sum(1 for job in self._jobs.values() if job.status == "queued")
            running = sum(1 for job in self._jobs.values() if job.status == "running")
            succeeded = sum(
                1 for job in self._jobs.values() if job.status == "succeeded"
            )
            failed = sum(1 for job in self._jobs.values() if job.status == "failed")

            durations: list[float] = []
            for job in self._jobs.values():
                if not job.started_at or not job.finished_at:
                    continue
                try:
                    started = datetime.fromisoformat(job.started_at).timestamp()
                    finished = datetime.fromisoformat(job.finished_at).timestamp()
                    if finished >= started:
                        durations.append(finished - started)
                except Exception:
                    continue

            avg_duration_seconds = (
                float(sum(durations) / len(durations)) if durations else 0.0
            )

            return {
                "queued": queued,
                "running": running,
                "succeeded": succeeded,
                "failed": failed,
                "queue_depth": queued + running,
                "running_lotteries": len(self._running_by_lottery),
                "max_queue_size": self._max_queue_size,
                "avg_job_duration_seconds": avg_duration_seconds,
            }


def _now_ts() -> float:
    """Fornece epoch atual em segundos para cálculos de cooldown/TTL."""
    import time

    return time.time()

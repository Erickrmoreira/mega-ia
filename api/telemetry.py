import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _EndpointMetrics:
    """Acumula contadores e distribuição de latência para um endpoint lógico."""

    count: int = 0
    error_count: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def observe(self, elapsed_ms: float, is_error: bool) -> None:
        self.count += 1
        if is_error:
            self.error_count += 1
        self.latency_ms.append(float(elapsed_ms))
        if len(self.latency_ms) > 5000:
            self.latency_ms = self.latency_ms[-5000:]

    def snapshot(self) -> dict[str, float | int]:
        if not self.latency_ms:
            return {
                "count": self.count,
                "error_count": self.error_count,
                "error_rate": 0.0,
                "latency_avg_ms": 0.0,
                "latency_p95_ms": 0.0,
            }
        sorted_lat = sorted(self.latency_ms)
        p95_idx = min(
            len(sorted_lat) - 1, max(0, math.ceil(len(sorted_lat) * 0.95) - 1)
        )
        return {
            "count": self.count,
            "error_count": self.error_count,
            "error_rate": float(self.error_count / self.count) if self.count else 0.0,
            "latency_avg_ms": float(sum(self.latency_ms) / len(self.latency_ms)),
            "latency_p95_ms": float(sorted_lat[p95_idx]),
        }


class InMemoryTelemetry:
    """
    Mantém métricas operacionais em memória para inspeção rápida da API.

    Efeito colateral: métricas são voláteis e reiniciam com o processo.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._endpoints: dict[str, _EndpointMetrics] = defaultdict(_EndpointMetrics)
        self._generation_requests = 0
        self._generation_games = 0
        self._training_requests = 0

    def observe_http(self, path: str, status_code: int, elapsed_ms: float) -> None:
        endpoint = path or "unknown"
        with self._lock:
            self._endpoints[endpoint].observe(
                elapsed_ms=elapsed_ms, is_error=(status_code >= 400)
            )

    def observe_generation(self, total_games: int) -> None:
        with self._lock:
            self._generation_requests += 1
            self._generation_games += max(0, int(total_games))

    def observe_training(self) -> None:
        with self._lock:
            self._training_requests += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            endpoints = {k: v.snapshot() for k, v in sorted(self._endpoints.items())}
            avg_games = (
                float(self._generation_games / self._generation_requests)
                if self._generation_requests
                else 0.0
            )
            return {
                "service": "loteria_ia",
                "endpoints": endpoints,
                "generation": {
                    "requests": self._generation_requests,
                    "games_returned_total": self._generation_games,
                    "games_per_request_avg": avg_games,
                },
                "training": {
                    "requests": self._training_requests,
                },
            }

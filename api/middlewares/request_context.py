import json
import time
import uuid

from fastapi import FastAPI, Request


def register_request_context_middleware(app: FastAPI, logger) -> None:
    """
    Registra middleware HTTP de rastreabilidade por requisição.

    Responsabilidade:
    - propagar ou gerar `request_id`;
    - medir latência fim a fim;
    - emitir log estruturado com método, rota e status;
    - publicar métricas HTTP na telemetria da aplicação.
    """

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            # Mantém trilha de auditoria mesmo quando o handler falha.
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "elapsed_ms": elapsed_ms,
                    }
                )
            )
            telemetry = getattr(app.state, "telemetry", None)
            if telemetry is not None:
                telemetry.observe_http(
                    path=request.url.path,
                    status_code=status_code,
                    elapsed_ms=elapsed_ms,
                )
            if response is not None:
                response.headers["X-Request-ID"] = request_id

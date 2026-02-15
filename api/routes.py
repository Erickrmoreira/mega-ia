import time
from collections.abc import Callable
from logging import Logger
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.deps import validate_generation_cost
from core.config import Settings
from core.formatter import format_game
from core.lottery import supported_lotteries
from core.service import generate_ranked_games, get_lottery_config
from ml.train import ModelArtifactError, load_model_artifact, train_lottery_model


def build_generate_router(
    require_api_key_dep: Callable[..., None],
    settings: Settings,
    logger: Logger,
) -> APIRouter:
    """
    Define endpoints de geração ranqueada por modalidade.

    Contrato: valida custo da requisição e devolve jogos ordenados com score relativo.
    """
    router = APIRouter()

    @router.get("/generate/{lottery}")
    def generate_games_api(
        request: Request,
        lottery: str,
        n_games: int = Query(
            10, gt=0, le=100, description="Quantidade de jogos para retorno"
        ),
        candidates: int = Query(
            5000, gt=100, le=20000, description="Quantidade de candidatos"
        ),
        save_csv: bool = Query(False, description="Salvar resultado em CSV"),
        _auth: None = Depends(require_api_key_dep),
    ):
        validate_generation_cost(
            settings=settings, n_games=n_games, candidates=candidates
        )

        started = time.perf_counter()
        request_id = getattr(request.state, "request_id", "-")
        try:
            lottery_key, config = get_lottery_config(lottery)
            ranked_games, model_metrics = generate_ranked_games(
                lottery_key=lottery_key,
                config=config,
                n_games=n_games,
                candidates=candidates,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelArtifactError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Falha interna na geracao loteria=%s", lottery)
            raise HTTPException(
                status_code=500, detail="Erro interno ao gerar jogos."
            ) from exc

        ranked_games["game_formatted"] = ranked_games["game"].apply(format_game)
        ranked_games["predicted_score"] = ranked_games["predicted_score"].round(4)
        response_games = [
            {
                "game": row["game_formatted"],
                "predicted_score": float(row["predicted_score"]),
            }
            for _, row in ranked_games.iterrows()
        ]

        if save_csv:
            output_dir = request.app.state.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / f"jogos_api_{lottery_key}.csv"
            ranked_games[["game_formatted", "predicted_score"]].rename(
                columns={"game_formatted": "game"}
            ).to_csv(output, index=False)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Geracao concluida request_id=%s loteria=%s n_games=%s candidates=%s elapsed_ms=%s",
            request_id,
            lottery_key,
            n_games,
            candidates,
            elapsed_ms,
        )
        request.app.state.telemetry.observe_generation(len(response_games))
        return {
            "lottery": config["name"],
            "total_games": len(response_games),
            "games": response_games,
            "model_metrics": model_metrics,
            "score_info": {
                "score_type": "relative_ranking_signal",
                "is_probability": False,
                "note": "O score ordena candidatos e nao representa probabilidade real de acerto.",
            },
        }

    return router


def build_health_router(rate_limiter, settings: Settings) -> APIRouter:
    """
    Define endpoints de saúde e prontidão operacional.

    Contrato: readiness considera backend de rate limit e disponibilidade de artefato.
    """
    router = APIRouter()

    @router.get("/healthz")
    def health_check():
        return {"status": "ok", "service": "Loteria IA API"}

    @router.get("/readyz")
    def readiness_check(lottery: str | None = None):
        redis_ok, redis_msg = rate_limiter.healthcheck()
        if settings.require_redis_rate_limit and not redis_ok:
            raise HTTPException(
                status_code=503, detail=f"Rate limiter indisponivel: {redis_msg}"
            )

        if not lottery:
            return {"status": "ready", "rate_limiter": redis_msg}

        try:
            lottery_key, _ = get_lottery_config(lottery)
            artifact = load_model_artifact(lottery_key)
            return {
                "status": "ready",
                "lottery": lottery_key,
                "trained_at": artifact.get("trained_at"),
                "rate_limiter": redis_msg,
            }
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Modelo indisponivel para {lottery}."
            ) from exc

    return router


def build_models_router(
    require_train_api_key_dep: Callable[..., None],
    require_model_info_api_key_dep: Callable[..., None],
    settings: Settings,
) -> APIRouter:
    """
    Define endpoints de ciclo de vida de modelos (treino, status e metadados).

    Efeito colateral: submissão de treino enfileira job assíncrono no gerenciador.
    """
    router = APIRouter()

    @router.post("/models/train/{lottery}", status_code=202)
    def train_model_api(
        request: Request,
        lottery: str,
        _auth: None = Depends(require_train_api_key_dep),
    ) -> dict[str, Any]:
        try:
            lottery_key, config = get_lottery_config(lottery)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        remaining = request.app.state.train_jobs.remaining_cooldown(
            lottery_key,
            settings.train_cooldown_seconds,
        )
        if remaining > 0:
            raise HTTPException(
                status_code=429,
                detail=f"Treino em cooldown para '{lottery_key}'. Tente em {remaining}s.",
            )

        def _train_job() -> dict[str, Any]:
            artifact = train_lottery_model(lottery_key, config)
            request.app.state.train_jobs.mark_trained(
                lottery_key, settings.train_cooldown_seconds
            )
            request.app.state.telemetry.observe_training()
            return {
                "lottery": lottery_key,
                "trained_at": artifact.get("trained_at"),
                "metrics": artifact.get("metrics", {}),
                "promotion": artifact.get("promotion", {}),
            }

        try:
            job = request.app.state.train_jobs.submit(lottery_key, _train_job)
        except RuntimeError as exc:
            if "Fila de treino lotada" in str(exc):
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        detail = "Treino em andamento. Consulte /models/train/status/{job_id}."
        if job.status == "queued":
            detail = "Treino enfileirado. Consulte /models/train/status/{job_id}."
        return {
            "job_id": job.job_id,
            "lottery": lottery_key,
            "status": job.status,
            "detail": detail,
        }

    @router.get("/models/train/status/{job_id}")
    def train_status_api(
        request: Request,
        job_id: str,
        _auth: None = Depends(require_train_api_key_dep),
    ) -> dict[str, Any]:
        job = request.app.state.train_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job de treino nao encontrado.")

        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "lottery": job.lottery,
            "status": job.status,
            "attempts": job.attempts,
            "max_retries": job.max_retries,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
        if job.result is not None:
            payload["result"] = job.result
        if job.error is not None:
            payload["error"] = job.error
        return payload

    @router.get("/models/info/{lottery}")
    def model_info(
        lottery: str,
        _auth: None = Depends(require_model_info_api_key_dep),
    ):
        try:
            lottery_key, _ = get_lottery_config(lottery)
            artifact = load_model_artifact(lottery_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "lottery": lottery_key,
            "trained_at": artifact.get("trained_at"),
            "metrics": artifact.get("metrics", {}),
            "feature_columns": artifact.get("feature_columns", []),
            "promotion": artifact.get("promotion", {}),
            "score_contract": artifact.get("score_contract", {}),
        }

    @router.get("/models/supported")
    def supported_models():
        return {"lotteries": supported_lotteries()}

    return router


def build_metrics_router(
    require_metrics_api_key_dep: Callable[..., None],
    settings: Settings,
) -> APIRouter:
    """
    Define endpoint de observabilidade com autenticação e chave de habilitação.

    Contrato: responde 404 quando a exposição de métricas estiver desativada.
    """
    router = APIRouter()

    @router.get("/metrics")
    def metrics(
        request: Request,
        _auth: None = Depends(require_metrics_api_key_dep),
    ):
        if not settings.enable_metrics_endpoint:
            raise HTTPException(
                status_code=404, detail="Endpoint de metricas desabilitado."
            )
        payload = request.app.state.telemetry.snapshot()
        train_jobs = getattr(request.app.state, "train_jobs", None)
        if train_jobs is not None and hasattr(train_jobs, "snapshot_metrics"):
            payload["training_jobs"] = train_jobs.snapshot_metrics()
        return payload

    return router

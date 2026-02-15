import hmac
import json
import os
import pickle
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

from core.repository import load_data_with_cache
from core.statistics import build_context, normalize_draws
from ml.train_config import (
    _DEFAULT_ALLOWED_SKOPS_PREFIXES,
    _DEFAULT_ALLOWED_SKOPS_TYPES,
    _MODEL_LOCKS,
    _MODEL_LOCKS_GUARD,
    ALLOW_PICKLE_FALLBACK,
    APP_ENV,
    ARTIFACT_HMAC_KEY,
    FIRST_MODEL_MAX_LIFT_CV,
    FIRST_MODEL_MAX_MAE_FACTOR,
    FIRST_MODEL_MIN_BACKTEST_WINDOWS,
    FIRST_MODEL_MIN_LIFT_CI_LOW,
    FIRST_MODEL_MIN_MEAN_LIFT_VS_RANDOM,
    FIRST_MODEL_MIN_MEAN_RANK_CORRELATION,
    FIRST_MODEL_MIN_MEAN_TOP_DECILE_LIFT,
    FIRST_MODEL_MIN_TOP_DECILE_LIFT,
    MAX_LIFT_CV,
    MAX_MAE_REGRESSION,
    MIN_BACKTEST_WINDOWS,
    MIN_LIFT_CI_LOW,
    MIN_LIFT_MARGIN,
    MIN_MEAN_LIFT_VS_RANDOM,
    MIN_MEAN_RANK_CORRELATION,
    MIN_MEAN_TOP_DECILE_LIFT,
    MIN_TOP_DECILE_LIFT,
    MODEL_DIR,
    REQUIRE_ARTIFACT_HMAC,
    STRICT_ARTIFACT_TYPES,
    ModelArtifactError,
    TrainConfig,
    logger,
)
from ml.train_dataset import build_temporal_dataset as _build_temporal_dataset
from ml.train_metrics import evaluate_backtest_gate as _evaluate_backtest_gate
from ml.train_metrics import evaluate_predictions as _evaluate_predictions
from ml.train_metrics import rank_correlation as _rank_correlation
from ml.train_metrics import temporal_backtest as _temporal_backtest


def _model_path(lottery_key: str) -> Path:
    return MODEL_DIR / f"{lottery_key}_model.skops"


def _metadata_path(lottery_key: str) -> Path:
    return MODEL_DIR / f"{lottery_key}_meta.json"


def _artifact_hash_path(lottery_key: str) -> Path:
    return MODEL_DIR / f"{lottery_key}_artifact.sha256"


def _artifact_signature_path(lottery_key: str) -> Path:
    return MODEL_DIR / f"{lottery_key}_artifact.sig"


def _ensure_safe_artifact_path(path: Path) -> None:
    models_root = MODEL_DIR.resolve()
    resolved = path.resolve()
    if models_root not in resolved.parents and resolved != models_root:
        raise ModelArtifactError("Caminho de artefato fora do diretorio permitido.")
    if path.is_symlink():
        raise ModelArtifactError("Symlink de artefato nao permitido.")


def _artifact_signature(payload: bytes) -> str:
    if not ARTIFACT_HMAC_KEY:
        return ""
    return hmac.new(
        ARTIFACT_HMAC_KEY.encode("utf-8"),
        payload,
        digestmod="sha256",
    ).hexdigest()


def _model_lock(lottery_key: str) -> threading.Lock:
    with _MODEL_LOCKS_GUARD:
        if lottery_key not in _MODEL_LOCKS:
            _MODEL_LOCKS[lottery_key] = threading.Lock()
        return _MODEL_LOCKS[lottery_key]


def _metadata_payload(artifact: dict[str, Any]) -> bytes:
    return json.dumps(artifact, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _build_artifact_digest(model_bytes: bytes, metadata_bytes: bytes) -> bytes:
    return model_bytes + b"\n--META--\n" + metadata_bytes


def _encode_pairs_map(pairs: dict[tuple[int, int], int]) -> dict[str, int]:
    return {f"{int(a)},{int(b)}": int(v) for (a, b), v in pairs.items()}


def _decode_pairs_map(pairs: dict[Any, Any] | None) -> dict[tuple[int, int], int]:
    if not pairs:
        return {}
    decoded: dict[tuple[int, int], int] = {}
    for key, value in pairs.items():
        text = str(key)
        if "," not in text:
            continue
        left, right = text.split(",", 1)
        decoded[(int(left), int(right))] = int(value)
    return decoded


def _decode_int_key_map(values: dict[Any, Any] | None) -> dict[int, int]:
    if not values:
        return {}
    return {int(k): int(v) for k, v in values.items()}


def _load_model_skops(path: Path):
    def _load_pickle_fallback() -> Any:
        try:
            return pickle.loads(path.read_bytes())
        except Exception as pexc:
            raise ModelArtifactError(
                "Falha ao carregar artefato local via pickle."
            ) from pexc

    try:
        import skops.io as sio
    except Exception as exc:
        if APP_ENV == "prod":
            raise ModelArtifactError(
                "Dependencia skops nao disponivel para carregar modelo."
            ) from exc
        if not ALLOW_PICKLE_FALLBACK:
            raise ModelArtifactError(
                "Fallback pickle desativado. Defina MEGA_IA_ALLOW_PICKLE_FALLBACK=true para migracao local."
            ) from exc
        return _load_pickle_fallback()

    env_exact = os.environ.get("MEGA_IA_ALLOWED_SKOPS_TYPES", "").strip()
    env_prefixes = os.environ.get("MEGA_IA_ALLOWED_SKOPS_PREFIXES", "").strip()
    allowed_exact = {
        token.strip() for token in env_exact.split(",") if token.strip()
    } or _DEFAULT_ALLOWED_SKOPS_TYPES
    allowed_prefixes = (
        tuple(prefix.strip() for prefix in env_prefixes.split(",") if prefix.strip())
        or _DEFAULT_ALLOWED_SKOPS_PREFIXES
    )

    try:
        untrusted = set(sio.get_untrusted_types(file=str(path)))
    except Exception as exc:
        if APP_ENV == "prod":
            raise ModelArtifactError("Falha ao validar artefato com skops.") from exc
        if not ALLOW_PICKLE_FALLBACK:
            raise ModelArtifactError(
                "Artefato invalido para skops e fallback pickle desativado."
            ) from exc
        logger.warning(
            "Artefato nao reconhecido como skops, usando fallback pickle em dev."
        )
        return _load_pickle_fallback()
    unknown_types = sorted(
        token
        for token in untrusted
        if token not in allowed_exact
        and not any(token.startswith(prefix) for prefix in allowed_prefixes)
    )
    if unknown_types and STRICT_ARTIFACT_TYPES:
        raise ModelArtifactError(
            "Tipos nao permitidos no artefato: " + ", ".join(unknown_types[:10])
        )
    if unknown_types:
        logger.warning(
            "Tipos nao permitidos detectados em modo permissivo: %s", unknown_types
        )
    try:
        return sio.load(str(path), trusted=sorted(untrusted))
    except Exception as exc:
        if APP_ENV == "prod":
            raise ModelArtifactError("Falha ao carregar artefato com skops.") from exc
        if not ALLOW_PICKLE_FALLBACK:
            raise ModelArtifactError(
                "Falha ao carregar via skops e fallback pickle desativado."
            ) from exc
        logger.warning(
            "Falha ao carregar artefato via skops, usando fallback pickle em dev."
        )
        return _load_pickle_fallback()


def _save_model_skops(model) -> bytes:
    try:
        import skops.io as sio
    except Exception as exc:
        if APP_ENV == "prod":
            raise ModelArtifactError(
                "Dependencia skops nao disponivel para salvar modelo."
            ) from exc
        if not ALLOW_PICKLE_FALLBACK:
            raise ModelArtifactError(
                "Dependencia skops indisponivel e fallback pickle desativado."
            ) from exc
        return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    with tempfile.NamedTemporaryFile(suffix=".skops", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sio.dump(model, str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def train_lottery_model(
    lottery_key: str,
    config: dict[str, Any],
    cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    """
    Executa ciclo completo de treino, validação de gates e promoção de artefato.

    Efeito colateral: persiste modelo/metadados assinados no diretório de artefatos.
    """
    cfg = cfg or TrainConfig()
    min_num, max_num = config["num_range"]
    num_count = config["num_count"]

    logger.info("Iniciando treino loteria=%s", lottery_key)
    df = load_data_with_cache(config)
    dataset = _build_temporal_dataset(df, num_count, min_num, max_num, cfg)

    val_cut = int(dataset["time_idx"].quantile(0.7))
    test_cut = int(dataset["time_idx"].quantile(0.85))
    train_df = dataset[dataset["time_idx"] <= val_cut].copy()
    val_df = dataset[
        (dataset["time_idx"] > val_cut) & (dataset["time_idx"] <= test_cut)
    ].copy()
    test_df = dataset[dataset["time_idx"] > test_cut].copy()
    if val_df.empty:
        val_df = train_df.tail(max(1, len(train_df) // 6)).copy()
    if test_df.empty:
        test_df = val_df.tail(max(1, len(val_df) // 2)).copy()

    feature_columns = [
        column
        for column in dataset.columns
        if column not in ("time_idx", "target_overlap")
    ]
    x_train = train_df[feature_columns]
    y_train = train_df["target_overlap"]
    x_val = val_df[feature_columns]
    y_val = val_df["target_overlap"]
    x_test = test_df[feature_columns]
    y_test = test_df["target_overlap"]

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
    )
    model.fit(x_train, y_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_val, y_val)
    pred_test_cal = calibrator.predict(pred_test)
    baseline_pred = [float(y_train.mean())] * len(y_test)
    eval_metrics = _evaluate_predictions(y_test, list(pred_test_cal), baseline_pred)
    eval_metrics["rank_correlation"] = _rank_correlation(
        list(pred_test_cal), list(y_test.values)
    )
    backtest = _temporal_backtest(dataset, feature_columns, cfg)

    full_draws: list[tuple[int, ...]] = normalize_draws(
        df.itertuples(index=False, name=None)
    )
    universe = list(range(min_num, max_num + 1))
    context = build_context(full_draws, universe)

    metrics = {
        **eval_metrics,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "dataset_rows": int(len(dataset)),
        "backtest": backtest,
    }

    score_contract = {
        "score_type": "relative_ranking_signal",
        "is_probability": False,
        "target": "expected_overlap_with_next_draw",
        "notes": "Score serve apenas para ordenar jogos candidatos dentro da mesma geracao.",
        "ensemble": {
            "enabled": True,
            "components": ["ml_score", "diversity_component", "coverage_component"],
            "weights_env": [
                "MEGA_IA_ENSEMBLE_W_ML",
                "MEGA_IA_ENSEMBLE_W_DIVERSITY",
                "MEGA_IA_ENSEMBLE_W_COVERAGE",
            ],
        },
    }

    artifact = {
        "lottery_key": lottery_key,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "context": {
            "hot": context["hot"],
            "warm": context["warm"],
            "cold": context["cold"],
            "repeated": {str(int(k)): int(v) for k, v in context["repeated"].items()},
            "pairs": _encode_pairs_map(context["pairs"]),
        },
        "promotion": {
            "promoted": True,
            "reason": "first_model",
        },
        "score_contract": score_contract,
        "calibration": {
            "method": "isotonic_regression",
            "fitted_on": "validation_split",
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(MODEL_DIR, 0o700)
    except Exception:
        pass

    existing_artifact: dict[str, Any] | None = None
    try:
        existing_artifact = load_model_artifact(lottery_key)
    except ModelArtifactError:
        existing_artifact = None

    if existing_artifact:
        prev_mae = float(existing_artifact.get("metrics", {}).get("mae", float("inf")))
        curr_mae = float(artifact["metrics"]["mae"])
        allowed_max = prev_mae * (1 + MAX_MAE_REGRESSION)
        curr_lift = float(artifact["metrics"].get("top_decile_lift", 1.0))
        backtest_ok, backtest_gate = _evaluate_backtest_gate(
            backtest=backtest,
            min_mean_lift=MIN_MEAN_TOP_DECILE_LIFT,
            min_ci_low=MIN_LIFT_CI_LOW,
            max_lift_cv=MAX_LIFT_CV,
            min_windows=MIN_BACKTEST_WINDOWS,
            min_mean_rank_correlation=MIN_MEAN_RANK_CORRELATION,
            min_mean_lift_vs_random=MIN_MEAN_LIFT_VS_RANDOM,
        )
        if (
            (curr_mae > allowed_max)
            or (curr_lift < MIN_TOP_DECILE_LIFT)
            or (curr_lift < 1.0 + MIN_LIFT_MARGIN)
            or (not backtest_ok)
        ):
            logger.warning(
                "Treino rejeitado loteria=%s candidate_mae=%.4f allowed_max=%.4f candidate_lift=%.4f",
                lottery_key,
                curr_mae,
                allowed_max,
                curr_lift,
            )
            result = deepcopy(existing_artifact)
            result["promotion"] = {
                "promoted": False,
                "reason": "rejected_gate",
                "previous_mae": prev_mae,
                "candidate_mae": curr_mae,
                "allowed_max_mae": allowed_max,
                "candidate_top_decile_lift": curr_lift,
                "required_min_top_decile_lift": MIN_TOP_DECILE_LIFT,
                "required_min_lift_margin": MIN_LIFT_MARGIN,
                "backtest_gate": backtest_gate,
            }
            return result
        artifact["promotion"] = {
            "promoted": True,
            "reason": "accepted_vs_previous",
            "previous_mae": prev_mae,
            "candidate_mae": curr_mae,
            "allowed_max_mae": allowed_max,
            "candidate_top_decile_lift": curr_lift,
            "required_min_top_decile_lift": MIN_TOP_DECILE_LIFT,
            "required_min_lift_margin": MIN_LIFT_MARGIN,
            "backtest_gate": backtest_gate,
        }
    else:
        curr_mae = float(artifact["metrics"]["mae"])
        baseline_mae = float(artifact["metrics"]["baseline_mae"])
        curr_lift = float(artifact["metrics"].get("top_decile_lift", 1.0))
        allowed_first_mae = baseline_mae * FIRST_MODEL_MAX_MAE_FACTOR
        backtest_ok, backtest_gate = _evaluate_backtest_gate(
            backtest=backtest,
            min_mean_lift=FIRST_MODEL_MIN_MEAN_TOP_DECILE_LIFT,
            min_ci_low=FIRST_MODEL_MIN_LIFT_CI_LOW,
            max_lift_cv=FIRST_MODEL_MAX_LIFT_CV,
            min_windows=FIRST_MODEL_MIN_BACKTEST_WINDOWS,
            min_mean_rank_correlation=FIRST_MODEL_MIN_MEAN_RANK_CORRELATION,
            min_mean_lift_vs_random=FIRST_MODEL_MIN_MEAN_LIFT_VS_RANDOM,
        )
        if (
            (curr_mae > allowed_first_mae)
            or (curr_lift < FIRST_MODEL_MIN_TOP_DECILE_LIFT)
            or (not backtest_ok)
        ):
            logger.warning(
                "Primeiro modelo rejeitado loteria=%s candidate_mae=%.4f allowed_max=%.4f candidate_lift=%.4f",
                lottery_key,
                curr_mae,
                allowed_first_mae,
                curr_lift,
            )
            raise ModelArtifactError(
                "Primeiro modelo rejeitado por gate minimo: "
                f"candidate_mae={curr_mae:.4f}, allowed_first_mae={allowed_first_mae:.4f}, "
                f"candidate_top_decile_lift={curr_lift:.4f}, "
                f"required_min_top_decile_lift={FIRST_MODEL_MIN_TOP_DECILE_LIFT:.4f}, "
                f"backtest_gate={backtest_gate}"
            )
        artifact["promotion"] = {
            "promoted": True,
            "reason": "accepted_first_model",
            "candidate_mae": curr_mae,
            "allowed_first_mae": allowed_first_mae,
            "candidate_top_decile_lift": curr_lift,
            "required_min_top_decile_lift": FIRST_MODEL_MIN_TOP_DECILE_LIFT,
            "backtest_gate": backtest_gate,
        }

    model_path = _model_path(lottery_key)
    metadata_path = _metadata_path(lottery_key)
    artifact_hash_path = _artifact_hash_path(lottery_key)
    artifact_sig_path = _artifact_signature_path(lottery_key)

    model_payload = _save_model_skops({"model": model, "calibrator": calibrator})
    metadata_payload = _metadata_payload(artifact)
    digest_payload = _build_artifact_digest(model_payload, metadata_payload)
    digest = sha256(digest_payload).hexdigest()

    if REQUIRE_ARTIFACT_HMAC and not ARTIFACT_HMAC_KEY:
        raise ModelArtifactError(
            "Assinatura HMAC obrigatoria, mas MEGA_IA_ARTIFACT_HMAC_KEY nao foi configurada."
        )
    signature = _artifact_signature(digest_payload)

    tmp_model = model_path.with_suffix(".skops.tmp")
    tmp_meta = metadata_path.with_suffix(".json.tmp")
    tmp_hash = artifact_hash_path.with_suffix(".sha256.tmp")
    tmp_sig = artifact_sig_path.with_suffix(".sig.tmp")

    with _model_lock(lottery_key):
        for path in (model_path, metadata_path, artifact_hash_path, artifact_sig_path):
            _ensure_safe_artifact_path(path)

        tmp_model.write_bytes(model_payload)
        tmp_meta.write_bytes(metadata_payload)
        tmp_hash.write_text(digest, encoding="utf-8")
        tmp_sig.write_text(signature, encoding="utf-8")

        tmp_model.replace(model_path)
        tmp_meta.replace(metadata_path)
        tmp_hash.replace(artifact_hash_path)
        tmp_sig.replace(artifact_sig_path)

        try:
            os.chmod(model_path, 0o600)
            os.chmod(metadata_path, 0o600)
            os.chmod(artifact_hash_path, 0o600)
            os.chmod(artifact_sig_path, 0o600)
        except Exception:
            pass

    logger.info(
        "Treino promovido loteria=%s mae=%.4f baseline_mae=%.4f top_decile_lift=%.4f",
        lottery_key,
        float(metrics["mae"]),
        float(metrics["baseline_mae"]),
        float(metrics.get("top_decile_lift", 1.0)),
    )
    return {
        **artifact,
        "model": model,
        "calibrator": calibrator,
    }


def load_model_artifact(lottery_key: str) -> dict[str, Any]:
    """
    Carrega artefato promovido com validação de integridade, assinatura e tipos confiáveis.

    Contrato: falha explicitamente quando segurança/integridade não for atendida.
    """
    model_path = _model_path(lottery_key)
    metadata_path = _metadata_path(lottery_key)
    hash_path = _artifact_hash_path(lottery_key)
    sig_path = _artifact_signature_path(lottery_key)

    if not model_path.exists() or not metadata_path.exists():
        raise ModelArtifactError(
            f"Modelo nao encontrado para '{lottery_key}'. Treine antes via endpoint de treino."
        )
    if not hash_path.exists():
        raise ModelArtifactError(f"Hash do modelo ausente para '{lottery_key}'.")
    if REQUIRE_ARTIFACT_HMAC and not ARTIFACT_HMAC_KEY:
        raise ModelArtifactError("Assinatura HMAC obrigatoria sem chave configurada.")
    if ARTIFACT_HMAC_KEY and not sig_path.exists():
        raise ModelArtifactError(f"Assinatura do modelo ausente para '{lottery_key}'.")

    with _model_lock(lottery_key):
        for path in (model_path, metadata_path, hash_path):
            _ensure_safe_artifact_path(path)
        if sig_path.exists():
            _ensure_safe_artifact_path(sig_path)

        model_payload = model_path.read_bytes()
        metadata_payload = metadata_path.read_bytes()
        digest_payload = _build_artifact_digest(model_payload, metadata_payload)

        expected = hash_path.read_text(encoding="utf-8").strip().lower()
        actual = sha256(digest_payload).hexdigest().lower()
        if actual != expected:
            raise ModelArtifactError(
                f"Integridade do modelo '{lottery_key}' invalida (hash divergente)."
            )

        if ARTIFACT_HMAC_KEY:
            expected_sig = sig_path.read_text(encoding="utf-8").strip().lower()
            actual_sig = _artifact_signature(digest_payload).lower()
            if not hmac.compare_digest(expected_sig, actual_sig):
                raise ModelArtifactError(
                    f"Assinatura do modelo '{lottery_key}' invalida."
                )

        try:
            metadata = json.loads(metadata_payload.decode("utf-8"))
        except Exception as exc:
            raise ModelArtifactError(
                f"Metadata do modelo '{lottery_key}' invalida."
            ) from exc

        packed = _load_model_skops(model_path)
        model = (
            packed["model"]
            if isinstance(packed, dict) and "model" in packed
            else packed
        )
        calibrator = (
            packed["calibrator"]
            if isinstance(packed, dict) and "calibrator" in packed
            else None
        )
        meta_context = metadata.get("context", {})
        if isinstance(meta_context, dict):
            metadata["context"] = {
                **meta_context,
                "repeated": _decode_int_key_map(meta_context.get("repeated")),
                "pairs": _decode_pairs_map(meta_context.get("pairs")),
            }

        return {
            **metadata,
            "model": model,
            "calibrator": calibrator,
        }


__all__ = [
    "ModelArtifactError",
    "TrainConfig",
    "_temporal_backtest",
    "load_model_artifact",
    "train_lottery_model",
]

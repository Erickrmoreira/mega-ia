import math
import random
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ml.train_config import REQUIRE_BACKTEST_GATE, TrainConfig


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: list[float],
    baseline_pred: list[float],
) -> dict[str, float]:
    """
    Consolida métricas de erro e de qualidade de ranking em um único payload.

    Contrato: retorna MAE/R2 do modelo e baseline, além de lift no top decil.
    """
    eval_df = pd.DataFrame({"y": y_true.values, "pred": y_pred})
    top_k = max(1, int(len(eval_df) * 0.1))
    top_decile_mean = float(eval_df.nlargest(top_k, "pred")["y"].mean())
    baseline_mean = float(eval_df["y"].mean())
    top_decile_lift = (
        float(top_decile_mean / baseline_mean) if baseline_mean > 0 else 1.0
    )
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(eval_df) > 1 else 0.0,
        "baseline_mae": float(mean_absolute_error(y_true, baseline_pred)),
        "baseline_r2": (
            float(r2_score(y_true, baseline_pred)) if len(eval_df) > 1 else 0.0
        ),
        "top_decile_mean_overlap": top_decile_mean,
        "baseline_mean_overlap": baseline_mean,
        "top_decile_lift": top_decile_lift,
    }


def temporal_backtest(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    cfg: TrainConfig,
) -> dict[str, Any]:
    """
    Executa validação walk-forward temporal para medir robustez fora da amostra.

    Contrato: retorna métricas por janela e agregadas usadas no gate de promoção
    de artefatos (lift, variabilidade e correlação ordinal).
    """
    times = sorted(int(t) for t in dataset["time_idx"].unique())
    if len(times) < 12:
        return {"enabled": False, "reason": "insufficient_time_points"}

    train_min = max(8, int(len(times) * 0.45))
    remaining = max(1, len(times) - train_min - 1)
    max_windows = max(1, cfg.backtest_windows)
    step = max(1, remaining // max_windows)
    windows: list[dict[str, Any]] = []

    for split_pos in range(train_min, len(times) - 1, step):
        train_cut = times[split_pos]
        test_end_idx = min(len(times) - 1, split_pos + step)
        test_end = times[test_end_idx]
        train_df = dataset[dataset["time_idx"] <= train_cut]
        test_df = dataset[
            (dataset["time_idx"] > train_cut) & (dataset["time_idx"] <= test_end)
        ]
        if train_df.empty or test_df.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
        )
        x_train = train_df[feature_columns]
        y_train = train_df["target_overlap"]
        x_test = test_df[feature_columns]
        y_test = test_df["target_overlap"]
        model.fit(x_train, y_train)
        pred_test = model.predict(x_test)
        baseline_pred = [float(y_train.mean())] * len(y_test)
        metrics = evaluate_predictions(y_test, pred_test, baseline_pred)
        rng = random.Random(cfg.random_state + int(split_pos))
        random_pred = [rng.random() for _ in range(len(y_test))]
        random_metrics = evaluate_predictions(y_test, random_pred, baseline_pred)
        random_lift = float(random_metrics.get("top_decile_lift", 1.0))
        lift_vs_random = float(metrics["top_decile_lift"] / max(random_lift, 1e-9))
        rank_corr = rank_correlation(list(pred_test), list(y_test.values))
        windows.append(
            {
                "train_until_time_idx": int(train_cut),
                "test_until_time_idx": int(test_end),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "rank_correlation": rank_corr,
                "random_top_decile_lift": random_lift,
                "lift_vs_random": lift_vs_random,
                **metrics,
            }
        )
        if len(windows) >= max_windows:
            break

    if not windows:
        return {"enabled": False, "reason": "no_valid_windows"}

    maes = [w["mae"] for w in windows]
    lifts = [w["top_decile_lift"] for w in windows]
    ci_low, ci_high = bootstrap_mean_ci(lifts, random_state=cfg.random_state)
    lift_mean = float(sum(lifts) / len(lifts))
    lift_std = (
        float(math.sqrt(sum((x - lift_mean) ** 2 for x in lifts) / len(lifts)))
        if lifts
        else 0.0
    )
    lift_cv = float(lift_std / lift_mean) if lift_mean > 0 else 0.0
    rank_corrs = [float(w.get("rank_correlation", 0.0)) for w in windows]
    random_lifts = [float(w.get("random_top_decile_lift", 1.0)) for w in windows]
    lifts_vs_random = [float(w.get("lift_vs_random", 1.0)) for w in windows]
    return {
        "enabled": True,
        "window_count": len(windows),
        "mean_mae": float(sum(maes) / len(maes)),
        "max_mae": float(max(maes)),
        "mean_top_decile_lift": lift_mean,
        "min_top_decile_lift": float(min(lifts)),
        "top_decile_lift_ci_low": ci_low,
        "top_decile_lift_ci_high": ci_high,
        "top_decile_lift_std": lift_std,
        "top_decile_lift_cv": lift_cv,
        "mean_random_top_decile_lift": (
            float(sum(random_lifts) / len(random_lifts)) if random_lifts else 1.0
        ),
        "mean_lift_vs_random": (
            float(sum(lifts_vs_random) / len(lifts_vs_random))
            if lifts_vs_random
            else 1.0
        ),
        "mean_rank_correlation": (
            float(sum(rank_corrs) / len(rank_corrs)) if rank_corrs else 0.0
        ),
        "windows": windows,
    }


def bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Estima intervalo de confiança da média por reamostragem bootstrap."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(random_state)
    means: list[float] = []
    n = len(values)
    for _ in range(max(100, n_resamples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(sum(sample) / n))
    means.sort()
    low_idx = int((alpha / 2) * (len(means) - 1))
    high_idx = int((1 - alpha / 2) * (len(means) - 1))
    return float(means[low_idx]), float(means[high_idx])


def rank_correlation(pred: list[float], truth: list[float]) -> float:
    """
    Mede consistência ordinal entre score previsto e alvo observado via Spearman.

    Contrato: retorna correlação no intervalo [-1, 1] e aplica fallback para 0
    quando a amostra é insuficiente ou estatisticamente degenerada.
    """
    if len(pred) < 2 or len(truth) < 2:
        return 0.0
    s_pred = pd.Series(pred).rank(method="average")
    s_truth = pd.Series(truth).rank(method="average")
    corr = s_pred.corr(s_truth, method="spearman")
    if corr is None or pd.isna(corr):
        return 0.0
    return float(corr)


def evaluate_backtest_gate(
    backtest: dict[str, Any],
    min_mean_lift: float,
    min_ci_low: float,
    max_lift_cv: float,
    min_windows: int,
    min_mean_rank_correlation: float,
    min_mean_lift_vs_random: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Avalia critérios mínimos de estabilidade estatística para promoção de artefato.

    Contrato: retorna decisão booleana e payload de auditoria com limites e valores
    observados no backtest.
    """
    if not backtest.get("enabled", False):
        return (not REQUIRE_BACKTEST_GATE), {
            "enabled": False,
            "reason": backtest.get("reason", "backtest_disabled"),
            "require_backtest_gate": REQUIRE_BACKTEST_GATE,
        }

    mean_lift = float(backtest.get("mean_top_decile_lift", 1.0))
    ci_low = float(backtest.get("top_decile_lift_ci_low", mean_lift))
    lift_cv = float(backtest.get("top_decile_lift_cv", 0.0))
    windows = int(backtest.get("window_count", 0))
    mean_rank_corr = float(backtest.get("mean_rank_correlation", 0.0))
    mean_lift_vs_random = float(backtest.get("mean_lift_vs_random", 1.0))
    ok = (
        (mean_lift >= min_mean_lift)
        and (ci_low >= min_ci_low)
        and (lift_cv <= max_lift_cv)
        and (windows >= min_windows)
        and (mean_rank_corr >= min_mean_rank_correlation)
        and (mean_lift_vs_random >= min_mean_lift_vs_random)
    )
    return ok, {
        "enabled": True,
        "window_count": windows,
        "required_min_windows": min_windows,
        "mean_top_decile_lift": mean_lift,
        "required_min_mean_top_decile_lift": min_mean_lift,
        "top_decile_lift_ci_low": ci_low,
        "required_min_lift_ci_low": min_ci_low,
        "top_decile_lift_cv": lift_cv,
        "allowed_max_lift_cv": max_lift_cv,
        "mean_rank_correlation": mean_rank_corr,
        "required_min_mean_rank_correlation": min_mean_rank_correlation,
        "mean_lift_vs_random": mean_lift_vs_random,
        "required_min_mean_lift_vs_random": min_mean_lift_vs_random,
    }

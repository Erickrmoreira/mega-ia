import random
from typing import Any

import pandas as pd

from core.features import extract_features
from core.statistics import (
    hot_warm_cold,
    normalize_draws,
    pairs_frequency,
    repeated_between_contests,
)
from ml.train_config import ModelArtifactError, TrainConfig


def _random_game(
    num_count: int,
    min_num: int,
    max_num: int,
    rng: random.Random,
) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(min_num, max_num + 1), num_count)))


def build_temporal_dataset(
    df: pd.DataFrame,
    num_count: int,
    min_num: int,
    max_num: int,
    cfg: TrainConfig,
) -> pd.DataFrame:
    """
    Constrói dataset supervisionado temporal com amostras positivas e negativas difíceis.

    Contrato: preserva ordenação histórica e falha quando não há volume mínimo útil.
    """
    if len(df) <= cfg.min_history + 1:
        raise ModelArtifactError(
            f"Dados insuficientes para treino temporal. Necessario > {cfg.min_history + 1} concursos."
        )

    rng = random.Random(cfg.random_state)
    rows: list[dict[str, Any]] = []
    ordered = df.reset_index(drop=True)
    draws: list[tuple[int, ...]] = normalize_draws(
        ordered.itertuples(index=False, name=None)
    )
    universe = list(range(min_num, max_num + 1))

    time_points = list(
        range(cfg.min_history, len(ordered) - 1, max(1, cfg.time_stride))
    )
    if len(time_points) > cfg.max_time_points:
        step = len(time_points) / cfg.max_time_points
        time_points = [time_points[int(i * step)] for i in range(cfg.max_time_points)]

    for t in time_points:
        start_idx = max(0, t - cfg.history_window)
        history_draws = draws[start_idx:t]
        next_draw = set(draws[t + 1])

        hot, warm, cold = hot_warm_cold(history_draws, universe)
        repeated = repeated_between_contests(history_draws)
        pairs = pairs_frequency(history_draws)

        positive_game = tuple(sorted(draws[t]))
        sample_games = [positive_game]
        # Hard negatives elevam discriminação ao usar concursos reais próximos do contexto.
        if t > start_idx:
            history_slice = draws[max(start_idx, t - min(10, t - start_idx)) : t]
            for hist_game in history_slice[: max(1, cfg.negative_samples // 2)]:
                if hist_game != positive_game:
                    sample_games.append(tuple(sorted(hist_game)))
        for _ in range(max(1, cfg.negative_samples - (len(sample_games) - 1))):
            sample_games.append(_random_game(num_count, min_num, max_num, rng))

        for game in sample_games:
            features = extract_features(game, hot, warm, cold, pairs, repeated)
            target_overlap = len(set(game) & next_draw)
            rows.append(
                {
                    "time_idx": t,
                    "target_overlap": float(target_overlap),
                    **features,
                }
            )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ModelArtifactError("Dataset temporal ficou vazio.")
    return dataset

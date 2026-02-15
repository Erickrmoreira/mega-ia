import os
from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from core.features_engine import _extract_features_fast

Game = tuple[int, ...]
ENSEMBLE_W_ML = float(os.environ.get("MEGA_IA_ENSEMBLE_W_ML", "0.70"))
ENSEMBLE_W_DIVERSITY = float(os.environ.get("MEGA_IA_ENSEMBLE_W_DIVERSITY", "0.20"))
ENSEMBLE_W_COVERAGE = float(os.environ.get("MEGA_IA_ENSEMBLE_W_COVERAGE", "0.10"))
LOTOFACIL_MAX_SEQUENCE_LEN = int(
    os.environ.get("MEGA_IA_LOTOFACIL_MAX_SEQUENCE_LEN", "6")
)
LOTOFACIL_SEQUENCE_PENALTY = float(
    os.environ.get("MEGA_IA_LOTOFACIL_SEQUENCE_PENALTY", "0.08")
)


class PredictModel(Protocol):
    """Contrato mínimo de modelo compatível com etapa de inferência de ranking."""

    def predict(self, x: pd.DataFrame): ...


class CalibratorModel(Protocol):
    """Contrato de calibrador aplicado após predição bruta do modelo."""

    def predict(self, x): ...


def _minmax_norm(series: pd.Series) -> pd.Series:
    """
    Normaliza score por lote para intervalo [0, 1] antes da composição do ensemble.

    Trade-off: usa normalização local da amostra atual (não global), priorizando
    ordenação relativa entre candidatos e protegendo contra divisão por zero.
    """
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v <= min_v:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def generate_games_with_model(
    candidate_games: Sequence[Game],
    model: PredictModel,
    hot: Sequence[int],
    warm: Sequence[int],
    cold: Sequence[int],
    pairs: Mapping[tuple[int, int], int],
    repeated: Mapping[int, int],
    top_n: int = 100,
    feature_columns: Sequence[str] | None = None,
    calibrator: CalibratorModel | None = None,
) -> pd.DataFrame:
    """
    Ordena candidatos combinando score do modelo e componentes heurísticos de diversidade.

    Contrato: retorna DataFrame com colunas `game` e `predicted_score`.
    """
    hot_set = {int(n) for n in hot}
    warm_set = {int(n) for n in warm}
    cold_set = {int(n) for n in cold}
    rows: list[dict] = []

    for game in candidate_games:
        parsed_game = sorted(int(n) for n in game)
        features = _extract_features_fast(
            parsed_game=parsed_game,
            hot_set=hot_set,
            warm_set=warm_set,
            cold_set=cold_set,
            pairs_freq=pairs,
            repeated_freq=repeated,
        )
        rows.append({"game": game, **features})

    df = pd.DataFrame(rows)
    x = df.drop(columns=["game"])
    if feature_columns is not None:
        for col in feature_columns:
            if col not in x.columns:
                x[col] = 0
        x = x[list(feature_columns)]
    raw_pred = model.predict(x)
    if calibrator is not None:
        raw_pred = calibrator.predict(raw_pred)
    df["ml_score_raw"] = raw_pred
    ml_norm = _minmax_norm(df["ml_score_raw"])

    game_len = max(1.0, float(len(candidate_games[0]) if candidate_games else 1))
    coverage_raw = (
        df["hot_count"] + 0.6 * df["warm_count"] - 0.15 * df["cold_count"]
    ) / game_len
    coverage_norm = _minmax_norm(coverage_raw)

    diversity_raw = -(
        df["pair_score"] + 0.5 * df["repeated_score"] + 0.1 * df["max_sequence_len"]
    )
    diversity_norm = _minmax_norm(diversity_raw)

    df["predicted_score"] = (
        ENSEMBLE_W_ML * ml_norm
        + ENSEMBLE_W_DIVERSITY * diversity_norm
        + ENSEMBLE_W_COVERAGE * coverage_norm
    )

    # Penalização protege o ranking de concentração excessiva em sequências longas.
    if game_len >= 15:
        allowed_df = df[df["max_sequence_len"] <= float(LOTOFACIL_MAX_SEQUENCE_LEN)]
        if len(allowed_df) >= top_n:
            df = allowed_df
        else:
            excess = (df["max_sequence_len"] - float(LOTOFACIL_MAX_SEQUENCE_LEN)).clip(
                lower=0.0
            )
            df["predicted_score"] = (
                df["predicted_score"] - (excess * LOTOFACIL_SEQUENCE_PENALTY)
            ).clip(lower=0.0)

    df = df.sort_values(by="predicted_score", ascending=False)
    selected = df.head(top_n)

    return selected[["game", "predicted_score"]]

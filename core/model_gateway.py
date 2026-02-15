from collections.abc import Sequence
from typing import Any

import pandas as pd

from core.ports import Game, RankerGateway
from ml.infer import generate_games_with_model
from ml.train import load_model_artifact


class MlRankerGateway(RankerGateway):
    """Implementa porta de ranking usando artefato treinado e pipeline de inferência atual."""

    def load_artifact(self, lottery_key: str) -> dict[str, Any]:
        return load_model_artifact(lottery_key)

    def rank_candidates(
        self,
        candidate_games: Sequence[Game],
        artifact: dict[str, Any],
        top_n: int,
    ) -> pd.DataFrame:
        context = artifact["context"]
        return generate_games_with_model(
            candidate_games=candidate_games,
            model=artifact["model"],
            hot=context["hot"],
            warm=context["warm"],
            cold=context["cold"],
            pairs=context["pairs"],
            repeated=context["repeated"],
            top_n=top_n,
            feature_columns=artifact.get("feature_columns"),
            calibrator=artifact.get("calibrator"),
        )


_DEFAULT_GATEWAY = MlRankerGateway()


def load_ranker_artifact(lottery_key: str) -> dict[str, Any]:
    """Função de compatibilidade para carga de artefato via gateway padrão."""
    return _DEFAULT_GATEWAY.load_artifact(lottery_key)


def rank_candidate_games(
    candidate_games: Sequence[Game],
    artifact: dict[str, Any],
    top_n: int,
) -> pd.DataFrame:
    """Função de compatibilidade para ranking via gateway padrão."""
    return _DEFAULT_GATEWAY.rank_candidates(
        candidate_games=list(candidate_games),
        artifact=artifact,
        top_n=top_n,
    )

from typing import Any, Protocol

import pandas as pd

Game = tuple[int, ...]


class RankerGateway(Protocol):
    """Porta de infraestrutura para integração de ranking com o domínio."""

    def load_artifact(self, lottery_key: str) -> dict[str, Any]:
        """Carrega estado de modelo necessário para inferência da modalidade."""

    def rank_candidates(
        self, candidate_games: list[Game], artifact: dict[str, Any], top_n: int
    ) -> pd.DataFrame:
        """Ordena candidatos e retorna contrato tabular `game` + `predicted_score`."""

from typing import Any

import pandas as pd

from core.generator import generate_games_random
from core.lottery import LOTTERIES, resolve_lottery_key
from core.model_gateway import MlRankerGateway
from core.ports import RankerGateway


def get_lottery_config(lottery: str) -> tuple[str, dict[str, Any]]:
    """
    Resolve modalidade informada e retorna configuração oficial usada pelo domínio.

    Contrato: lança `ValueError` quando a modalidade não for suportada.
    """
    lottery_key = resolve_lottery_key(lottery)
    if lottery_key not in LOTTERIES:
        raise ValueError(f"Loteria nao suportada: {lottery}")
    return lottery_key, LOTTERIES[lottery_key]


def generate_ranked_games(
    lottery_key: str,
    config: dict[str, Any],
    n_games: int,
    candidates: int,
    ranker_gateway: RankerGateway | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gera candidatos aleatorios e delega ranking ao gateway configurado.

    Trade-off: manter gateway injetavel reduz acoplamento ao stack ML concreto
    sem alterar o contrato externo do servico.
    """
    min_num, max_num = config["num_range"]
    num_count = config["num_count"]
    gateway = ranker_gateway or MlRankerGateway()

    artifact = gateway.load_artifact(lottery_key)

    candidate_games = generate_games_random(
        n_games=candidates,
        min_num=min_num,
        max_num=max_num,
        num_count=num_count,
    )

    ranked = gateway.rank_candidates(
        candidate_games=candidate_games,
        artifact=artifact,
        top_n=n_games,
    )
    return ranked, artifact.get("metrics", {})

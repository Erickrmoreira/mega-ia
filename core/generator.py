import logging
import random
from collections import Counter
from collections.abc import Mapping
from typing import Any

from core.strategies import Strategy
from core.validator import valid_game

Game = tuple[int, ...]
logger = logging.getLogger("mega_ia.generator")


def generate_games(
    n_games: int,
    hot: list[int],
    warm: list[int],
    cold: list[int],
    repeated: Mapping[int, int],
    pairs: Mapping[tuple[int, int], int],
    sequences: Any,
    strategies: list[Strategy],
    num_count: int,
    min_num: int,
    max_num: int,
) -> list[Game]:
    """
    Constrói jogos com restrições de diversidade, similaridade e saturação de dezenas.

    Efeito colateral: pode retornar menos jogos quando o espaço viável é exaurido.
    """
    n_games = min(n_games, 100)
    games_set: set[Game] = set()
    games_list: list[Game] = []
    game_sets: list[set[int]] = []
    number_index: dict[int, set[int]] = {}
    usage_counter: Counter[int] = Counter()

    if num_count >= 15:
        max_usage = n_games
        max_similar = int(num_count * 0.9)
        max_attempts = n_games * 500
    else:
        max_usage = max(1, int(n_games * 0.25))
        max_similar = (num_count * 2) // 3
        max_attempts = n_games * 100

    attempts = 0
    strategy_errors = 0
    max_strategy_errors = max(10, n_games * 2)
    strategy_error_counter: Counter[str] = Counter()

    while len(games_set) < n_games and attempts < max_attempts:
        attempts += 1
        strat = random.choice(strategies)

        try:
            raw_game = strat(hot, warm, cold, repeated, pairs, sequences)
        except Exception as exc:
            strategy_errors += 1
            strategy_name = getattr(strat, "__name__", strat.__class__.__name__)
            strategy_error_counter[strategy_name] += 1
            logger.warning(
                "Falha em estrategia name=%s erro=%s erro_count=%s/%s",
                strategy_name,
                exc,
                strategy_errors,
                max_strategy_errors,
            )
            if strategy_errors >= max_strategy_errors:
                raise RuntimeError(
                    f"Excesso de falhas em estrategias: {dict(strategy_error_counter)}"
                ) from exc
            continue

        game = tuple(sorted(int(x) for x in raw_game))

        if not valid_game(list(game), num_count, min_num, max_num):
            continue

        if any(usage_counter[n] >= max_usage for n in game):
            continue

        if game in games_set:
            continue

        candidate_set = set(game)
        candidate_indices: set[int] = set()
        for n in candidate_set:
            candidate_indices.update(number_index.get(n, set()))

        too_similar = False
        for idx in candidate_indices:
            if len(candidate_set & game_sets[idx]) >= max_similar:
                too_similar = True
                break
        if too_similar:
            continue

        games_set.add(game)
        game_idx = len(games_list)
        games_list.append(game)
        game_sets.append(candidate_set)
        for n in candidate_set:
            number_index.setdefault(n, set()).add(game_idx)
        for n in game:
            usage_counter[n] += 1

    if len(games_set) < n_games:
        logger.warning(
            "Apenas %s de %s jogos foram gerados (restricoes matematicas atingidas)",
            len(games_set),
            n_games,
        )

    return games_list


def generate_games_random(
    n_games: int,
    min_num: int,
    max_num: int,
    num_count: int,
) -> list[Game]:
    """
    Produz candidatos aleatórios únicos respeitando cardinalidade e faixa da modalidade.

    Contrato: pode retornar menos itens quando não há unicidade suficiente no limite de tentativas.
    """
    games: set[Game] = set()
    max_attempts = max(n_games * 20, 1000)
    attempts = 0

    while len(games) < n_games and attempts < max_attempts:
        attempts += 1
        game = tuple(sorted(random.sample(range(min_num, max_num + 1), num_count)))
        if valid_game(list(game), num_count, min_num, max_num):
            games.add(game)

    if len(games) < n_games:
        logger.warning(
            "Nao foi possivel gerar %s jogos unicos; retornando %s.",
            n_games,
            len(games),
        )

    return list(games)

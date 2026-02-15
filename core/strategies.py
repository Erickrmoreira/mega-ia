import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

GameLike = Sequence[int]
PairFreq = Mapping[tuple[int, int], int]
RepeatedFreq = Mapping[int, int]
Strategy = Callable[
    [list[int], list[int], list[int], RepeatedFreq, PairFreq, Any],
    GameLike,
]


def safe_sample(
    pool: Sequence[int] | Mapping[Any, Any] | set[int],
    k: int,
    min_num: int = 1,
    max_num: int = 60,
) -> list[int]:
    """Amostra `k` valores de um pool heterogeneo com fallback seguro."""
    if isinstance(pool, Mapping):
        values = [int(v) for v in pool.keys()]
    elif isinstance(pool, set):
        values = [int(v) for v in pool]
    else:
        values = [int(v) for v in pool]

    if len(values) < k:
        values = list(range(min_num, max_num + 1))

    return random.sample(values, k)


def make_strategy_mixed(k: int, min_num: int, max_num: int) -> Strategy:
    """Balanceia quentes/mornos/frios para aumentar diversidade com viés estatistico leve."""

    def strat(
        hot: list[int],
        warm: list[int],
        cold: list[int],
        _repeated: RepeatedFreq,
        _pairs: PairFreq,
        _sequences: Any,
    ) -> list[int]:
        hot_c = k // 2
        warm_c = k // 3
        cold_c = k - hot_c - warm_c

        game: list[int] = []
        game += safe_sample(hot, hot_c, min_num, max_num)
        game += safe_sample(warm, warm_c, min_num, max_num)
        game += safe_sample(cold, cold_c, min_num, max_num)

        all_nums = list(set(hot + warm + cold)) or list(range(min_num, max_num + 1))
        while len(game) < k:
            game.append(random.choice(all_nums))

        return game[:k]

    return strat


def make_strategy_repeated(k: int, min_num: int, max_num: int) -> Strategy:
    """Prioriza dezenas com repeticao historica; tende a reduzir diversidade quando dominante."""

    def strat(
        _hot: list[int],
        _warm: list[int],
        _cold: list[int],
        repeated: RepeatedFreq,
        _pairs: PairFreq,
        _sequences: Any,
    ) -> list[int]:
        if not repeated:
            return random.sample(range(min_num, max_num + 1), k)
        return safe_sample(repeated, k, min_num, max_num)

    return strat


def make_strategy_pairs(k: int, min_num: int, max_num: int) -> Strategy:
    """Explora pares frequentes; melhora coerencia estatistica local, mas pode superconcentrar."""

    def strat(
        _hot: list[int],
        _warm: list[int],
        _cold: list[int],
        _repeated: RepeatedFreq,
        pairs: PairFreq,
        _sequences: Any,
    ) -> list[int]:
        if not pairs:
            return random.sample(range(min_num, max_num + 1), k)

        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

        nums: set[int] = set()
        num_pairs = (k + 1) // 2
        for (a, b), _ in sorted_pairs[:num_pairs]:
            nums.add(int(a))
            nums.add(int(b))

        while len(nums) < k:
            nums.add(random.randint(min_num, max_num))

        return list(nums)[:k]

    return strat


def make_strategy_sequences(k: int, min_num: int, max_num: int) -> Strategy:
    """Reaproveita sequencias vistas no historico, aceitando risco de padroes artificiais."""

    def strat(
        _hot: list[int],
        _warm: list[int],
        _cold: list[int],
        _repeated: RepeatedFreq,
        _pairs: PairFreq,
        sequences: Any,
    ) -> list[int]:
        flat: set[int] = set()

        for item in sequences:
            if isinstance(item, (list, tuple)):
                if len(item) > 0 and isinstance(item[0], (list, tuple)):
                    flat.update(int(v) for v in item[0])
                else:
                    flat.update(int(v) for v in item)
            elif isinstance(item, int):
                flat.add(item)

        values = list(flat)
        if len(values) < k:
            values.extend(range(min_num, max_num + 1))

        return random.sample(values, k)

    return strat


def make_strategy_random(min_num: int, max_num: int, k: int) -> Strategy:
    """Gera jogo totalmente aleatorio para manter exploracao e evitar colapso de diversidade."""

    def strat(
        _hot: list[int],
        _warm: list[int],
        _cold: list[int],
        _repeated: RepeatedFreq,
        _pairs: PairFreq,
        _sequences: Any,
    ) -> list[int]:
        return random.sample(range(min_num, max_num + 1), k)

    return strat


def make_strategy_interval_weighted(
    interval_weights: Mapping[int, float],
    k: int,
    min_num: int,
    max_num: int,
) -> Strategy:
    """Pondera por atraso de aparicao; favorece cobertura de dezenas frias recentes."""

    def strat(
        _hot: list[int],
        _warm: list[int],
        _cold: list[int],
        _repeated: RepeatedFreq,
        _pairs: PairFreq,
        _sequences: Any,
    ) -> list[int]:
        nums = list(interval_weights.keys())
        weights = list(interval_weights.values())

        if not weights or sum(weights) == 0:
            return random.sample(range(min_num, max_num + 1), k)

        probs = np.array(weights) / np.sum(weights)
        try:
            chosen = np.random.choice(nums, size=k, replace=False, p=probs)
            return [int(v) for v in chosen.tolist()]
        except ValueError:
            return random.sample(range(min_num, max_num + 1), k)

    return strat

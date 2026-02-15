import itertools
from collections import Counter
from collections.abc import Iterable, Sequence

Draw = tuple[int, ...]


def normalize_draws(rows: Iterable[Sequence[int]]) -> list[Draw]:
    """Converte representação de concursos para formato canônico imutável."""
    return [tuple(int(v) for v in row) for row in rows]


def frequency(draws: Sequence[Draw], universe: Sequence[int]) -> Counter[int]:
    """Computa frequência absoluta por dezena no universo de referência."""
    freq: Counter[int] = Counter()
    for draw in draws:
        freq.update(draw)
    for n in universe:
        freq.setdefault(int(n), 0)
    return freq


def hot_warm_cold(
    draws: Sequence[Draw], universe: Sequence[int]
) -> tuple[list[int], list[int], list[int]]:
    """Segmenta dezenas por tercis de frequência para uso em features e estratégia."""
    freq = frequency(draws, universe)
    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    third = max(1, len(universe) // 3)
    hot = [n for n, _ in sorted_nums[:third]]
    cold = [n for n, _ in sorted_nums[-third:]]
    warm = [int(n) for n in universe if int(n) not in hot and int(n) not in cold]
    return hot, warm, cold


def repeated_between_contests(draws: Sequence[Draw]) -> Counter[int]:
    """Mede persistência de dezenas entre concursos consecutivos."""
    repeated: Counter[int] = Counter()
    for i in range(len(draws) - 1):
        repeated.update(set(draws[i]) & set(draws[i + 1]))
    return repeated


def pairs_frequency(draws: Sequence[Draw]) -> Counter[tuple[int, int]]:
    """Mede coocorrência de pares ordenados no histórico."""
    pairs: Counter[tuple[int, int]] = Counter()
    for draw in draws:
        parsed = sorted(int(v) for v in draw)
        for pair in itertools.combinations(parsed, 2):
            pairs[pair] += 1
    return pairs


def sequence_frequency(draws: Sequence[Draw]) -> Counter[tuple[int, ...]]:
    """Mede incidência de sequências consecutivas com tamanho mínimo dois."""
    sequences: Counter[tuple[int, ...]] = Counter()
    for draw in draws:
        parsed = sorted(int(v) for v in draw)
        if not parsed:
            continue
        current = [parsed[0]]
        for i in range(1, len(parsed)):
            if parsed[i] == parsed[i - 1] + 1:
                current.append(parsed[i])
            else:
                if len(current) >= 2:
                    sequences[tuple(current)] += 1
                current = [parsed[i]]
        if len(current) >= 2:
            sequences[tuple(current)] += 1
    return sequences


def interval_weights(
    draws: Sequence[Draw], universe: Sequence[int]
) -> dict[int, float]:
    """Calcula pesos por atraso de aparição para privilegiar cobertura temporal."""
    last_seen: dict[int, int] = {}
    total_games = len(draws)

    for idx, draw in enumerate(draws):
        for num in draw:
            last_seen[int(num)] = idx

    max_interval = 0
    for n in universe:
        last_idx = last_seen.get(int(n), -1)
        interval = total_games - last_idx - 1
        max_interval = max(max_interval, interval)

    weights: dict[int, float] = {}
    for n in universe:
        n_int = int(n)
        last_idx = last_seen.get(n_int, -1)
        interval = total_games - last_idx - 1
        weights[n_int] = 1 + (interval / max_interval if max_interval > 0 else 0)
    return weights


def build_context(draws: Sequence[Draw], universe: Sequence[int]) -> dict[str, object]:
    """Consolida sinais estatísticos usados de forma compartilhada por treino e inferência."""
    hot, warm, cold = hot_warm_cold(draws, universe)
    repeated = repeated_between_contests(draws)
    pairs = pairs_frequency(draws)
    return {
        "hot": hot,
        "warm": warm,
        "cold": cold,
        "repeated": dict(repeated),
        "pairs": dict(pairs),
    }

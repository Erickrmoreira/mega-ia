from collections.abc import Mapping, Sequence


def _extract_features_fast(
    parsed_game: list[int],
    hot_set: set[int],
    warm_set: set[int],
    cold_set: set[int],
    pairs_freq: Mapping[tuple[int, int], int],
    repeated_freq: Mapping[int, int],
) -> dict[str, float]:
    """Extrai vetor de atributos com estruturas pré-computadas para reduzir custo por jogo."""
    features: dict[str, float] = {}

    features["hot_count"] = float(sum(1 for n in parsed_game if n in hot_set))
    features["warm_count"] = float(sum(1 for n in parsed_game if n in warm_set))
    features["cold_count"] = float(sum(1 for n in parsed_game if n in cold_set))

    seq_len = 1
    max_seq = 1
    num_seq = 0
    for i in range(1, len(parsed_game)):
        if parsed_game[i] == parsed_game[i - 1] + 1:
            seq_len += 1
            max_seq = max(max_seq, seq_len)
        else:
            if seq_len >= 2:
                num_seq += 1
            seq_len = 1
    if seq_len >= 2:
        num_seq += 1

    features["num_sequences"] = float(num_seq)
    features["max_sequence_len"] = float(max_seq)

    pair_score = 0
    for i in range(len(parsed_game)):
        for j in range(i + 1, len(parsed_game)):
            pair = (parsed_game[i], parsed_game[j])
            pair_score += int(pairs_freq.get(pair, 0))
    features["pair_score"] = float(pair_score)

    features["repeated_score"] = float(
        sum(int(repeated_freq.get(n, 0)) for n in parsed_game)
    )
    return features


def extract_features(
    game: Sequence[int],
    hot: Sequence[int],
    warm: Sequence[int],
    cold: Sequence[int],
    pairs_freq: Mapping[tuple[int, int], int],
    repeated_freq: Mapping[int, int],
) -> dict[str, float]:
    """Normaliza entrada e produz atributos usados por treino e inferência de ranking."""
    parsed_game = sorted(int(n) for n in game)
    return _extract_features_fast(
        parsed_game=parsed_game,
        hot_set=set(hot),
        warm_set=set(warm),
        cold_set=set(cold),
        pairs_freq=pairs_freq,
        repeated_freq=repeated_freq,
    )

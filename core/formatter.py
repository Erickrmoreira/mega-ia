from collections.abc import Sequence


def format_game(game: Sequence[int]) -> str:
    """Converte sequência de dezenas para representação textual padrão da interface."""
    return " - ".join(f"{int(n):02d}" for n in game)

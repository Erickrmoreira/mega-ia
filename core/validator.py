def valid_game(game: list[int], num_count: int, min_num: int, max_num: int) -> bool:
    """Aplica contrato mínimo de integridade do jogo antes de persistir/ranquear."""
    if len(game) != num_count or len(set(game)) != num_count:
        return False
    if any(not min_num <= n <= max_num for n in game):
        return False
    return True

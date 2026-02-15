import unicodedata

LOTTERIES = {
    "mega": {
        "name": "Mega-Sena",
        "num_count": 6,
        "num_range": (1, 60),
        "number_columns": ["n1", "n2", "n3", "n4", "n5", "n6"],
        "data_files": ["datasets/mega.csv"],
        "output_csv": "output/jogos_mega.csv",
    },
    "quina": {
        "name": "Quina",
        "num_count": 5,
        "num_range": (1, 80),
        "number_columns": ["n1", "n2", "n3", "n4", "n5"],
        "data_files": ["datasets/quina.csv"],
        "output_csv": "output/jogos_quina.csv",
    },
    "lotofacil": {
        "name": "Lotofacil",
        "num_count": 15,
        "num_range": (1, 25),
        "number_columns": [
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "n10",
            "n11",
            "n12",
            "n13",
            "n14",
            "n15",
        ],
        "data_files": ["datasets/lotofacil.csv"],
        "output_csv": "output/jogos_lotofacil.csv",
    },
}

LOTTERY_ALIASES = {
    "mega-sena": "mega",
    "megasena": "mega",
    "mega sena": "mega",
    "lotofacil": "lotofacil",
    "loto facil": "lotofacil",
}


def _normalize_key(value: str) -> str:
    """Normaliza identificador textual para resolução estável de aliases."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.strip().lower()


def resolve_lottery_key(lottery: str) -> str:
    """
    Resolve identificador externo para chave canônica de modalidade.

    Contrato: entradas acentuadas, com variação de caixa ou alias legados convergem
    para a chave interna estável usada por API, CLI e pipeline.
    """
    key = _normalize_key(lottery)
    return LOTTERY_ALIASES.get(key, key)


def supported_lotteries() -> list[str]:
    """Expõe catálogo público de modalidades aceitas pela API/CLI."""
    return sorted(set(LOTTERIES) | set(LOTTERY_ALIASES))

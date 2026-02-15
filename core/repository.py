from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from core.loader import load_lottery_data


def _files_signature(paths: list[str]) -> tuple[tuple[str, float], ...]:
    """Calcula assinatura de arquivos para invalidação determinística do cache."""
    signature: list[tuple[str, float]] = []
    for path in paths:
        p = Path(path)
        signature.append((str(p), p.stat().st_mtime if p.exists() else -1.0))
    return tuple(signature)


@lru_cache(maxsize=8)
def _cached_data(
    data_files: tuple[str, ...],
    data_signature: tuple[tuple[str, float], ...],
    num_count: int,
    min_num: int,
    max_num: int,
    expected_number_columns: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Materializa dataset consolidado em cache LRU indexado por assinatura de entrada."""
    del data_signature
    df = load_lottery_data(
        list(data_files),
        num_count=num_count,
        min_num=min_num,
        max_num=max_num,
        expected_number_columns=(
            list(expected_number_columns) if expected_number_columns else None
        ),
    )
    return df.copy()


def load_data_with_cache(config: dict[str, Any]) -> pd.DataFrame:
    """
    Entrega dataset histórico com cache invalidável por assinatura de arquivo.

    Contrato: alterações de mtime em qualquer CSV da modalidade renovam o cache
    automaticamente, evitando reprocessamento desnecessário sem servir dado stale.
    """
    min_num, max_num = config["num_range"]
    num_count = config["num_count"]
    files = tuple(config["data_files"])
    expected_number_columns = tuple(config.get("number_columns", []))
    signature = _files_signature(list(files))
    return _cached_data(
        data_files=files,
        data_signature=signature,
        num_count=num_count,
        min_num=min_num,
        max_num=max_num,
        expected_number_columns=expected_number_columns,
    )

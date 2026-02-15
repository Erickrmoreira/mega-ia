import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("mega_ia.loader")

MAX_REJECT_RATIO = 0.8
MAX_DUPLICATE_RATIO = 0.35
MIN_VALID_ROWS = 20


def _find_number_columns(columns: list[str], num_count: int) -> list[str]:
    """Identifica colunas de dezenas no schema para habilitar parsing vetorizado."""
    normalized = {c: c.strip().lower() for c in columns}
    n_cols = [c for c in columns if re.fullmatch(r"n\d+", normalized[c])]
    if len(n_cols) >= num_count:
        return sorted(n_cols, key=lambda c: int(normalized[c][1:]))[:num_count]
    return []


def _extract_numbers_from_row(row: pd.Series, num_count: int) -> list[int]:
    """Extrai dezenas de uma linha com fallback resiliente para entradas inconsistentes."""
    numeric = pd.to_numeric(row, errors="coerce").dropna().astype(int).tolist()
    if len(numeric) < num_count:
        text = " ".join(str(v) for v in row.values)
        numeric = [int(v) for v in re.findall(r"\d{1,2}", text)]
    if len(numeric) < num_count:
        return []
    return numeric[-num_count:]


def _row_sort_key(row: pd.Series, columns: list[str]) -> tuple[int, int]:
    """Prioriza ordenação temporal por data, ano ou concurso para preservar sequência histórica."""
    lowered = {c.lower(): c for c in columns}

    if "data" in lowered:
        raw = str(row.get(lowered["data"], "")).strip()
        for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                return (0, int(dt.timestamp()))
            except ValueError:
                continue

    if "ano" in lowered:
        try:
            year = int(row.get(lowered["ano"]))
            return (1, year)
        except Exception:
            pass

    if "concurso" in lowered:
        try:
            contest = int(row.get(lowered["concurso"]))
            return (2, contest)
        except Exception:
            pass

    return (3, 0)


def load_data(
    path: str,
    num_count: int,
    min_num: int = 1,
    max_num: int | None = None,
    expected_number_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carrega arquivo de concursos e normaliza para schema canônico `n1..nN`.

    Contrato: aplica validações de qualidade (rejeição, duplicidade e faixa numérica).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    raw_df = pd.read_csv(csv_path, sep=None, engine="python")
    raw_df = raw_df.rename(columns=lambda c: str(c).strip())
    number_cols = _find_number_columns(list(raw_df.columns), num_count)
    if expected_number_columns:
        expected = [c.strip().lower() for c in expected_number_columns]
        if not all(col in [c.lower() for c in raw_df.columns] for col in expected):
            raise ValueError(
                f"Schema invalido em {path}: colunas esperadas ausentes ({expected_number_columns})."
            )
        number_cols = [
            raw_df.columns[[c.lower() for c in raw_df.columns].index(col)]
            for col in expected
        ]

    accepted: list[tuple[tuple[int, int], list[int]]] = []
    rejected = 0
    accepted_idx: set[int] = set()

    if number_cols:
        numeric_df = raw_df[number_cols].apply(pd.to_numeric, errors="coerce")
        valid = numeric_df.notna().all(axis=1)
        if max_num is not None:
            valid &= numeric_df.ge(min_num).all(axis=1) & numeric_df.le(max_num).all(
                axis=1
            )

        valid_rows = numeric_df[valid].astype(int)
        if not valid_rows.empty:
            unique_mask = valid_rows.nunique(axis=1) == num_count
            valid_rows = valid_rows[unique_mask]
            for idx, row_nums in valid_rows.iterrows():
                sort_key = _row_sort_key(raw_df.loc[idx], list(raw_df.columns))
                accepted.append((sort_key, sorted(int(v) for v in row_nums.tolist())))
                accepted_idx.add(int(idx))

        # Fallback restrito evita capturar números de metadados fora das colunas-alvo.
        for idx in raw_df.index:
            idx_int = int(idx)
            if idx_int in accepted_idx:
                continue
            row = raw_df.loc[idx, number_cols] if number_cols else raw_df.loc[idx]
            numbers = _extract_numbers_from_row(row, num_count)
            if len(numbers) != num_count:
                rejected += 1
                continue
            if len(set(numbers)) != num_count:
                rejected += 1
                continue
            if max_num is not None and any(
                not (min_num <= n <= max_num) for n in numbers
            ):
                rejected += 1
                continue
            sort_key = _row_sort_key(row, list(raw_df.columns))
            accepted.append((sort_key, sorted(numbers)))
            accepted_idx.add(idx_int)
    else:
        for _, row in raw_df.iterrows():
            numbers = _extract_numbers_from_row(row, num_count)
            if len(numbers) != num_count:
                rejected += 1
                continue
            if len(set(numbers)) != num_count:
                rejected += 1
                continue
            if max_num is not None and any(
                not (min_num <= n <= max_num) for n in numbers
            ):
                rejected += 1
                continue

            sort_key = _row_sort_key(row, list(raw_df.columns))
            accepted.append((sort_key, sorted(numbers)))

    if not accepted:
        raise ValueError(f"Nenhuma linha valida encontrada em {path}")

    accepted.sort(key=lambda item: item[0])
    rows = [nums for _, nums in accepted]
    total_rows = len(raw_df)
    valid_rows = len(rows)
    reject_ratio = float(rejected / total_rows) if total_rows > 0 else 1.0

    if reject_ratio > MAX_REJECT_RATIO:
        raise ValueError(
            f"Qualidade de dados insuficiente em {path}: "
            f"{reject_ratio:.1%} de linhas rejeitadas."
        )
    row_counter = Counter(tuple(r) for r in rows)
    duplicate_rows = int(sum(c - 1 for c in row_counter.values() if c > 1))
    duplicate_ratio = float(duplicate_rows / valid_rows) if valid_rows > 0 else 0.0
    if duplicate_ratio > MAX_DUPLICATE_RATIO:
        raise ValueError(
            f"Qualidade de dados insuficiente em {path}: "
            f"{duplicate_ratio:.1%} de concursos duplicados."
        )

    if duplicate_rows > 0:
        deduped_rows = []
        seen = set()
        for row in rows:
            key = tuple(row)
            if key in seen:
                continue
            seen.add(key)
            deduped_rows.append(row)
        rows = deduped_rows

    logger.info(
        "load_data path=%s accepted=%s rejected=%s reject_ratio=%.3f dup_rows=%s dup_ratio=%.3f number_cols=%s",
        path,
        len(rows),
        rejected,
        reject_ratio,
        duplicate_rows,
        duplicate_ratio,
        bool(number_cols),
    )

    columns = [f"n{i+1}" for i in range(num_count)]
    return pd.DataFrame(rows, columns=columns)


def load_lottery_data(
    data_files: list[str],
    num_count: int,
    min_num: int = 1,
    max_num: int | None = None,
    expected_number_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Consolida múltiplos arquivos da modalidade em um único dataset validado.

    Contrato: falha quando o volume consolidado fica abaixo do mínimo operacional.
    """
    if not data_files:
        raise ValueError("Nenhum arquivo de dados fornecido")

    dfs = [
        load_data(
            path,
            num_count=num_count,
            min_num=min_num,
            max_num=max_num,
            expected_number_columns=expected_number_columns,
        )
        for path in data_files
    ]
    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) < MIN_VALID_ROWS:
        raise ValueError(
            f"Dados insuficientes no conjunto consolidado: {len(merged)} linhas validas "
            f"(minimo {MIN_VALID_ROWS})."
        )
    return merged

import argparse
import logging
import sys
from typing import Any

import uvicorn

from core.lottery import resolve_lottery_key, supported_lotteries
from core.service import generate_ranked_games, get_lottery_config
from ml.train import ModelArtifactError, train_lottery_model

logger = logging.getLogger("mega_ia.main")


def run_generation(
    n_games: int,
    modality: str = "mega",
    candidates: int = 5000,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """Executa pipeline de geração para CLI e devolve jogos ordenados com métricas."""
    lottery_key, config = get_lottery_config(modality)
    ranked_df, model_metrics = generate_ranked_games(
        lottery_key=lottery_key,
        config=config,
        n_games=min(n_games, 100),
        candidates=candidates,
    )
    return [tuple(game) for game in ranked_df["game"].tolist()], model_metrics


def parse_args():
    """Define contrato de argumentos do CLI e fallback para modo servidor."""
    parser = argparse.ArgumentParser(
        description="Sistema de geracao de apostas para loterias"
    )
    parser.add_argument("--games", type=int)
    parser.add_argument(
        "--modality",
        type=str,
        default="mega",
        choices=supported_lotteries(),
    )
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument(
        "--train", action="store_true", help="Treina modelo da modalidade"
    )
    parser.add_argument("--serve", action="store_true", help="Executa API FastAPI")
    if len(sys.argv) == 1:
        return parser.parse_args(["--serve"])
    return parser.parse_args()


def run_cli(args):
    """Orquestra fluxo de treino/geração no modo terminal conforme argumentos."""
    key = resolve_lottery_key(args.modality)
    _, config = get_lottery_config(args.modality)

    if args.train:
        artifact = train_lottery_model(key, config)
        print(f"Modelo treinado: {key}")
        print(f"Metrics: {artifact['metrics']}")
        return

    if args.games is None:
        print("Use --games para gerar jogos ou --serve para subir a API.")
        return

    games, metrics = run_generation(args.games, args.modality, args.candidates)
    print(f"\n{len(games)} jogos gerados ({key})")
    print(f"Model metrics: {metrics}")
    for idx, game in enumerate(games, start=1):
        print(f"Jogo {idx:03d}: {' - '.join(f'{n:02d}' for n in sorted(game))}")


if __name__ == "__main__":
    args = parse_args()
    if args.serve:
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        try:
            run_cli(args)
        except ModelArtifactError as exc:
            print(str(exc))
            print("Use --train para treinar o modelo antes da geracao.")
        except Exception:
            logger.exception("Falha na execucao do CLI")
            raise

import unittest

import pandas as pd

from core.generator import generate_games_random
from core.lottery import resolve_lottery_key
from core.service import generate_ranked_games
from ml.train import ModelArtifactError, load_model_artifact


class CoreContractsTest(unittest.TestCase):
    """Garante contratos basicos de dominio e integracao do service core."""

    def test_resolve_lottery_alias(self):
        self.assertEqual(resolve_lottery_key("mega-sena"), "mega")
        self.assertEqual(resolve_lottery_key("lotof\u00e1cil"), "lotofacil")

    def test_random_generation_count(self):
        games = generate_games_random(n_games=5, min_num=1, max_num=25, num_count=5)
        self.assertEqual(len(games), 5)
        self.assertTrue(all(len(game) == 5 for game in games))

    def test_missing_model_raises(self):
        with self.assertRaises(ModelArtifactError):
            load_model_artifact("nao_existe")

    def test_generate_ranked_games_accepts_ranker_gateway_contract(self):
        class FakeGateway:
            def load_artifact(self, lottery_key: str):
                return {"metrics": {"mae": 0.1}}

            def rank_candidates(self, candidate_games, artifact, top_n: int):
                selected = list(candidate_games)[:top_n]
                return pd.DataFrame(
                    [{"game": game, "predicted_score": 0.5} for game in selected]
                )

        config = {"num_range": (1, 60), "num_count": 6}
        ranked, metrics = generate_ranked_games(
            lottery_key="mega",
            config=config,
            n_games=3,
            candidates=20,
            ranker_gateway=FakeGateway(),
        )
        self.assertEqual(len(ranked), 3)
        self.assertEqual(metrics.get("mae"), 0.1)


if __name__ == "__main__":
    unittest.main()

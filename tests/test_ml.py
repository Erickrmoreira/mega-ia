import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.loader import load_data
from ml.train import TrainConfig, _temporal_backtest


class MlQualityTest(unittest.TestCase):
    """Verifica invariantes de qualidade para backtest temporal e loader."""

    def test_temporal_backtest_generates_windows(self):
        rows = []
        for t in range(5, 35):
            for i in range(4):
                rows.append(
                    {
                        "time_idx": t,
                        "target_overlap": float((t + i) % 4),
                        "hot_count": float((i + 1) % 3),
                        "warm_count": float((i + 2) % 3),
                        "cold_count": float(i % 3),
                        "num_sequences": float(i % 2),
                        "max_sequence_len": float(2 + (i % 2)),
                        "pair_score": float(10 + t + i),
                        "repeated_score": float(5 + (t % 3)),
                    }
                )
        dataset = pd.DataFrame(rows)
        feature_columns = [
            "hot_count",
            "warm_count",
            "cold_count",
            "num_sequences",
            "max_sequence_len",
            "pair_score",
            "repeated_score",
        ]
        result = _temporal_backtest(
            dataset, feature_columns, TrainConfig(backtest_windows=3)
        )
        self.assertTrue(result["enabled"])
        self.assertGreaterEqual(result["window_count"], 1)
        self.assertIn("mean_mae", result)
        self.assertIn("mean_top_decile_lift", result)
        self.assertIn("top_decile_lift_ci_low", result)
        self.assertIn("top_decile_lift_ci_high", result)
        self.assertIn("top_decile_lift_cv", result)

    def test_loader_rejects_excessive_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.csv"
            lines = ["n1,n2,n3,n4,n5,n6"]
            for _ in range(30):
                lines.append("1,2,3,4,5,6")
            path.write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_data(str(path), num_count=6, min_num=1, max_num=60)

    def test_loader_fallback_does_not_use_date_column_when_number_cols_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contaminated.csv"
            lines = [
                "data,n1,n2,n3,n4,n5,n6",
                "20/12/2025,1,2,3,4,5,x",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_data(str(path), num_count=6, min_num=1, max_num=60)


if __name__ == "__main__":
    unittest.main()

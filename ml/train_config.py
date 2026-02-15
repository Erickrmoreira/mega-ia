import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MODEL_DIR = Path("ml/artifacts")
APP_ENV = os.environ.get("MEGA_IA_ENV", "dev").strip().lower()

MAX_MAE_REGRESSION = float(os.environ.get("MEGA_IA_MAX_MAE_REGRESSION", "0.03"))
MIN_TOP_DECILE_LIFT = float(os.environ.get("MEGA_IA_MIN_TOP_DECILE_LIFT", "1.0"))
MIN_MEAN_TOP_DECILE_LIFT = float(
    os.environ.get("MEGA_IA_MIN_MEAN_TOP_DECILE_LIFT", "1.02")
)
_min_windows_default = "4" if APP_ENV == "prod" else "2"
MIN_BACKTEST_WINDOWS = int(
    os.environ.get("MEGA_IA_MIN_BACKTEST_WINDOWS", _min_windows_default)
)
_min_rank_corr_default = "0.03" if APP_ENV == "prod" else "0.00"
MIN_MEAN_RANK_CORRELATION = float(
    os.environ.get("MEGA_IA_MIN_MEAN_RANK_CORRELATION", _min_rank_corr_default)
)
_min_lift_vs_random_default = "1.01" if APP_ENV == "prod" else "1.00"
MIN_MEAN_LIFT_VS_RANDOM = float(
    os.environ.get("MEGA_IA_MIN_MEAN_LIFT_VS_RANDOM", _min_lift_vs_random_default)
)
MIN_LIFT_CI_LOW = float(os.environ.get("MEGA_IA_MIN_LIFT_CI_LOW", "1.0"))
MAX_LIFT_CV = float(os.environ.get("MEGA_IA_MAX_LIFT_CV", "0.35"))
MIN_LIFT_MARGIN = float(os.environ.get("MEGA_IA_MIN_LIFT_MARGIN", "0.02"))
FIRST_MODEL_MAX_MAE_FACTOR = float(
    os.environ.get("MEGA_IA_FIRST_MODEL_MAX_MAE_FACTOR", "1.20")
)
FIRST_MODEL_MIN_TOP_DECILE_LIFT = float(
    os.environ.get("MEGA_IA_FIRST_MODEL_MIN_TOP_DECILE_LIFT", "0.50")
)
FIRST_MODEL_MIN_MEAN_TOP_DECILE_LIFT = float(
    os.environ.get("MEGA_IA_FIRST_MODEL_MIN_MEAN_TOP_DECILE_LIFT", "1.0")
)
_first_min_windows_default = "3" if APP_ENV == "prod" else "2"
FIRST_MODEL_MIN_BACKTEST_WINDOWS = int(
    os.environ.get(
        "MEGA_IA_FIRST_MODEL_MIN_BACKTEST_WINDOWS", _first_min_windows_default
    )
)
_first_min_rank_corr_default = "0.00" if APP_ENV == "prod" else "-0.05"
FIRST_MODEL_MIN_MEAN_RANK_CORRELATION = float(
    os.environ.get(
        "MEGA_IA_FIRST_MODEL_MIN_MEAN_RANK_CORRELATION", _first_min_rank_corr_default
    )
)
_first_min_lift_vs_random_default = "1.00" if APP_ENV == "prod" else "0.98"
FIRST_MODEL_MIN_MEAN_LIFT_VS_RANDOM = float(
    os.environ.get(
        "MEGA_IA_FIRST_MODEL_MIN_MEAN_LIFT_VS_RANDOM",
        _first_min_lift_vs_random_default,
    )
)
_first_ci_low_default = "0.95" if APP_ENV == "prod" else "0.80"
FIRST_MODEL_MIN_LIFT_CI_LOW = float(
    os.environ.get("MEGA_IA_FIRST_MODEL_MIN_LIFT_CI_LOW", _first_ci_low_default)
)
_first_cv_default = "0.50" if APP_ENV == "prod" else "0.60"
FIRST_MODEL_MAX_LIFT_CV = float(
    os.environ.get("MEGA_IA_FIRST_MODEL_MAX_LIFT_CV", _first_cv_default)
)

ARTIFACT_HMAC_KEY = os.environ.get("MEGA_IA_ARTIFACT_HMAC_KEY", "")
REQUIRE_ARTIFACT_HMAC = _as_bool(
    os.environ.get("MEGA_IA_REQUIRE_ARTIFACT_HMAC"),
    default=(APP_ENV == "prod"),
)
ALLOW_PICKLE_FALLBACK = _as_bool(
    os.environ.get("MEGA_IA_ALLOW_PICKLE_FALLBACK"),
    default=False,
)

STRICT_ARTIFACT_TYPES = _as_bool(
    os.environ.get("MEGA_IA_STRICT_ARTIFACT_TYPES"),
    default=(APP_ENV == "prod"),
)
REQUIRE_BACKTEST_GATE = _as_bool(
    os.environ.get("MEGA_IA_REQUIRE_BACKTEST_GATE"),
    default=(APP_ENV == "prod"),
)

_DEFAULT_ALLOWED_SKOPS_TYPES = {
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.tree._classes.DecisionTreeRegressor",
    "sklearn.tree._tree.Tree",
    "sklearn.isotonic.IsotonicRegression",
    "numpy.dtype",
    "numpy.ndarray",
    "numpy.random._pickle.__randomstate_ctor",
    "numpy.random._pickle.__bit_generator_ctor",
    "numpy.random._mt19937.MT19937",
    "numpy.random.mtrand.RandomState",
    "numpy.random.bit_generator.SeedSequence",
}
_DEFAULT_ALLOWED_SKOPS_PREFIXES = (
    "builtins.",
    "collections.",
    "datetime.",
    "numpy.",
    "scipy.sparse.",
)

_MODEL_LOCKS: dict[str, threading.Lock] = {}
_MODEL_LOCKS_GUARD = threading.Lock()

logger = logging.getLogger("mega_ia.model_artifact")


class ModelArtifactError(Exception):
    """Erro de domínio para violações de contrato no ciclo de artefatos de modelo."""


@dataclass
class TrainConfig:
    """Configuração operacional de treino e avaliação temporal por modalidade."""

    min_history: int = 20
    negative_samples: int = 4
    random_state: int = 42
    n_estimators: int = 180
    history_window: int = 180
    time_stride: int = 2
    max_time_points: int = 350
    backtest_windows: int = 5

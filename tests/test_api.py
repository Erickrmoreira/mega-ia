import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api.app import app
from api.deps import client_ip
from api.rate_limit_backend import MemoryRateLimiter
from core.training_jobs import TrainingJob


class ApiIntegrationTest(unittest.TestCase):
    """Valida contratos principais da API (health, geracao, treino e metricas)."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "ok")

    def test_generation_cost_limit(self):
        res = self.client.get(
            "/generate/mega",
            params={"n_games": 100, "candidates": 20000},
        )
        self.assertEqual(res.status_code, 422)

    @patch("api.routes.load_model_artifact")
    @patch("api.routes.get_lottery_config")
    def test_model_info(self, get_cfg, load_artifact):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        load_artifact.return_value = {
            "trained_at": "2026-01-01T00:00:00Z",
            "metrics": {"mae": 0.5},
            "feature_columns": ["hot_count"],
            "promotion": {"promoted": True},
            "score_contract": {"score_type": "relative_ranking_signal"},
        }
        res = self.client.get("/models/info/mega")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["lottery"], "mega")
        self.assertEqual(
            res.json()["score_contract"]["score_type"], "relative_ranking_signal"
        )

    @patch("api.routes.load_model_artifact")
    @patch("api.routes.get_lottery_config")
    def test_readyz_for_model(self, get_cfg, load_artifact):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        load_artifact.return_value = {"trained_at": "2026-01-01T00:00:00Z"}
        res = self.client.get("/readyz", params={"lottery": "mega"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ready")

    @patch("api.routes.generate_ranked_games")
    @patch("api.routes.get_lottery_config")
    def test_generate_success(self, get_cfg, generate_ranked):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        generate_ranked.return_value = (
            pd.DataFrame(
                [
                    {"game": (1, 2, 3, 4, 5, 6), "predicted_score": 0.8},
                    {"game": (7, 8, 9, 10, 11, 12), "predicted_score": 0.7},
                ]
            ),
            {"mae": 0.6},
        )

        res = self.client.get(
            "/generate/mega",
            params={"n_games": 2, "candidates": 500},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["lottery"], "Mega-Sena")
        self.assertEqual(body["total_games"], 2)
        self.assertEqual(len(body["games"]), 2)

    def test_legacy_generate_removed(self):
        res = self.client.post(
            "/api/generate",
            json={"games": 10, "modality": "mega", "candidates": 1000},
        )
        self.assertIn(res.status_code, {404, 405})

    def test_auth_required_when_enabled(self):
        import api.app as app_module

        original_require = app_module.settings.require_api_key
        original_key = app_module.settings.api_key
        object.__setattr__(app_module.settings, "require_api_key", True)
        object.__setattr__(app_module.settings, "api_key", "abc")
        try:
            res = self.client.get(
                "/generate/mega", params={"n_games": 2, "candidates": 500}
            )
            self.assertEqual(res.status_code, 401)
        finally:
            object.__setattr__(app_module.settings, "require_api_key", original_require)
            object.__setattr__(app_module.settings, "api_key", original_key)

    def test_metrics_endpoint_available(self):
        res = self.client.get("/metrics")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("endpoints", body)
        self.assertIn("generation", body)
        self.assertIn("training_jobs", body)
        self.assertIn("queue_depth", body["training_jobs"])

    @patch("api.routes.get_lottery_config")
    @patch("api.app.app.state.train_jobs.submit")
    def test_train_enqueues_job(self, submit_job, get_cfg):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        submit_job.return_value = TrainingJob(
            job_id="job123",
            lottery="mega",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
        )
        res = self.client.post("/models/train/mega")
        self.assertEqual(res.status_code, 202)
        body = res.json()
        self.assertEqual(body["job_id"], "job123")
        self.assertEqual(body["status"], "queued")

    def test_train_status_not_found(self):
        res = self.client.get("/models/train/status/unknown_job")
        self.assertEqual(res.status_code, 404)

    @patch("api.routes.generate_ranked_games")
    @patch("api.routes.get_lottery_config")
    def test_metrics_generation_counter_increments(self, get_cfg, generate_ranked):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        generate_ranked.return_value = (
            pd.DataFrame([{"game": (1, 2, 3, 4, 5, 6), "predicted_score": 0.8}]),
            {"mae": 0.6},
        )
        self.client.get("/generate/mega", params={"n_games": 1, "candidates": 500})
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertGreaterEqual(metrics.json()["generation"]["requests"], 1)

    @patch("api.app.app.state.train_jobs.submit")
    @patch("api.routes.generate_ranked_games")
    @patch("api.routes.get_lottery_config")
    def test_rate_limit_scopes_generate_and_train_are_isolated(
        self, get_cfg, generate_ranked, submit_job
    ):
        import api.app as app_module

        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        generate_ranked.return_value = (
            pd.DataFrame([{"game": (1, 2, 3, 4, 5, 6), "predicted_score": 0.8}]),
            {"mae": 0.6},
        )
        submit_job.return_value = TrainingJob(
            job_id="job-limit-scope",
            lottery="mega",
            status="queued",
            created_at="2026-01-01T00:00:00+00:00",
        )

        old_gen_max = app_module.settings.rate_limit_max_requests
        old_gen_window = app_module.settings.rate_limit_window_seconds
        old_train_max = app_module.settings.train_rate_limit_max_requests
        old_train_window = app_module.settings.train_rate_limit_window_seconds
        limiter = app_module.rate_limiter
        try:
            object.__setattr__(app_module.settings, "rate_limit_max_requests", 1)
            object.__setattr__(app_module.settings, "rate_limit_window_seconds", 60)
            object.__setattr__(app_module.settings, "train_rate_limit_max_requests", 1)
            object.__setattr__(
                app_module.settings, "train_rate_limit_window_seconds", 60
            )
            if hasattr(limiter, "_buckets"):
                limiter._buckets.clear()
            if hasattr(limiter, "_last_seen"):
                limiter._last_seen.clear()

            gen_res = self.client.get(
                "/generate/mega", params={"n_games": 1, "candidates": 500}
            )
            self.assertEqual(gen_res.status_code, 200)

            train_res = self.client.post("/models/train/mega")
            self.assertEqual(train_res.status_code, 202)
        finally:
            object.__setattr__(
                app_module.settings, "rate_limit_max_requests", old_gen_max
            )
            object.__setattr__(
                app_module.settings, "rate_limit_window_seconds", old_gen_window
            )
            object.__setattr__(
                app_module.settings, "train_rate_limit_max_requests", old_train_max
            )
            object.__setattr__(
                app_module.settings, "train_rate_limit_window_seconds", old_train_window
            )
            if hasattr(limiter, "_buckets"):
                limiter._buckets.clear()
            if hasattr(limiter, "_last_seen"):
                limiter._last_seen.clear()


class SecurityHelpersTest(unittest.TestCase):
    """Cobre utilitarios de seguranca (IP de cliente e limitadores)."""

    def test_client_ip_ignores_untrusted_proxy_header(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "203.0.113.7"},
            client=SimpleNamespace(host="198.51.100.10"),
            app=SimpleNamespace(
                state=SimpleNamespace(
                    settings=SimpleNamespace(
                        trust_proxy_headers=True, trusted_proxy_ips=["10.0.0.1"]
                    )
                )
            ),
        )
        self.assertEqual(client_ip(request), "198.51.100.10")

    def test_client_ip_accepts_trusted_proxy_header(self):
        request = SimpleNamespace(
            headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"},
            client=SimpleNamespace(host="10.0.0.1"),
            app=SimpleNamespace(
                state=SimpleNamespace(
                    settings=SimpleNamespace(
                        trust_proxy_headers=True, trusted_proxy_ips=["10.0.0.1"]
                    )
                )
            ),
        )
        self.assertEqual(client_ip(request), "203.0.113.7")

    @patch.dict(
        "os.environ",
        {"MEGA_IA_RATE_LIMIT_MAX_BUCKETS": "3"},
        clear=False,
    )
    def test_memory_rate_limiter_evicts_oldest_keys(self):
        limiter = MemoryRateLimiter()
        for key in ("k1", "k2", "k3", "k4"):
            limiter.allow(key=key, window_seconds=60, max_requests=10)
        self.assertLessEqual(len(limiter._buckets), 3)

    @patch("api.app.app.state.train_jobs.submit")
    @patch("api.routes.get_lottery_config")
    def test_train_queue_full_returns_429(self, get_cfg, submit_job):
        get_cfg.return_value = ("mega", {"name": "Mega-Sena"})
        submit_job.side_effect = RuntimeError("Fila de treino lotada (max=10).")
        client = TestClient(app)
        res = client.post("/models/train/mega")
        self.assertEqual(res.status_code, 429)


if __name__ == "__main__":
    unittest.main()

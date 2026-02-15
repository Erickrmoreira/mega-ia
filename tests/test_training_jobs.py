import time
import unittest
from datetime import datetime, timedelta, timezone

from core.training_jobs import TrainingJob, TrainingJobManager


class TrainingJobsTest(unittest.TestCase):
    """Valida retries, concorrencia e limpeza do gerenciador de jobs de treino."""

    def _wait_for_job(
        self, manager: TrainingJobManager, job_id: str, timeout_s: float = 5.0
    ):
        start = time.time()
        while time.time() - start < timeout_s:
            job = manager.get(job_id)
            if job and job.status in {"succeeded", "failed"}:
                return job
            time.sleep(0.05)
        self.fail(f"Job {job_id} nao finalizou no tempo esperado.")

    def test_retry_succeeds_on_second_attempt(self):
        manager = TrainingJobManager(
            max_workers=1, retry_count=1, retry_backoff_seconds=0
        )
        state = {"calls": 0}

        def flaky_job():
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("falha transitória")
            return {"ok": True}

        job = manager.submit("mega", flaky_job)
        done = self._wait_for_job(manager, job.job_id)
        self.assertEqual(done.status, "succeeded")
        self.assertEqual(done.attempts, 2)
        self.assertEqual(done.result, {"ok": True})

    def test_retry_fails_after_max_attempts(self):
        manager = TrainingJobManager(
            max_workers=1, retry_count=1, retry_backoff_seconds=0
        )

        def always_fail():
            raise RuntimeError("falha permanente")

        job = manager.submit("quina", always_fail)
        done = self._wait_for_job(manager, job.job_id)
        self.assertEqual(done.status, "failed")
        self.assertEqual(done.attempts, 2)
        self.assertIn("falha permanente", done.error or "")

    def test_cleanup_expired_jobs(self):
        manager = TrainingJobManager(
            max_workers=1, retry_count=0, retry_backoff_seconds=0
        )
        manager._job_ttl_seconds = 1
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        job = TrainingJob(
            job_id="old_job",
            lottery="mega",
            status="succeeded",
            created_at=old_time,
            started_at=old_time,
            finished_at=old_time,
            result={"ok": True},
        )
        manager._jobs[job.job_id] = job
        removed = manager.cleanup_expired_jobs()
        self.assertEqual(removed, 1)
        self.assertIsNone(manager.get("old_job"))

    def test_submit_same_lottery_returns_running_job(self):
        manager = TrainingJobManager(
            max_workers=1, retry_count=0, retry_backoff_seconds=0
        )

        def slow_job():
            time.sleep(0.2)
            return {"ok": True}

        job1 = manager.submit("mega", slow_job)
        job2 = manager.submit("mega", slow_job)
        self.assertEqual(job1.job_id, job2.job_id)
        done = self._wait_for_job(manager, job1.job_id)
        self.assertEqual(done.status, "succeeded")

    def test_queue_limit_rejects_when_full(self):
        manager = TrainingJobManager(
            max_workers=1,
            retry_count=0,
            retry_backoff_seconds=0,
            max_queue_size=1,
        )

        def slow_job():
            time.sleep(0.2)
            return {"ok": True}

        manager.submit("mega", slow_job)
        with self.assertRaises(RuntimeError):
            manager.submit("quina", slow_job)


if __name__ == "__main__":
    unittest.main()

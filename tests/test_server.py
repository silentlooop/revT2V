"""CPU-only, no model download: server/app.py's job queue, validation, auth
and error recovery, exercised through a FakeRegistry (no MethodBank, no
torch.cuda). The real GPU registry (RealRegistry) is intentionally not
touched here -- see server/colab_server.ipynb for that."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server.app as server_app
from server.app import create_app


class FakeRegistry:
    """A method registry with no models: `run` returns a tiny random video
    (or whatever `run_fn` returns), after an optional delay."""

    def __init__(self, entries=None, run_fn=None, delay=0.0):
        self.entries = entries or [
            {"id": "teacher", "name": "Teacher", "description": "d", "available": True, "reason": None},
            {"id": "matched_target", "name": "Matched target", "description": "d", "available": True, "reason": None},
            {"id": "conv_lora", "name": "Conv LoRA", "description": "d", "available": True, "reason": None},
            {"id": "attn_injection", "name": "Attention LoRA", "description": "d", "available": False, "reason": "no checkpoint found"},
        ]
        self.run_fn = run_fn
        self.delay = delay
        self.calls = []

    def list(self):
        return self.entries

    def run(self, method_id, prompt, seed, steps, cfg, negative_prompt, on_step):
        self.calls.append(method_id)
        if self.delay:
            time.sleep(self.delay)
        if on_step is not None:
            on_step(0, steps)
        if self.run_fn is not None:
            return self.run_fn(method_id)
        rng = np.random.RandomState(seed)
        return (rng.rand(4, 16, 16, 3) * 255).astype(np.uint8)


def wait_for_status(client: TestClient, job_id: str, statuses=("done", "error"), timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not reach {statuses} within {timeout}s")


@pytest.fixture(autouse=True)
def _results_dir(tmp_path, monkeypatch):
    """Keep generated mp4/jpg files out of the real results/serve/ dir."""
    monkeypatch.setattr(server_app, "RESULTS_DIR", tmp_path)
    yield


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests opt in to an API key explicitly; default to it being unset."""
    monkeypatch.setattr(server_app, "API_KEY", None)
    yield


def test_job_lifecycle_populates_results():
    client = TestClient(create_app(FakeRegistry()))

    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["teacher", "conv_lora"], "seed": 0})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    body = wait_for_status(client, job_id)
    assert body["status"] == "done"
    assert body["error"] is None
    methods_done = {r["method"] for r in body["results"]}
    assert methods_done == {"teacher", "conv_lora"}
    for result in body["results"]:
        assert result["video_url"].startswith(f"/files/{job_id}/")
        assert result["thumbnail_url"].startswith(f"/files/{job_id}/")
        assert result["metrics"] is None  # compare_with_target defaults to False


def test_compare_with_target_adds_metrics_but_not_extra_results():
    client = TestClient(create_app(FakeRegistry()))

    resp = client.post(
        "/generate",
        json={"prompt": "a cat", "methods": ["conv_lora"], "seed": 1, "compare_with_target": True},
    )
    job_id = resp.json()["job_id"]
    body = wait_for_status(client, job_id)

    assert body["status"] == "done"
    assert [r["method"] for r in body["results"]] == ["conv_lora"]  # teacher/matched_target stay internal
    metrics = body["results"][0]["metrics"]
    assert metrics is not None
    assert set(metrics) == {"flow_cosine_vs_target", "motion_ratio_vs_teacher", "frozen", "mean_pixel_diff_vs_target"}


def test_same_seed_runs_every_requested_method_with_identical_seed():
    seeds_seen = []
    registry = FakeRegistry()
    original_run = registry.run

    def spy_run(method_id, prompt, seed, steps, cfg, negative_prompt, on_step):
        seeds_seen.append(seed)
        return original_run(method_id, prompt, seed, steps, cfg, negative_prompt, on_step)

    registry.run = spy_run
    client = TestClient(create_app(registry))

    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["teacher", "conv_lora"], "seed": 42})
    wait_for_status(client, resp.json()["job_id"])

    assert seeds_seen == [42, 42]


def test_queue_position_reported_before_jobs_drain():
    # The worker thread may dequeue job 0 before job 1/2 are even submitted
    # (it's single-threaded but asynchronous), so only job 0's position is
    # deterministic (computed in the same critical section as its enqueue);
    # what matters is the field exists, is sane, and every job still finishes.
    client = TestClient(create_app(FakeRegistry(delay=0.2)))

    responses = [
        client.post("/generate", json={"prompt": "a cat", "methods": ["teacher"], "seed": 0}) for _ in range(3)
    ]
    positions = [r.json()["position"] for r in responses]
    job_ids = [r.json()["job_id"] for r in responses]

    assert positions[0] == 0
    assert all(isinstance(p, int) and p >= 0 for p in positions)
    assert len(set(job_ids)) == 3
    for job_id in job_ids:
        assert wait_for_status(client, job_id, timeout=30.0)["status"] == "done"


def test_unknown_method_rejected():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["not_a_method"], "seed": 0})
    assert resp.status_code == 400
    assert "not_a_method" in resp.json()["detail"]


def test_unavailable_method_rejected_with_reason():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["attn_injection"], "seed": 0})
    assert resp.status_code == 400
    assert "no checkpoint found" in resp.json()["detail"]


@pytest.mark.parametrize("steps", [5, 51])
def test_steps_out_of_range_rejected(steps):
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["teacher"], "seed": 0, "steps": steps})
    assert resp.status_code == 400
    assert "steps" in resp.json()["detail"]


def test_empty_prompt_rejected():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "", "methods": ["teacher"], "seed": 0})
    assert resp.status_code == 400


def test_unknown_job_id_is_404():
    client = TestClient(create_app(FakeRegistry()))
    assert client.get("/jobs/does_not_exist").status_code == 404


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setattr(server_app, "API_KEY", "secret123")
    client = TestClient(create_app(FakeRegistry()))

    assert client.get("/methods").status_code == 401
    assert client.get("/methods", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/methods", headers={"X-API-Key": "secret123"}).status_code == 200


def test_health_never_requires_api_key(monkeypatch):
    monkeypatch.setattr(server_app, "API_KEY", "secret123")
    client = TestClient(create_app(FakeRegistry()))
    assert client.get("/health").status_code == 200


def test_failed_job_reports_error_and_server_keeps_working():
    calls = {"n": 0}

    def flaky(method_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return (np.random.RandomState(0).rand(4, 16, 16, 3) * 255).astype(np.uint8)

    client = TestClient(create_app(FakeRegistry(run_fn=flaky)))

    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["teacher"], "seed": 0})
    body = wait_for_status(client, resp.json()["job_id"])
    assert body["status"] == "error"
    assert "simulated CUDA OOM" in body["error"]

    # the worker thread must still be alive and serving the next job
    resp2 = client.post("/generate", json={"prompt": "a dog", "methods": ["teacher"], "seed": 0})
    body2 = wait_for_status(client, resp2.json()["job_id"])
    assert body2["status"] == "done"


def test_files_are_served():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "methods": ["teacher"], "seed": 0})
    body = wait_for_status(client, resp.json()["job_id"])

    video_url = body["results"][0]["video_url"]
    assert client.get(video_url).status_code == 200

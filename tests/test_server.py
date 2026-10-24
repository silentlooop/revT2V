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
            {"id": "teacher", "label": "Teacher", "kind": "main", "description": "d", "trained": False, "available": True, "checkpoint": None, "reason": None},
            {"id": "conv_lora", "label": "Conv LoRA", "kind": "main", "description": "d", "trained": True, "available": True, "checkpoint": "repo/conv_lora", "reason": None},
            {"id": "attn_injection", "label": "Attention injection", "kind": "trained", "description": "d", "trained": True, "available": False, "checkpoint": None, "reason": "no checkpoint found"},
        ]
        self.run_fn = run_fn
        self.delay = delay
        self.calls = []

    def list(self):
        return self.entries

    def run(self, method_id, prompt, seed, num_frames, steps, cfg, negative_prompt, on_step):
        self.calls.append(method_id)
        if self.delay:
            time.sleep(self.delay)
        if on_step is not None:
            on_step(0, steps)
        if self.run_fn is not None:
            return self.run_fn(method_id)
        rng = np.random.RandomState(seed + hash(method_id) % 1000)
        return (rng.rand(num_frames, 16, 16, 3) * 255).astype(np.uint8)


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
    """Keep generated mp4/png files out of the real results/serve/ dir."""
    monkeypatch.setattr(server_app, "RESULTS_DIR", tmp_path)
    yield


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests opt in to an API key explicitly; default to it being unset."""
    monkeypatch.setattr(server_app, "API_KEY", None)
    yield


@pytest.fixture(autouse=True)
def _stub_metrics(monkeypatch):
    """compute_pair_metrics needs CLIP/RAFT (real models); stub it so the
    job-lifecycle/queue/auth tests stay CPU-only and fast. Metric *content*
    isn't under test here -- it's exercised by scripts/evaluate.py's own
    tests."""

    def fake_metrics(student, teacher, teacher_reversed, prompt, raft_model):
        return {
            "fvd": float("nan"),
            "clip": {"teacher": 30.0, "student": 29.0},
            "dynamic_degree": {"teacher": 1.0, "student": 1.0},
            "optical_flow": {"teacher_mag": 1.0, "student_mag": 1.0, "direction_cosine": 0.9, "frozen": False},
            "notes": {"fvd": "stubbed in tests", "clip": None, "dynamic_degree": None, "optical_flow": None},
        }

    monkeypatch.setattr(server_app, "compute_pair_metrics", fake_metrics)
    monkeypatch.setattr(server_app, "get_raft", lambda: None)
    yield


def test_job_lifecycle_populates_result():
    client = TestClient(create_app(FakeRegistry()))

    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = wait_for_status(client, job_id)
    assert status["status"] == "done"
    assert status["error"] is None
    assert status["id"] == job_id
    assert status["request"]["student_method"] == "conv_lora"

    result = client.get(f"/jobs/{job_id}/result").json()
    assert result["job_id"] == job_id
    for key in ("teacher", "student", "teacher_reversed"):
        video = result[key]
        assert video["video_url"].startswith(f"/files/{job_id}/")
        assert all(f.startswith(f"/files/{job_id}/") for f in video["frames"])
        assert video["flow"]["grid_w"] == server_app.FLOW_GRID
    assert result["metrics"]["clip"]["student"] == 29.0


def test_result_is_409_before_done():
    client = TestClient(create_app(FakeRegistry(delay=0.3)))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0})
    job_id = resp.json()["job_id"]
    assert client.get(f"/jobs/{job_id}/result").status_code == 409
    wait_for_status(client, job_id)
    assert client.get(f"/jobs/{job_id}/result").status_code == 200


def test_teacher_reversed_flow_is_negated_and_reversed():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0})
    job_id = resp.json()["job_id"]
    wait_for_status(client, job_id)
    result = client.get(f"/jobs/{job_id}/result").json()

    teacher_flow = result["teacher"]["flow"]["frames"]
    reversed_flow = result["teacher_reversed"]["flow"]["frames"]
    assert len(teacher_flow) == len(reversed_flow)
    first_teacher_vec = teacher_flow[-1][0]
    first_reversed_vec = reversed_flow[0][0]
    assert first_reversed_vec == [-first_teacher_vec[0], -first_teacher_vec[1]]


def test_same_seed_runs_teacher_and_student_with_identical_seed():
    seeds_seen = []
    registry = FakeRegistry()
    original_run = registry.run

    def spy_run(method_id, prompt, seed, num_frames, steps, cfg, negative_prompt, on_step):
        seeds_seen.append(seed)
        return original_run(method_id, prompt, seed, num_frames, steps, cfg, negative_prompt, on_step)

    registry.run = spy_run
    client = TestClient(create_app(registry))

    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 42})
    wait_for_status(client, resp.json()["job_id"])

    assert seeds_seen == [42, 42]


def test_queue_position_reported_before_jobs_drain():
    client = TestClient(create_app(FakeRegistry(delay=0.2)))

    responses = [
        client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0}) for _ in range(3)
    ]
    job_ids = [r.json()["job_id"] for r in responses]
    assert len(set(job_ids)) == 3

    first_status = client.get(f"/jobs/{job_ids[0]}").json()
    assert isinstance(first_status["queue_pos"], int) and first_status["queue_pos"] >= 0
    for job_id in job_ids:
        assert wait_for_status(client, job_id, timeout=30.0)["status"] == "done"


def test_unknown_method_rejected():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "not_a_method", "seed": 0})
    assert resp.status_code == 400
    assert "not_a_method" in resp.json()["detail"]


def test_unavailable_method_rejected_with_reason():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "attn_injection", "seed": 0})
    assert resp.status_code == 400
    assert "no checkpoint found" in resp.json()["detail"]


@pytest.mark.parametrize("steps", [0, 101])
def test_steps_out_of_range_rejected(steps):
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0, "steps": steps})
    assert resp.status_code == 400
    assert "steps" in resp.json()["detail"]


def test_bad_num_frames_rejected():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0, "num_frames": 17})
    assert resp.status_code == 400
    assert "num_frames" in resp.json()["detail"]


def test_empty_prompt_rejected():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "", "student_method": "conv_lora", "seed": 0})
    assert resp.status_code == 400


def test_unknown_job_id_is_404():
    client = TestClient(create_app(FakeRegistry()))
    assert client.get("/jobs/does_not_exist").status_code == 404
    assert client.get("/jobs/does_not_exist/result").status_code == 404


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setattr(server_app, "API_KEY", "secret123")
    client = TestClient(create_app(FakeRegistry()))

    assert client.get("/methods").status_code == 401
    assert client.get("/methods", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/methods", headers={"X-API-Key": "secret123"}).status_code == 200


def test_health_never_requires_api_key(monkeypatch):
    monkeypatch.setattr(server_app, "API_KEY", "secret123")
    client = TestClient(create_app(FakeRegistry()))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert set(resp.json()) == {"online", "gpu", "model_loaded", "queue_length", "version"}


def test_failed_job_reports_error_and_server_keeps_working():
    calls = {"n": 0}

    def flaky(method_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return (np.random.RandomState(0).rand(16, 16, 16, 3) * 255).astype(np.uint8)

    client = TestClient(create_app(FakeRegistry(run_fn=flaky)))

    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0})
    body = wait_for_status(client, resp.json()["job_id"])
    assert body["status"] == "error"
    assert "simulated CUDA OOM" in body["error"]

    # the worker thread must still be alive and serving the next job
    resp2 = client.post("/generate", json={"prompt": "a dog", "student_method": "conv_lora", "seed": 0})
    body2 = wait_for_status(client, resp2.json()["job_id"])
    assert body2["status"] == "done"


def test_files_are_served():
    client = TestClient(create_app(FakeRegistry()))
    resp = client.post("/generate", json={"prompt": "a cat", "student_method": "conv_lora", "seed": 0})
    job_id = resp.json()["job_id"]
    wait_for_status(client, job_id)
    result = client.get(f"/jobs/{job_id}/result").json()

    video_url = result["teacher"]["video_url"]
    file_resp = client.get(video_url)
    assert file_resp.status_code == 200

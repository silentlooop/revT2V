"""Serve generated videos through a minimal FastAPI job API."""

from __future__ import annotations

import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from revt2v import infer  # noqa: E402
from revt2v.methods import METHODS  # noqa: E402
from revt2v.utils import default_checkpoint_dir, default_results_dir  # noqa: E402

RESULTS_DIR = Path(os.environ.get("REVT2V_RESULTS_DIR", default_results_dir())) / "serve"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT = Path(os.environ.get("REVT2V_CHECKPOINT_DIR", default_checkpoint_dir()))
HF_REPO = os.environ.get("REVT2V_HF_REPO")  # e.g. "username/revt2v-checkpoints"
DEFAULT_METHOD = os.environ.get("REVT2V_DEFAULT_METHOD", "conv_lora")


class GenerateRequest(BaseModel):
    prompt: str
    method: str = DEFAULT_METHOD
    seed: Optional[int] = None


class JobStatus(BaseModel):
    status: str  # "queued" | "running" | "done" | "error"
    video_url: Optional[str] = None
    error: Optional[str] = None


app = FastAPI(title="revT2V demo server")
app.mount("/videos", StaticFiles(directory=str(RESULTS_DIR)), name="videos")

_jobs: dict[str, JobStatus] = {}
_queue: "queue.Queue[tuple[str, GenerateRequest]]" = queue.Queue()
_model_cache: dict[str, tuple] = {}  # method name -> (teacher_pipeline, student)
_cache_lock = threading.Lock()


def _get_model(method: str):
    """Load (and cache) the teacher + student for a method. Loading is
    expensive (pulls the checkpoint from the Hub if not local), so this
    only happens once per method for the life of the server process."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}', expected one of {list(METHODS)}")
    with _cache_lock:
        if method not in _model_cache:
            _model_cache[method] = infer.load_student_for_inference(
                method,
                checkpoint_dir=str(CHECKPOINT_ROOT / method),
                hf_repo_id=HF_REPO,
            )
        return _model_cache[method]


def _worker() -> None:
    while True:
        job_id, request = _queue.get()
        _jobs[job_id] = JobStatus(status="running")
        try:
            teacher_pipeline, student = _get_model(request.method)
            result = infer.generate(request.prompt, teacher_pipeline, student, seed=request.seed)
            video_path = RESULTS_DIR / f"{job_id}.mp4"
            from revt2v.utils import save_video

            save_video(result["video"], video_path)
            _jobs[job_id] = JobStatus(status="done", video_url=f"/videos/{job_id}.mp4")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            _jobs[job_id] = JobStatus(status="error", error=str(exc))
        finally:
            _queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "loaded_methods": list(_model_cache),
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    if request.method not in METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{request.method}', expected one of {list(METHODS)}")
    job_id = uuid.uuid4().hex
    _jobs[job_id] = JobStatus(status="queued")
    _queue.put((job_id, request))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _jobs[job_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

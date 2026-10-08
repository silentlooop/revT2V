"""revT2V Colab inference server: a method registry, a FastAPI job API, and a
single background worker running one job at a time on one GPU.

Every method runs through revt2v.infer.MethodBank, which already is the
shared-U-Net method switcher (LoRA adapters, conv flips, attention processor
swaps, clean state restored after each call) -- see scripts/evaluate.py for
another caller of the same class. Nothing here duplicates model code; this
file is glue plus a FastAPI app.

`create_app(registry=None)` takes an optional registry so tests can inject a
fake one and exercise the whole job/queue/validation/auth machinery on a CPU,
with no model ever loaded. The real registry (`RealRegistry`) only touches
the GPU the first time `/methods` or `/generate` is actually called.

Run directly for local testing:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
See server/colab_server.ipynb for the Colab launcher (model load + tunnel).
"""

from __future__ import annotations

import logging
import os
import queue
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import imageio.v3 as iio
import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import evaluate  # scripts/evaluate.py: optical-flow metrics, reused as-is
from revt2v.infer import MethodBank
from revt2v.utils import _to_uint8_frames, default_checkpoint_dir, default_results_dir, save_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("revt2v.server")

# --------------------------------------------------------------------------
# Config (env vars only)
# --------------------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get("REVT2V_RESULTS_DIR", default_results_dir())) / "serve"
CHECKPOINT_ROOT = str(Path(os.environ.get("REVT2V_CHECKPOINT_DIR", default_checkpoint_dir())))
HF_REPO = os.environ.get("REVT2V_HF_REPO", "silentlooop/revt2v-ckpt")
API_KEY = os.environ.get("API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 16
WIDTH = HEIGHT = 256
FPS = 8
MAX_JOBS_KEPT = 50
MAX_PROMPT_LEN = 500
MIN_STEPS, MAX_STEPS = 10, 50

_START_TIME = time.time()

# --------------------------------------------------------------------------
# MethodBank: one shared pipeline, loaded lazily (never at import time, so
# the app is importable/testable on a CPU-only machine).
# --------------------------------------------------------------------------

_bank: Optional[MethodBank] = None
_bank_lock = threading.Lock()


def get_bank() -> MethodBank:
    global _bank
    with _bank_lock:
        if _bank is None:
            logger.info("Loading teacher pipeline on %s...", DEVICE)
            _bank = MethodBank(hf_repo_id=HF_REPO, checkpoint_root=CHECKPOINT_ROOT, device=DEVICE)
            for method in ("conv_lora", "attn_injection"):
                loaded = _bank.load_adapter(method)
                logger.info("checkpoint for %s: %s", method, "found" if loaded else "not found (zero-shot)" if method == "attn_injection" else "not found")
        return _bank


def to_uint8(video: torch.Tensor) -> np.ndarray:
    return _to_uint8_frames(video)


def _build_z(bank: MethodBank, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=bank.device).manual_seed(seed)
    return torch.randn(
        (1, 4, NUM_FRAMES, HEIGHT // 8, WIDTH // 8),
        generator=generator,
        device=bank.device,
        dtype=bank.pipeline.unet.dtype,
    )


RunFn = Callable[[str, int, int, float, str, Optional[Callable[[int, int], None]]], np.ndarray]


def _run_bank_method(bank_method: str) -> RunFn:
    def run(prompt, seed, steps, cfg, negative_prompt, on_step):
        bank = get_bank()
        z = _build_z(bank, seed)
        result = bank.generate(
            prompt,
            bank_method,
            latents=z,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt=negative_prompt,
            on_step=on_step,
        )
        return to_uint8(result["video"])

    return run


def _run_matched_target(prompt, seed, steps, cfg, negative_prompt, on_step):
    bank = get_bank()
    z = _build_z(bank, seed)
    result = bank.matched_target(
        prompt,
        z,
        num_inference_steps=steps,
        guidance_scale=cfg,
        negative_prompt=negative_prompt,
        on_step=on_step,
    )
    return to_uint8(result["video"])


# id -> (display name, description, bank method to check availability
# against (None = always available), run function). "teacher"/"matched_target"
# are internal-only (used by compare_with_target's metrics, see run_job) --
# PUBLIC_METHOD_IDS below is what RealRegistry.list() (and so /methods, and
# so what a client may request) actually exposes.
_METHOD_TABLE: dict[str, dict[str, Any]] = {
    "teacher": {
        "name": "Teacher",
        "description": "Normal forward-time generation from the frozen teacher.",
        "bank_method": "teacher",
        "run": _run_bank_method("teacher"),
    },
    "matched_target": {
        "name": "Matched target",
        "description": "reverse(teacher(latents=flip(z))) — the correct reference a reverse generator should reproduce.",
        "bank_method": None,
        "run": _run_matched_target,
    },
    "attn_injection": {
        "name": None,  # dynamic -- see _attn_injection_name()
        "description": (
            "Attention injection: captures the base U-Net's own attention on the time-flipped latent "
            "(stock attention, no LoRA) and re-applies it, rotated 180°, against the real trajectory's "
            "own V/output projections — a to_v/to_out.0 LoRA trained for this mechanism is used if one "
            "exists on the Hub, otherwise this runs zero-shot."
        ),
        "bank_method": "attn_injection",
        "run": _run_bank_method("attn_injection"),
    },
    "conv_oracle": {
        "name": "Conv oracle",
        "description": "Every temporal conv kernel flipped in time — an exact, training-free mirror of the teacher.",
        "bank_method": "conv_oracle",
        "run": _run_bank_method("conv_oracle"),
    },
    "conv_lora": {
        "name": "Conv LoRA",
        "description": "LoRA on temporal convs + attention, ε-MSE + mirror loss (main method).",
        "bank_method": "conv_lora",
        "run": _run_bank_method("conv_lora"),
    },
}

PUBLIC_METHOD_IDS = ("attn_injection", "conv_oracle", "conv_lora")


def _attn_injection_name(bank: MethodBank) -> str:
    return "Attention injection (LoRA)" if "attn_injection" in bank.adapters else "Attention injection (zero-shot)"


class RealRegistry:
    """The GPU-backed registry. Loads the bank (and so the GPU) on first
    `.list()` or `.run()` call -- never at construction."""

    def list(self) -> list[dict]:
        bank = get_bank()
        entries = []
        for method_id in PUBLIC_METHOD_IDS:
            info = _METHOD_TABLE[method_id]
            available = info["bank_method"] is None or info["bank_method"] in bank.methods
            name = _attn_injection_name(bank) if method_id == "attn_injection" else info["name"]
            entries.append(
                {
                    "id": method_id,
                    "name": name,
                    "description": info["description"],
                    "available": available,
                    "reason": None if available else f"no checkpoint for '{info['bank_method']}' found at {HF_REPO}",
                }
            )
        return entries

    def run(self, method_id, prompt, seed, steps, cfg, negative_prompt, on_step) -> np.ndarray:
        return _METHOD_TABLE[method_id]["run"](prompt, seed, steps, cfg, negative_prompt, on_step)


# --------------------------------------------------------------------------
# Metrics (only when compare_with_target=True). Reuses scripts/evaluate.py's
# optical_flow()/compute_flow_metrics(); the ratio/pixel-diff are a couple of
# lines of numpy, not model logic, so they're written here directly.
# --------------------------------------------------------------------------


def compute_metrics(video: np.ndarray, teacher_video: np.ndarray, target_video: np.ndarray) -> dict:
    flow_cosine = evaluate.compute_flow_metrics(video, target_video)["flow_cosine"]
    mag_video = float(np.linalg.norm(evaluate.optical_flow(video), axis=-1).mean())
    mag_teacher = float(np.linalg.norm(evaluate.optical_flow(teacher_video), axis=-1).mean())
    ratio = mag_video / max(mag_teacher, 1e-6)
    pixel_diff = float(np.abs(video.astype(np.float32) - target_video.astype(np.float32)).mean() / 255.0)
    return {
        "flow_cosine_vs_target": round(flow_cosine, 3),
        "motion_ratio_vs_teacher": round(ratio, 3),
        "frozen": ratio < 0.3,
        "mean_pixel_diff_vs_target": round(pixel_diff, 4),
    }


# --------------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    methods: list[str]
    seed: Optional[int] = None
    steps: int = 25
    cfg: float = 9.0
    negative_prompt: str = "watermark, text"
    compare_with_target: bool = False


def validate_request(req: GenerateRequest, entries: dict[str, dict]) -> None:
    if not req.prompt or len(req.prompt) > MAX_PROMPT_LEN:
        raise HTTPException(400, detail=f"prompt must be 1-{MAX_PROMPT_LEN} characters")
    if not (MIN_STEPS <= req.steps <= MAX_STEPS):
        raise HTTPException(400, detail=f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    if not req.methods:
        raise HTTPException(400, detail="methods must be a non-empty list")
    for method_id in req.methods:
        entry = entries.get(method_id)
        if entry is None:
            raise HTTPException(400, detail=f"unknown method '{method_id}', expected one of {sorted(entries)}")
        if not entry["available"]:
            raise HTTPException(400, detail=f"method '{method_id}' is not available: {entry['reason']}")


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(registry: Optional[Any] = None) -> FastAPI:
    reg = registry or RealRegistry()

    app = FastAPI(title="revT2V inference server")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(RESULTS_DIR)), name="files")

    jobs: dict[str, dict] = {}
    job_queue: "queue.Queue[str]" = queue.Queue()
    queue_order: list[str] = []
    jobs_lock = threading.Lock()

    def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if API_KEY and x_api_key != API_KEY:
            raise HTTPException(401, detail="missing or invalid X-API-Key")

    def cleanup_old_jobs() -> None:
        job_dirs = sorted(
            (p for p in RESULTS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in job_dirs[MAX_JOBS_KEPT:]:
            for f in stale.glob("*"):
                f.unlink(missing_ok=True)
            stale.rmdir()

    def run_job(job_id: str) -> None:
        job = jobs[job_id]
        req = job["request"]
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        def on_step_for(method_id: str):
            def on_step(i: int, total: int) -> None:
                job["progress"] = {"current_method": method_id, "step": i + 1, "total_steps": total}

            return on_step

        cache: dict[str, np.ndarray] = {}

        def get_video(method_id: str) -> np.ndarray:
            if method_id not in cache:
                cache[method_id] = reg.run(
                    method_id,
                    req["prompt"],
                    req["seed"],
                    req["steps"],
                    req["cfg"],
                    req["negative_prompt"],
                    on_step_for(method_id),
                )
            return cache[method_id]

        teacher_video = get_video("teacher") if req["compare_with_target"] else None
        target_video = get_video("matched_target") if req["compare_with_target"] else None

        for method_id in req["methods"]:
            video = get_video(method_id)
            video_path = job_dir / f"{method_id}.mp4"
            thumb_path = job_dir / f"{method_id}_thumb.jpg"
            save_video(video, video_path, fps=FPS)
            iio.imwrite(thumb_path, video[0])

            metrics = None
            if req["compare_with_target"] and method_id != "matched_target":
                metrics = compute_metrics(video, teacher_video, target_video)

            job["results"].append(
                {
                    "method": method_id,
                    "video_url": f"/files/{job_id}/{video_path.name}",
                    "thumbnail_url": f"/files/{job_id}/{thumb_path.name}",
                    "seconds": round(time.time() - job["created_time"], 1),
                    "metrics": metrics,
                }
            )

        job["status"] = "done"

    def worker() -> None:
        while True:
            job_id = job_queue.get()
            with jobs_lock:
                if job_id in queue_order:
                    queue_order.remove(job_id)
                jobs[job_id]["status"] = "running"
            try:
                run_job(job_id)
                cleanup_old_jobs()
            except Exception as exc:  # noqa: BLE001 - surface any failure, keep serving
                logger.exception("job %s failed", job_id)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)
            finally:
                job_queue.task_done()

    threading.Thread(target=worker, daemon=True).start()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "uptime": round(time.time() - _START_TIME, 1),
            "queue_length": job_queue.qsize(),
        }

    @app.get("/methods", dependencies=[Depends(require_api_key)])
    def methods() -> list:
        return reg.list()

    @app.post("/generate", status_code=202, dependencies=[Depends(require_api_key)])
    def generate(req: GenerateRequest) -> dict:
        entries = {e["id"]: e for e in reg.list()}
        validate_request(req, entries)

        seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        with jobs_lock:
            jobs[job_id] = {
                "status": "queued",
                "progress": {"current_method": None, "step": 0, "total_steps": req.steps},
                "results": [],
                "error": None,
                "created_time": time.time(),
                "request": {
                    "prompt": req.prompt,
                    "methods": req.methods,
                    "seed": seed,
                    "steps": req.steps,
                    "cfg": req.cfg,
                    "negative_prompt": req.negative_prompt,
                    "compare_with_target": req.compare_with_target,
                },
            }
            queue_order.append(job_id)
            position = queue_order.index(job_id)
        job_queue.put(job_id)
        return {"job_id": job_id, "position": position}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"unknown job_id '{job_id}'")
        return {
            "status": job["status"],
            "progress": job["progress"],
            "results": job["results"],
            "error": job["error"],
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""revT2V Colab inference server: a method registry, a FastAPI job API, and a
single background worker running one job at a time on one GPU.

Implements `client/src/api/CONTRACT.md` exactly -- that file (and
`src/api/types.ts`) is the source of truth for every request/response shape
here. One job = one student method, run from the same seed/initial noise as
the teacher; `teacher_reversed` is just the teacher's own frames in reverse
order (not a second generation pass), per the contract.

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
from datetime import datetime, timezone
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

import evaluate  # scripts/evaluate.py: FVD/CLIP/flow metrics, reused as-is
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
VERSION = "0.1.0"

WIDTH = HEIGHT = 256
FPS = 8
ALLOWED_NUM_FRAMES = (8, 16, 24)
FLOW_GRID = 8  # VideoOut.flow is an 8x8 grid, averaged from the dense flow field
FRAME_SAMPLES = 4  # VideoOut.frames: 4 evenly spaced frames, incl. first and last
MAX_JOBS_KEPT = 50
MAX_PROMPT_LEN = 500
MIN_STEPS, MAX_STEPS = 1, 100  # matches the frontend's advanced-panel step input range
DYNAMIC_DEGREE_FLOW_THRESHOLD = 1.0  # px/frame; used only when VBench isn't installed

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
            loaded = _bank.load_attn_rotation_adapter()
            logger.info("checkpoint for attn_rotation: %s", "found" if loaded else "not found")
        return _bank


# --------------------------------------------------------------------------
# CLIP / RAFT: loaded lazily, once, shared across jobs (same pattern as the
# bank above). Optional in the sense that a failure degrades gracefully
# (RAFT -> Farneback already happens inside evaluate.load_raft itself).
# --------------------------------------------------------------------------

_clip: Optional[tuple] = None
_clip_lock = threading.Lock()


def get_clip() -> tuple:
    global _clip
    with _clip_lock:
        if _clip is None:
            logger.info("Loading CLIP for the clip score metric...")
            _clip = evaluate.load_clip(DEVICE)
        return _clip


_raft: Any = "_unset"
_raft_lock = threading.Lock()


def get_raft() -> Optional[torch.nn.Module]:
    global _raft
    with _raft_lock:
        if _raft == "_unset":
            _raft = evaluate.load_raft(DEVICE)
        return _raft


def to_uint8(video: torch.Tensor) -> np.ndarray:
    return _to_uint8_frames(video)


def _build_z(bank: MethodBank, seed: int, num_frames: int) -> torch.Tensor:
    generator = torch.Generator(device=bank.device).manual_seed(seed)
    return torch.randn(
        (1, 4, num_frames, HEIGHT // 8, WIDTH // 8),
        generator=generator,
        device=bank.device,
        dtype=bank.pipeline.unet.dtype,
    )


RunFn = Callable[[str, int, int, int, float, str, Optional[Callable[[int, int], None]]], np.ndarray]


def _run_bank_method(bank_method: str) -> RunFn:
    def run(prompt, seed, num_frames, steps, cfg, negative_prompt, on_step):
        bank = get_bank()
        z = _build_z(bank, seed, num_frames)
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


# id -> (label, description, kind, trained, bank method to check availability
# against (None = always available), run function). "teacher" is
# internal-only (every job runs it, but it's never a selectable
# student_method) -- PUBLIC_METHOD_IDS is what GET /methods (and so what a
# client may send as student_method) actually exposes.
_METHOD_TABLE: dict[str, dict[str, Any]] = {
    "teacher": {
        "label": "Teacher",
        "description": "Normal forward-time generation from the frozen teacher.",
        "kind": "main",
        "trained": False,
        "bank_method": "teacher",
        "run": _run_bank_method("teacher"),
    },
    "attn_injection": {
        "label": None,  # dynamic -- see _attn_injection_label()
        "description": (
            "Attention injection: captures the base U-Net's own attention on the time-flipped latent "
            "(stock attention, no LoRA) and re-applies it, rotated 180°, against the real trajectory's "
            "own V/output projections — a to_v/to_out.0 LoRA trained for this mechanism is used if one "
            "exists on the Hub, otherwise this runs zero-shot."
        ),
        "kind": "trained",
        "trained": True,
        "bank_method": "attn_injection",
        "run": _run_bank_method("attn_injection"),
    },
    "conv_mirror": {
        "label": "Conv mirror",
        "description": "Every temporal conv kernel flipped in time — an exact, training-free mirror of the teacher.",
        "kind": "training_free",
        "trained": False,
        "bank_method": "conv_mirror",
        "run": _run_bank_method("conv_mirror"),
    },
    "conv_lora": {
        "label": "Conv LoRA",
        "description": "LoRA on temporal convs + attention, ε-MSE + mirror loss (main method).",
        "kind": "main",
        "trained": True,
        "bank_method": "conv_lora",
        "run": _run_bank_method("conv_lora"),
    },
    "attn_rotation": {
        "label": "Attention rotation",
        "description": "Self-rotation (Q/K flipped on the frame axis) + LoRA on the 9 up-block temporal attn1 layers' to_v/to_out.0.",
        "kind": "trained",
        "trained": True,
        "bank_method": "attn_rotation",
        "run": _run_bank_method("attn_rotation"),
    },
}

# The project's four methods -- every id GET /methods exposes and every id
# POST /generate's student_method accepts.
PUBLIC_METHOD_IDS = ("attn_injection", "conv_mirror", "conv_lora", "attn_rotation")


def _attn_injection_label(bank: MethodBank) -> str:
    return "Attention injection (LoRA)" if "attn_injection" in bank.adapters else "Attention injection (zero-shot)"


class RealRegistry:
    """The GPU-backed registry. Loads the bank (and so the GPU) on first
    `.list()` or `.run()` call -- never at construction."""

    def list(self) -> list[dict]:
        bank = get_bank()
        entries = []
        for method_id in PUBLIC_METHOD_IDS:
            info = _METHOD_TABLE[method_id]
            available = info["bank_method"] in bank.methods
            label = _attn_injection_label(bank) if method_id == "attn_injection" else info["label"]
            entries.append(
                {
                    "id": method_id,
                    "label": label,
                    "kind": info["kind"],
                    "description": info["description"],
                    "trained": info["trained"],
                    "available": available,
                    "checkpoint": f"{HF_REPO}/{info['bank_method']}" if available else None,
                    "reason": None if available else f"no checkpoint for '{info['bank_method']}' found at {HF_REPO}",
                }
            )
        return entries

    def run(self, method_id, prompt, seed, num_frames, steps, cfg, negative_prompt, on_step) -> np.ndarray:
        return _METHOD_TABLE[method_id]["run"](prompt, seed, num_frames, steps, cfg, negative_prompt, on_step)


# --------------------------------------------------------------------------
# VideoOut construction: mp4 + 4 sampled frame PNGs + a sparse flow grid.
# --------------------------------------------------------------------------


def _frame_sample_indices(num_frames: int, count: int = FRAME_SAMPLES) -> list[int]:
    """`count` evenly spaced indices including the first and last frame,
    e.g. num_frames=16 -> [0, 5, 10, 15] (the contract's own example)."""
    if num_frames <= count:
        return list(range(num_frames))
    return sorted({round(i * (num_frames - 1) / (count - 1)) for i in range(count)})


def _sparse_flow_grid(dense_flow: np.ndarray, grid: int = FLOW_GRID) -> dict:
    """``(F-1, S, S, 2)`` dense flow -> the contract's sparse FlowField:
    average-pool each frame transition into a `grid x grid` grid, flattened
    row-major (`frames[t][j*grid+i] = [dx, dy]`)."""
    num_transitions, size, _, _ = dense_flow.shape
    block = size // grid
    pooled = dense_flow[:, : block * grid, : block * grid, :].reshape(num_transitions, grid, block, grid, block, 2).mean(axis=(2, 4))
    frames = [pooled[t].reshape(grid * grid, 2).tolist() for t in range(num_transitions)]
    return {"grid_w": grid, "grid_h": grid, "frames": frames}


def _video_out(video: np.ndarray, job_dir: Path, name: str, flow: Optional[dict] = None, raft_model: Any = "_unset") -> dict:
    """Save `video` (F, H, W, C) uint8 as `<name>.mp4` + `<name>_<idx>.png`
    frame samples under `job_dir`, and return a VideoOut dict. Pass `flow`
    to reuse an already-computed FlowField (e.g. teacher_reversed reusing
    the teacher's own flow, reversed+negated) instead of recomputing it."""
    num_frames = video.shape[0]
    video_path = job_dir / f"{name}.mp4"
    save_video(video, video_path, fps=FPS)

    indices = _frame_sample_indices(num_frames)
    frame_paths = []
    for idx in indices:
        frame_path = job_dir / f"{name}_{idx:02d}.png"
        iio.imwrite(frame_path, video[idx])
        frame_paths.append(f"/files/{job_dir.name}/{frame_path.name}")

    if flow is None:
        model = get_raft() if raft_model == "_unset" else raft_model
        flow = _sparse_flow_grid(evaluate.optical_flow(video, model))

    return {
        "video_url": f"/files/{job_dir.name}/{video_path.name}",
        "frames": frame_paths,
        "frame_indices": indices,
        "flow": flow,
        "fps": FPS,
        "width": video.shape[2],
        "height": video.shape[1],
        "num_frames": num_frames,
    }


def _reversed_flow(flow: dict) -> dict:
    """Reversing a video's frame order flips its flow's transition order
    and negates every vector (per CONTRACT.md's VideoOut docs)."""
    return {
        "grid_w": flow["grid_w"],
        "grid_h": flow["grid_h"],
        "frames": [[[-dx, -dy] for dx, dy in frame] for frame in reversed(flow["frames"])],
    }


# --------------------------------------------------------------------------
# Metrics (PairMetrics). Reuses scripts/evaluate.py's CLIP/flow functions as-
# is; FVD and dynamic_degree are noted as approximate/unavailable for a
# single live pair rather than silently faked, matching evaluate.py's own
# "VBench optional" / "FVD unreliable below 50 videos" precedents.
# --------------------------------------------------------------------------


def compute_pair_metrics(student: np.ndarray, teacher: np.ndarray, teacher_reversed: np.ndarray, prompt: str, raft_model: Any) -> dict:
    model, processor = get_clip()
    clip_teacher = evaluate.compute_clip_score(teacher, prompt, model, processor)
    clip_student = evaluate.compute_clip_score(student, prompt, model, processor)

    flow_student = evaluate.optical_flow(student, raft_model)
    flow_teacher_rev = evaluate.optical_flow(teacher_reversed, raft_model)
    mag_student = float(np.linalg.norm(flow_student, axis=-1).mean())
    mag_teacher = float(np.linalg.norm(flow_teacher_rev, axis=-1).mean())
    direction_cosine = evaluate.compute_flow_metrics(student, teacher_reversed, raft_model)["flow_cosine"]
    frozen = mag_student < 0.3 * mag_teacher

    try:
        dynamic_degree = _vbench_dynamic_degree(student, teacher)
        dynamic_note = None
    except Exception as exc:  # noqa: BLE001 - VBench is optional, fall back
        logger.warning("VBench dynamic_degree unavailable (%s); approximating from flow magnitude", exc)
        threshold = DYNAMIC_DEGREE_FLOW_THRESHOLD
        dynamic_degree = {"teacher": float(mag_teacher > threshold), "student": float(mag_student > threshold)}
        dynamic_note = "VBench unavailable — approximated from optical-flow magnitude, not the real VBench dynamic_degree."

    return {
        "fvd": float("nan"),
        "clip": {"teacher": clip_teacher, "student": clip_student},
        "dynamic_degree": dynamic_degree,
        "optical_flow": {
            "teacher_mag": mag_teacher,
            "student_mag": mag_student,
            "direction_cosine": direction_cosine,
            "frozen": frozen,
        },
        "notes": {
            "fvd": "Single-pair FVD needs a batch of samples to be meaningful; see scripts/evaluate.py for the set-level number.",
            "clip": None,
            "dynamic_degree": dynamic_note,
            "optical_flow": None,
        },
    }


def _vbench_dynamic_degree(student: np.ndarray, teacher: np.ndarray) -> dict:
    """Best-effort real VBench score for this one pair; raises if VBench
    isn't installed or fails (caller falls back to the flow-based proxy)."""
    import tempfile

    from revt2v.utils import save_video as _save

    evaluate._install_decord_shim()
    import vbench  # noqa: F401 - import error is the "unavailable" signal

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        teacher_dir, student_dir = tmp_dir / "teacher", tmp_dir / "student"
        teacher_dir.mkdir()
        student_dir.mkdir()
        _save(teacher, teacher_dir / "video.mp4", fps=FPS)
        _save(student, student_dir / "video.mp4", fps=FPS)
        scores = evaluate.run_vbench_dynamic_degree({"teacher": teacher_dir, "student": student_dir}, tmp_dir, device=DEVICE)
        return {
            "teacher": next(iter(scores.get("teacher", {}).values()), 0.0),
            "student": next(iter(scores.get("student", {}).values()), 0.0),
        }


# --------------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    student_method: str
    seed: Optional[int] = None
    steps: int = 25
    cfg: float = 9.0
    num_frames: int = 16
    negative_prompt: str = "watermark, text"


def validate_request(req: GenerateRequest, entries: dict[str, dict]) -> None:
    if not req.prompt or len(req.prompt) > MAX_PROMPT_LEN:
        raise HTTPException(400, detail=f"prompt must be 1-{MAX_PROMPT_LEN} characters")
    if not (MIN_STEPS <= req.steps <= MAX_STEPS):
        raise HTTPException(400, detail=f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    if req.num_frames not in ALLOWED_NUM_FRAMES:
        raise HTTPException(400, detail=f"num_frames must be one of {ALLOWED_NUM_FRAMES}")
    entry = entries.get(req.student_method)
    if entry is None:
        raise HTTPException(400, detail=f"unknown student_method '{req.student_method}', expected one of {sorted(entries)}")
    if not entry["available"]:
        raise HTTPException(400, detail=f"student_method '{req.student_method}' is not available: {entry['reason']}")


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

        def on_step_for(model: str):
            def on_step(i: int, total: int) -> None:
                job["current_model"] = model
                job["step"] = i + 1
                job["total_steps"] = total

            return on_step

        teacher_video = reg.run(
            "teacher", req["prompt"], req["seed"], req["num_frames"], req["steps"], req["cfg"], req["negative_prompt"], on_step_for("teacher")
        )
        student_video = reg.run(
            req["student_method"], req["prompt"], req["seed"], req["num_frames"], req["steps"], req["cfg"], req["negative_prompt"], on_step_for("student")
        )

        raft_model = get_raft()
        teacher_out = _video_out(teacher_video, job_dir, "teacher", raft_model=raft_model)
        student_out = _video_out(student_video, job_dir, "student", raft_model=raft_model)
        teacher_reversed_video = teacher_video[::-1].copy()
        teacher_reversed_out = _video_out(
            teacher_reversed_video, job_dir, "teacher_reversed", flow=_reversed_flow(teacher_out["flow"])
        )

        metrics = compute_pair_metrics(student_video, teacher_video, teacher_reversed_video, req["prompt"], raft_model)

        job["result"] = {
            "job_id": job_id,
            "request": req,
            "time_taken_s": round(time.time() - job["created_time"], 1),
            "teacher": teacher_out,
            "student": student_out,
            "teacher_reversed": teacher_reversed_out,
            "metrics": metrics,
        }
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
            "online": True,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_loaded": _bank is not None,
            "queue_length": job_queue.qsize(),
            "version": VERSION,
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
                "current_model": None,
                "step": 0,
                "total_steps": req.steps,
                "result": None,
                "error": None,
                "created_time": time.time(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "request": {
                    "prompt": req.prompt,
                    "student_method": req.student_method,
                    "seed": seed,
                    "steps": req.steps,
                    "cfg": req.cfg,
                    "num_frames": req.num_frames,
                    "negative_prompt": req.negative_prompt,
                },
            }
            queue_order.append(job_id)
        job_queue.put(job_id)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"unknown job_id '{job_id}'")
        with jobs_lock:
            queue_pos = queue_order.index(job_id) if job_id in queue_order else 0
        return {
            "id": job_id,
            "status": job["status"],
            "queue_pos": queue_pos,
            "current_model": job["current_model"],
            "step": job["step"],
            "total_steps": job["total_steps"],
            "elapsed_s": round(time.time() - job["created_time"], 1),
            "created_at": job["created_at"],
            "request": job["request"],
            "error": job["error"],
        }

    @app.get("/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
    def job_result(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, detail=f"unknown job_id '{job_id}'")
        if job["status"] != "done":
            raise HTTPException(409, detail=f"job '{job_id}' is not done yet (status: {job['status']})")
        return job["result"]

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Score reverse-time videos: FVD, CLIP score, VBench dynamic degree, optical flow.

Every method is scored against a REFERENCE method (default `matched_target`,
i.e. reverse(teacher(latents=flip(z))) from the same noise z). The reference's
own FVD and flow columns are NaN (never computed against itself).

- fvd                    per method, Frechet distance between I3D (Kinetics-400)
                         features of the method's videos and the reference's,
                         over the same prompts/seeds (lower = better). Unreliable
                         in absolute terms below 50 videos: directional only.
- clip_score             per video, 100 * CLIP cosine(frame, prompt), averaged
                         over frames (higher = better).
- vbench_dynamic_degree  per video, VBench's binary "is it moving" (1/0); per
                         method, the fraction of dynamic videos.
- flow_cosine            per video, cosine between the signed (u, v) flow vectors
                         of the video and the reference at the same frame pair
                         and pixel, weighted by the reference's flow magnitude:
                         +1 same motion direction, -1 opposite. Flow is RAFT
                         (torchvision), or OpenCV Farneback if RAFT is
                         unavailable, always at 256x256.

Two ways to run it:

1. Score videos already on disk (what notebooks/compare_methods.ipynb does):
       python scripts/evaluate.py --videos-dir /content/compare_results/physics/videos
   Layout: <videos-dir>/<method>/<name>.mp4 for every method (including the
   reference), and <videos-dir>/prompts.json mapping <name> -> prompt.

2. Generate with revt2v.infer.MethodBank, then score (needs a GPU):
       python scripts/evaluate.py --prompts prompts/test.txt --seeds 0 1 \\
           --hf-repo <you>/revt2v-ckpt --output results/eval

Writes metrics.csv (one row per method x video) and summary.csv /
summary.json (one row per method: fvd, plus the per-video metrics averaged)
next to the videos directory.

VBench is optional (pip install --no-deps vbench easydict; see the
notebook's install cell). It's skipped with a warning if it isn't importable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revt2v.utils import seed_everything  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

REFERENCE = "matched_target"
ALL_METRICS = ("fvd", "clip", "vbench", "flow")
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
# I3D (Kinetics-400) TorchScript used by StyleGAN-V and most recent FVD code.
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
CACHE_DIR = Path.home() / ".cache" / "revt2v"
VBENCH_DIM = "dynamic_degree"
FLOW_SIZE = 256  # every video's flow is computed at FLOW_SIZE x FLOW_SIZE
FVD_MIN_RELIABLE = 50
PER_VIDEO_COLUMNS = ("method", "video", "prompt", "clip_score", "flow_cosine", "vbench_dynamic_degree")
SUMMARY_COLUMNS = ("method", "videos", "fvd", "clip_score", "flow_cosine", "vbench_dynamic_degree")


# --------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------


def load_video(path: str | Path) -> np.ndarray:
    """Read an mp4 into a ``(F, H, W, 3)`` uint8 array."""
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        frames = np.stack([np.asarray(frame) for frame in reader])
    finally:
        reader.close()
    return frames[..., :3].astype(np.uint8)


def save_video_set(
    videos: Dict[str, Dict[str, np.ndarray]],
    prompts: Dict[str, str],
    videos_dir: str | Path,
    fps: int = 8,
) -> Path:
    """Write ``{method: {name: (F, H, W, C) float video in [0, 1]}}`` into the
    <videos-dir>/<method>/<name>.mp4 layout plus prompts.json."""
    from diffusers.utils import export_to_video

    videos_dir = Path(videos_dir)
    for method, by_name in videos.items():
        (videos_dir / method).mkdir(parents=True, exist_ok=True)
        for name, video in by_name.items():
            # export_to_video expects float frames in [0, 1], not uint8
            export_to_video(list(np.clip(video, 0, 1)), str(videos_dir / method / f"{name}.mp4"), fps=fps)
    with open(videos_dir / "prompts.json", "w") as handle:
        json.dump(prompts, handle, indent=2)
    return videos_dir


# --------------------------------------------------------------------------
# FVD
# --------------------------------------------------------------------------


def load_i3d(device: str = "cuda") -> torch.jit.ScriptModule:
    """Download (once, to ~/.cache/revt2v) and load the I3D TorchScript."""
    path = CACHE_DIR / "i3d_torchscript.pt"
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading I3D for FVD to %s", path)
        urllib.request.urlretrieve(I3D_URL, path)
    return torch.jit.load(str(path), map_location=device).eval()


@torch.no_grad()
def i3d_features(videos: Sequence[np.ndarray], detector: Any, device: str = "cuda", batch_size: int = 8) -> np.ndarray:
    """``N x (F, H, W, C)`` uint8 videos -> ``(N, 400)`` I3D features.
    Frames are resized to 224x224 and scaled to [-1, 1]."""
    features = []
    for start in range(0, len(videos), batch_size):
        batch = torch.from_numpy(np.stack(videos[start : start + batch_size])).to(device).float() / 255.0
        n, f, h, w, c = batch.shape
        frames = batch.permute(0, 1, 4, 2, 3).reshape(n * f, c, h, w)
        frames = torch.nn.functional.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        clip = frames.reshape(n, f, c, 224, 224).permute(0, 2, 1, 3, 4)  # (N, C, F, 224, 224)
        clip = clip * 2 - 1
        features.append(detector(clip, rescale=False, resize=False, return_features=True).cpu().numpy())
    return np.concatenate(features).astype(np.float64)


def _sqrtm_is_usable(covmean: np.ndarray) -> bool:
    if not np.isfinite(covmean).all():
        return False
    return not np.iscomplexobj(covmean) or np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between Gaussians fit to two ``(N, D)`` feature sets.

    With few samples the covariances are singular; if sqrtm(cov_a @ cov_b)
    fails or comes back non-finite/meaningfully complex, retry with eps*I
    added to both covariances. The real part is used either way.
    """
    from scipy import linalg

    if len(feats_a) < 2 or len(feats_b) < 2:
        return float("nan")
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    cov_a, cov_b = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    try:
        covmean = linalg.sqrtm(cov_a @ cov_b)
    except (linalg.LinAlgError, ValueError):
        covmean = None
    if covmean is None or not _sqrtm_is_usable(covmean):
        logger.warning("FVD: sqrtm unstable, adding %g*I to the covariances", eps)
        offset = np.eye(cov_a.shape[0]) * eps
        covmean = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    covmean = np.real(covmean)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


def compute_fvd(real_videos: Sequence[np.ndarray], fake_videos: Sequence[np.ndarray], detector: Any, device: str = "cuda") -> float:
    """FVD between two sets of ``(F, H, W, C)`` uint8 videos (lower = closer).
    Both sets must be the same size, resolution and frame count."""
    if len(real_videos) != len(fake_videos):
        raise ValueError(f"FVD needs equal set sizes, got {len(real_videos)} vs {len(fake_videos)}")
    shapes = {v.shape for v in [*real_videos, *fake_videos]}
    if len(shapes) != 1:
        raise ValueError(f"FVD needs one resolution/frame count for every video, got {sorted(shapes)}")
    if len(real_videos) < FVD_MIN_RELIABLE:
        logger.warning(
            "FVD on %d videos per set (< %d) is unreliable in absolute terms; compare methods only",
            len(real_videos),
            FVD_MIN_RELIABLE,
        )
    return frechet_distance(i3d_features(real_videos, detector, device), i3d_features(fake_videos, detector, device))


# --------------------------------------------------------------------------
# CLIP score
# --------------------------------------------------------------------------


def load_clip(device: str = "cuda"):
    from transformers import CLIPModel, CLIPProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    logger.info("CLIP score model: %s (same model for every method)", CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=dtype).to(device).eval()
    return model, CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


def _embedding(output: Any) -> torch.Tensor:
    # transformers<5 returns the projected embedding; >=5 returns an output
    # object whose pooler_output holds it.
    return output if torch.is_tensor(output) else output.pooler_output


@torch.no_grad()
def compute_clip_score(video: np.ndarray, prompt: str, model: Any, processor: Any) -> float:
    """100 * cos(frame, prompt), averaged over frames."""
    inputs = processor(text=[prompt], images=list(video), return_tensors="pt", padding=True, truncation=True)
    image = _embedding(model.get_image_features(pixel_values=inputs["pixel_values"].to(model.device, model.dtype)))
    text = _embedding(
        model.get_text_features(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
        )
    )
    image = torch.nn.functional.normalize(image.float(), dim=-1)
    text = torch.nn.functional.normalize(text.float(), dim=-1)
    return float(100 * (image @ text.T).mean())


# --------------------------------------------------------------------------
# Optical flow
# --------------------------------------------------------------------------


def load_raft(device: str = "cuda") -> Optional[torch.nn.Module]:
    """torchvision RAFT-large, or None (-> Farneback) if it can't be loaded."""
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        logger.info("Optical flow: torchvision RAFT-large at %dx%d", FLOW_SIZE, FLOW_SIZE)
        return model
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail
        logger.warning("RAFT unavailable (%s); optical flow falls back to OpenCV Farneback", exc)
        return None


def _resize_frames(video: np.ndarray, size: int = FLOW_SIZE) -> np.ndarray:
    import cv2

    if video.shape[1:3] == (size, size):
        return video
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in video])


@torch.no_grad()
def optical_flow(video: np.ndarray, model: Optional[torch.nn.Module] = None, batch_size: int = 8) -> np.ndarray:
    """Signed flow between consecutive frames at FLOW_SIZE: ``(F-1, S, S, 2)``.
    RAFT if `model` is given, else OpenCV Farneback."""
    video = _resize_frames(video)
    if model is None:
        import cv2

        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
        return np.stack(
            [cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0) for a, b in zip(gray[:-1], gray[1:])]
        )

    device = next(model.parameters()).device
    frames = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1  # RAFT expects [-1, 1]
    flows = []
    for start in range(0, len(frames) - 1, batch_size):
        stop = min(start + batch_size, len(frames) - 1)
        flows.append(model(frames[start:stop], frames[start + 1 : stop + 1])[-1])  # last refinement, (B, 2, S, S)
    return torch.cat(flows).permute(0, 2, 3, 1).cpu().numpy()


def compute_flow_metrics(video: np.ndarray, reference: np.ndarray, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """flow_cosine: direction agreement of the signed (u, v) vectors at the
    same frame pair and pixel, weighted where the reference moves."""
    if len(video) != len(reference):
        raise ValueError(f"flow needs the same frame count, got {len(video)} vs {len(reference)}")
    flow, flow_ref = optical_flow(video, model), optical_flow(reference, model)
    mag, mag_ref = np.linalg.norm(flow, axis=-1), np.linalg.norm(flow_ref, axis=-1)
    cosine = (flow * flow_ref).sum(-1) / (mag * mag_ref + 1e-6)
    return {"flow_cosine": float((cosine * mag_ref).sum() / (mag_ref.sum() + 1e-6))}


# --------------------------------------------------------------------------
# VBench
# --------------------------------------------------------------------------


def _install_decord_shim() -> None:
    """vbench imports `decord`, which has no wheels for recent Pythons
    (e.g. Colab's 3.13). If it's missing, register a minimal imageio-backed
    stand-in covering what VBench's mp4 loader uses."""
    try:
        import decord  # noqa: F401

        return
    except ImportError:
        pass

    class _Batch:
        def __init__(self, frames: np.ndarray) -> None:
            self._frames = frames

        def asnumpy(self) -> np.ndarray:
            return self._frames

    class VideoReader:
        def __init__(self, path, width=None, height=None, num_threads=1, ctx=None):
            import imageio

            reader = imageio.get_reader(str(path))
            self._fps = float(reader.get_meta_data().get("fps", 8))
            reader.close()
            frames = load_video(path)
            if width and height:
                import cv2

                frames = np.stack([cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in frames])
            self._frames = frames

        def __len__(self) -> int:
            return len(self._frames)

        def __getitem__(self, index):
            return _Batch(self._frames[index])

        def get_batch(self, indices):
            return _Batch(self._frames[list(indices)])

        def get_avg_fps(self) -> float:
            return self._fps

    shim = types.ModuleType("decord")
    shim.VideoReader = VideoReader
    shim.cpu = lambda *args, **kwargs: None
    shim.bridge = types.SimpleNamespace(set_bridge=lambda *args, **kwargs: None)
    sys.modules["decord"] = shim
    logger.info("decord not installed; using an imageio-backed stand-in for VBench")


def run_vbench_dynamic_degree(
    method_dirs: Dict[str, Path],
    output_dir: str | Path,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Return ``{method: {video_name: 1.0 or 0.0}}`` from VBench's
    dynamic_degree (its own RAFT; thresholds scale with resolution and frame
    count, and it keeps every frame of an 8 fps video)."""
    try:
        _install_decord_shim()
        import vbench
        from vbench import VBench
    except Exception as exc:  # noqa: BLE001 - VBench is optional
        logger.warning("VBench unavailable (%s); skipping VBench metrics", exc)
        return {}

    output_dir = Path(output_dir)
    full_info = Path(vbench.__file__).parent / "VBench_full_info.json"
    bench = VBench(device, str(full_info), str(output_dir))
    scores: Dict[str, Dict[str, float]] = {}
    for method, folder in method_dirs.items():
        name = f"{method}__{VBENCH_DIM}"
        try:
            bench.evaluate(videos_path=str(folder), name=name, dimension_list=[VBENCH_DIM], mode="custom_input")
            with open(output_dir / f"{name}_eval_results.json") as handle:
                _, per_video = json.load(handle)[VBENCH_DIM]
        except Exception as exc:  # noqa: BLE001 - keep the other methods
            logger.warning("VBench %s failed for %s: %s", VBENCH_DIM, method, exc)
            continue
        for entry in per_video:
            scores.setdefault(method, {})[Path(entry["video_path"]).stem] = float(bool(entry["video_results"]))
    return scores


# --------------------------------------------------------------------------
# Scoring a videos directory
# --------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mean(values: Iterable[Any]) -> float:
    values = _finite(values)
    return float(np.mean(values)) if values else float("nan")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def score_videos_dir(
    videos_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    reference: str = REFERENCE,
    metrics: Sequence[str] = ALL_METRICS,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Score every <videos-dir>/<method>/*.mp4. Returns ``{"rows": per-video
    dicts, "summary": per-method dicts}`` and writes metrics.csv,
    summary.csv and summary.json to `output_dir` (default: videos_dir's parent)."""
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir) if output_dir else videos_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; choose from {ALL_METRICS}")

    prompts_path = videos_dir / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    method_dirs = {d.name: d for d in sorted(videos_dir.iterdir()) if d.is_dir() and any(d.glob("*.mp4"))}
    if reference not in method_dirs:
        raise FileNotFoundError(f"No reference folder {videos_dir / reference}")
    videos = {m: {p.stem: load_video(p) for p in sorted(d.glob("*.mp4"))} for m, d in method_dirs.items()}
    logger.info("Scoring %s against %r: %s", videos_dir, reference, {m: len(v) for m, v in videos.items()})

    nan = float("nan")
    rows = []
    for method, by_name in videos.items():
        for name in by_name:
            row = dict.fromkeys(PER_VIDEO_COLUMNS, nan)
            row.update(method=method, video=name, prompt=prompts.get(name, ""))
            rows.append(row)
    row_of = {(r["method"], r["video"]): r for r in rows}

    if "clip" in metrics:
        model, processor = load_clip(device)
        for (method, name), row in row_of.items():
            if row["prompt"]:
                row["clip_score"] = compute_clip_score(videos[method][name], row["prompt"], model, processor)
        del model
        torch.cuda.empty_cache()

    if "flow" in metrics:
        raft = load_raft(device)
        for (method, name), row in row_of.items():
            ref = videos[reference].get(name)
            if method != reference and ref is not None:  # reference stays NaN
                row.update(compute_flow_metrics(videos[method][name], ref, raft))
        del raft
        torch.cuda.empty_cache()

    if "vbench" in metrics:
        for method, by_name in run_vbench_dynamic_degree(method_dirs, output_dir / "vbench", device).items():
            for name, dynamic in by_name.items():
                if (method, name) in row_of:
                    row_of[(method, name)]["vbench_dynamic_degree"] = dynamic

    fvd = {}
    if "fvd" in metrics:
        detector = load_i3d(device)
        ref_names = sorted(videos[reference])
        for method in videos:
            if method == reference:
                continue  # reference stays NaN
            if sorted(videos[method]) != ref_names:
                # Names are p<prompt>_s<seed>: a mismatch means different prompts/seeds/N.
                logger.error("FVD skipped for %s: its videos don't match %s's prompts/seeds", method, reference)
                fvd[method] = nan
                continue
            fvd[method] = compute_fvd(
                [videos[reference][n] for n in ref_names], [videos[method][n] for n in ref_names], detector, device
            )
        del detector
        torch.cuda.empty_cache()

    summary = []
    for method in videos:
        method_rows = [r for r in rows if r["method"] == method]
        entry: Dict[str, Any] = {"method": method, "videos": len(method_rows), "fvd": fvd.get(method, nan)}
        for key in ("clip_score", "flow_cosine"):
            entry[key] = _mean(r[key] for r in method_rows)
        # Binary per video -> fraction of dynamic videos; never averaged with anything else.
        entry["vbench_dynamic_degree"] = _mean(r["vbench_dynamic_degree"] for r in method_rows)
        summary.append(entry)

    _write_csv(rows, output_dir / "metrics.csv")
    _write_csv(summary, output_dir / "summary.csv")
    with open(output_dir / "summary.json", "w") as handle:
        json.dump({"reference": reference, "summary": summary, "per_video": rows}, handle, indent=2)
    logger.info("Wrote %s, summary.csv, summary.json", output_dir / "metrics.csv")
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------
# Generating videos with MethodBank
# --------------------------------------------------------------------------


def generate_videos(
    bank: Any,
    prompts: Sequence[str],
    seeds: Sequence[int],
    methods: Optional[Sequence[str]] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Run every method (plus `matched_target`) from the same noise per
    (prompt, seed). Returns ``({method: {name: float video}}, {name: prompt})``
    with names ``p<prompt index>_s<seed>``, ready for `save_video_set`."""
    gen_kwargs = dict(gen_kwargs or {})
    methods = [m for m in (methods or bank.methods) if m in bank.methods]
    num_frames = gen_kwargs.get("num_frames", 16)
    height, width = gen_kwargs.get("height", 256), gen_kwargs.get("width", 256)

    videos: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in [REFERENCE, *methods]}
    names: Dict[str, str] = {}
    for p_idx, prompt in enumerate(prompts):
        for seed in seeds:
            name = f"p{p_idx}_s{seed}"
            names[name] = prompt
            generator = torch.Generator(device=bank.device).manual_seed(seed)
            z = torch.randn(
                (1, 4, num_frames, height // 8, width // 8),
                generator=generator,
                device=bank.device,
                dtype=bank.pipeline.unet.dtype,
            )
            videos[REFERENCE][name] = bank.matched_target(prompt, z, **gen_kwargs)["video"].float().cpu().numpy()
            for method in methods:
                result = bank.generate(prompt, method, latents=z, **gen_kwargs)
                videos[method][name] = result["video"].float().cpu().numpy()
            logger.info("generated %s: %s", name, prompt)
    return videos, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=Path, default=None, help="Score existing videos (skip generation)")
    parser.add_argument("--prompts", type=Path, default=None, help="Prompt file: generate with MethodBank first")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", default=None, help="Default: every method MethodBank can load")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="results/checkpoints")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--negative-prompt", type=str, default="watermark, text")
    parser.add_argument("--output", type=Path, default=Path("results/eval"))
    parser.add_argument("--reference", type=str, default=REFERENCE)
    parser.add_argument("--metrics", nargs="+", default=list(ALL_METRICS), choices=ALL_METRICS)
    args = parser.parse_args()

    if args.videos_dir is None:
        if args.prompts is None:
            parser.error("pass --videos-dir to score existing videos, or --prompts to generate first")
        from revt2v.infer import MethodBank

        seed_everything(args.seeds[0])
        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
        prompts = prompts[: args.limit] if args.limit else prompts
        bank = MethodBank(hf_repo_id=args.hf_repo, checkpoint_root=args.checkpoint_root)
        for method in ("conv_lora", "attn_lora", "attn_injection"):
            try:
                bank.load_adapter(method)
            except Exception as exc:  # noqa: BLE001 - missing/mismatched checkpoint
                logger.warning("Skipping %s: %s", method, exc)
        gen_kwargs = dict(
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
        videos, names = generate_videos(bank, prompts, args.seeds, args.methods, gen_kwargs)
        args.videos_dir = save_video_set(videos, names, args.output / "videos")
        del bank
        torch.cuda.empty_cache()

    result = score_videos_dir(
        args.videos_dir,
        output_dir=args.output if args.prompts else None,
        reference=args.reference,
        metrics=args.metrics,
    )
    for entry in result["summary"]:
        logger.info("%s", entry)


if __name__ == "__main__":
    main()

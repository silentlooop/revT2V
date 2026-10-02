"""Provide shared configuration, video, environment, and checkpoint utilities."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's, and (if installed) PyTorch's RNGs.

    Args:
        seed: Any integer. Same seed + same code path -> same random numbers.

    Torch/NumPy are optional imports here because this module is imported by
    lightweight tooling (e.g. `scripts/serve.py` health checks) that may run
    before those heavier packages are needed.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Config loading with CLI overrides
# --------------------------------------------------------------------------


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for a dotted key ``"a.b.c"``, creating
    intermediate dicts as needed."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"Cannot set '{dotted_key}': '{key}' is not a section")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config file and apply ``key=value`` CLI overrides on top.

    Args:
        path: Path to a YAML file, e.g. ``configs/attn_rotation.yaml``.
        overrides: Strings of the form ``"key=value"`` or ``"a.b.key=value"``
            for nested sections, e.g. from ``--set learning_rate=0.0002``.
            Values are parsed with ``yaml.safe_load`` so ``"1e-4"`` becomes a
            float, ``"true"`` becomes a bool, and plain strings stay strings.

    Returns:
        The merged config dict.

    Example:
        >>> cfg = load_config("configs/attn_rotation.yaml", ["learning_rate=2e-4"])
    """
    with open(path, "r") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like 'key=value'")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value)
        _set_nested(config, key.strip(), value)

    return config


def add_config_args(parser: Any) -> None:
    """Add the standard ``--config`` / ``--set`` arguments to an argparse
    parser. Kept in one place so every script's `--set key=value` behaves
    identically."""
    parser.add_argument("--config", type=Path, default=Path("configs/attn_rotation.yaml"), help="Path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="key=value",
        help="Override config values, e.g. --set learning_rate=2e-4 method=attn_rotation",
    )


# --------------------------------------------------------------------------
# Runtime-environment detection
# --------------------------------------------------------------------------


def runtime_environment() -> str:
    """Return ``"kaggle"``, ``"colab"``, or ``"local"`` based on environment
    markers. Used to pick sensible default paths without any config."""
    if "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir():
        return "colab"
    return "local"


def default_data_root() -> Path:
    """Default directory for dataset shards, per runtime environment."""
    env = runtime_environment()
    if env == "kaggle":
        return Path("/kaggle/working/data")
    if env == "colab":
        return Path("/content/data")
    return Path("data")


def default_checkpoint_dir() -> Path:
    """Default directory for local checkpoints, per runtime environment."""
    env = runtime_environment()
    if env == "kaggle":
        return Path("/kaggle/working/checkpoints")
    if env == "colab":
        return Path("/content/checkpoints")
    return Path("results/checkpoints")


def default_results_dir() -> Path:
    """Default directory for generated videos / eval outputs."""
    env = runtime_environment()
    if env == "kaggle":
        return Path("/kaggle/working/results")
    if env == "colab":
        return Path("/content/results")
    return Path("results")


# --------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------


def _to_uint8_frames(video: Any) -> "Any":
    """Normalize a video to a ``(F, H, W, C)`` uint8 numpy array.

    Accepts torch tensors shaped ``(F, C, H, W)`` or ``(C, F, H, W)`` or
    numpy arrays already shaped ``(F, H, W, C)``, in either ``[0, 1]``,
    ``[-1, 1]``, or ``[0, 255]`` range. This is the one place that range/
    layout guessing happens so the rest of the codebase can just call
    `save_video` with whatever a pipeline handed back.
    """
    import numpy as np

    try:
        import torch

        if isinstance(video, torch.Tensor):
            video = video.detach().float().cpu().numpy()
    except ImportError:
        pass

    video = np.asarray(video)

    if video.ndim != 4:
        raise ValueError(f"Expected a 4D video array, got shape {video.shape}")

    # Move channel axis to the end if it looks like (F, C, H, W) or (C, F, H, W).
    if video.shape[1] in (1, 3) and video.shape[-1] not in (1, 3):
        # Move channels last for imageio: (F, C, H, W) -> (F, H, W, C).
        video = np.transpose(video, (0, 2, 3, 1))
    elif video.shape[0] in (1, 3) and video.shape[-1] not in (1, 3):
        # Move channels last for imageio: (C, F, H, W) -> (F, H, W, C).
        video = np.transpose(video, (1, 2, 3, 0))

    if video.dtype != np.uint8:
        vmin, vmax = float(video.min()), float(video.max())
        if vmin < 0.0:  # assume [-1, 1]
            video = (video + 1.0) / 2.0
        elif vmax <= 1.0:  # assume [0, 1]
            pass
        else:  # already roughly [0, 255]
            video = video / 255.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)

    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)

    return video


def save_video(video: Any, path: str | Path, fps: int = 8) -> Path:
    """Write a video (torch tensor or numpy array, any of the layouts/ranges
    handled by `_to_uint8_frames`) to an mp4 file.

    Args:
        video: Frames as described above.
        path: Output ``.mp4`` path; parent directories are created.
        fps: Playback frame rate.

    Returns:
        The path written to.
    """
    import imageio.v3 as iio

    frames = _to_uint8_frames(video)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, frames, fps=fps, codec="libx264")
    return output


def save_comparison_video(
    left: Any,
    right: Any,
    path: str | Path,
    fps: int = 8,
    left_label: str = "student",
    right_label: str = "teacher (reversed)",
) -> Path:
    """Write a side-by-side comparison video: `left` and `right` stacked
    horizontally, frame by frame. Used to eyeball student vs. reversed-
    teacher output during evaluation.

    Args:
        left: Video A, same accepted layouts as `save_video`.
        right: Video B. Must have the same number of frames as `left`; if
            heights differ, `right` is resized to match `left`'s height.
        path: Output ``.mp4`` path.
        fps: Playback frame rate.
        left_label, right_label: Unused for now beyond documenting intent;
            kept as parameters so callers can pass them without the file
            format needing to embed text overlays.

    Returns:
        The path written to.
    """
    import numpy as np

    left_frames = _to_uint8_frames(left)
    right_frames = _to_uint8_frames(right)

    if left_frames.shape[0] != right_frames.shape[0]:
        raise ValueError(f"Frame count mismatch: {left_frames.shape[0]} vs {right_frames.shape[0]}")

    if left_frames.shape[1:3] != right_frames.shape[1:3]:
        # Nearest-neighbour resize of `right` to `left`'s (H, W) via simple index sampling,
        # so this helper has no extra image-library dependency.
        target_h, target_w = left_frames.shape[1:3]
        src_h, src_w = right_frames.shape[1:3]
        row_idx = (np.arange(target_h) * src_h // target_h)
        col_idx = (np.arange(target_w) * src_w // target_w)
        right_frames = right_frames[:, row_idx][:, :, col_idx]

    # Concatenate two (F, H, W, C) videos horizontally: width becomes 2W.
    combined = np.concatenate([left_frames, right_frames], axis=2)
    return save_video(combined, path, fps=fps)


# --------------------------------------------------------------------------
# Hugging Face Hub token lookup
# --------------------------------------------------------------------------


def get_hf_token() -> Optional[str]:
    """Find an ``HF_TOKEN`` from, in order: Kaggle Secrets, Colab userdata,
    then the ``HF_TOKEN`` environment variable. Returns None if not found
    anywhere (public repos still work without a token for read-only pulls).
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    return os.environ.get("HF_TOKEN")


# --------------------------------------------------------------------------
# Checkpoint persistence (local + Hugging Face Hub bridge)
# --------------------------------------------------------------------------


def checkpoint_path(directory: str | Path, name: str = "latest.pt") -> Path:
    """Return a consistent local checkpoint path, creating `directory`."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


class CheckpointManager:
    """Bridges local checkpoints and a private Hugging Face Hub model repo.

    Kaggle sessions cannot mount Google Drive and are killed after 12 hours,
    so a training run resumes by pulling the latest checkpoint from the Hub
    on startup, then periodically pushes its local checkpoint back up (at
    most once per `push_every_seconds`, plus whenever `save(..., force_push=True)`
    is called, e.g. at the very end of a run). Colab's `serve.py` only ever
    pulls, never pushes.

    Stored payload shape (see `save`): a dict with keys
    ``{"step": int, "model": state_dict, "optimizer": state_dict | None, "metadata": dict}``.
    """

    def __init__(
        self,
        local_dir: str | Path,
        repo_id: Optional[str] = None,
        push_every_seconds: float = 600.0,
        token: Optional[str] = None,
        hub_subfolder: Optional[str] = None,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.push_every_seconds = push_every_seconds
        self.token = token or get_hf_token()
        # None -> "latest.pt" at the repo root (the original layout);
        # e.g. "conv_mirror" -> "conv_mirror/latest.pt", so methods don't
        # overwrite each other's checkpoints on the Hub.
        self.hub_subfolder = hub_subfolder.strip("/") if hub_subfolder else None
        self._last_push = 0.0

    @property
    def local_path(self) -> Path:
        return self.local_dir / "latest.pt"

    @property
    def hub_filename(self) -> str:
        return f"{self.hub_subfolder}/latest.pt" if self.hub_subfolder else "latest.pt"

    def resume(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint payload, preferring a local file and
        falling back to a pull from the Hub. Returns None if neither exists
        (i.e. this is a fresh run)."""
        import torch

        if self.local_path.exists():
            return torch.load(self.local_path, map_location=map_location)

        if self.repo_id is None:
            return None

        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.hub_filename,
                token=self.token,
            )
        except Exception:
            return None

        payload = torch.load(downloaded, map_location=map_location)
        # Cache it locally so the next resume() doesn't hit the network.
        torch.save(payload, self.local_path)
        return payload

    def save(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        force_push: bool = False,
        **metadata: Any,
    ) -> Path:
        """Save a checkpoint locally, then push to the Hub if `force_push`
        or `push_every_seconds` have elapsed since the last push.

        Args:
            step: Training step this checkpoint was taken at.
            model_state: Typically ``model.state_dict()`` (or just the LoRA
                subset — see `student.py`).
            optimizer_state: Typically ``optimizer.state_dict()``.
            force_push: Push regardless of the time interval (use at the end
                of a run, or right before a Kaggle session is expected to die).
            **metadata: Extra small values to store alongside (e.g. method name).
        """
        import torch

        payload = {
            "step": step,
            "model": model_state,
            "optimizer": optimizer_state,
            "metadata": metadata,
        }
        torch.save(payload, self.local_path)

        now = time.time()
        if force_push or (now - self._last_push) >= self.push_every_seconds:
            self._push_to_hub()
            self._last_push = now

        return self.local_path

    def _push_to_hub(self) -> None:
        if self.repo_id is None:
            return
        push_file_to_hub(self.local_path, self.repo_id, self.hub_filename, token=self.token)


def push_file_to_hub(
    local_path: str | Path,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "model",
    token: Optional[str] = None,
    private: bool = True,
) -> None:
    """Upload a single file to a Hugging Face Hub repo, creating it if
    needed. Used both by `CheckpointManager` (repo_type="model") and by
    `scripts/build_dataset.py` to shard latent files to a dataset repo
    (repo_type="dataset") so they survive a Kaggle session ending.
    """
    from huggingface_hub import HfApi, create_repo

    token = token or get_hf_token()
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )


def list_hub_files(repo_id: str, repo_type: str = "dataset", token: Optional[str] = None) -> list:
    """List filenames present in a Hugging Face Hub repo. Used to skip
    already-uploaded dataset shards when resuming `build_dataset.py`.
    Returns an empty list if the repo doesn't exist yet or isn't reachable.
    """
    from huggingface_hub import HfApi

    token = token or get_hf_token()
    try:
        return HfApi().list_repo_files(repo_id, repo_type=repo_type, token=token)
    except Exception:
        return []

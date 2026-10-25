"""Generate report_assets/ figures and tables for the IE 643 written report.

Eval-only, CPU-safe, no GPU or model weights needed. Every asset is its own
function and logs one clear line when it skips (missing logs/results/data),
so this always finishes and exits 0 -- even run against an empty repo.

    python scripts/make_report_assets.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("make_report_assets")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"
LOG_LINE_RE = re.compile(r"step=(\d+)\s+loss=([\d.eE+-]+)(?:\s+step_time=([\d.]+)s)?")


def _moving_average(values: Sequence[float], window: int = 20) -> List[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        window_vals = values[lo : i + 1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
    return None


def _find_col(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for f in fieldnames:
        fl = f.lower()
        for cand in candidates:
            if cand in fl:
                return f
    return None


def loss_curves() -> None:
    """(a) Parse logs/*.log (plain text or Kaggle's JSON log export) for
    `step=N loss=X [step_time=Ys]` lines. One PNG (raw + moving avg) and one
    line in loss_summary.md per run; states the numbers, doesn't interpret."""
    logs_dir = ROOT / "logs"
    if not logs_dir.is_dir():
        logger.info("[loss_curves] skip: no logs/ directory (drop training logs in logs/*.log)")
        return
    log_files = sorted(logs_dir.glob("*.log")) + sorted(logs_dir.glob("*.json"))
    if not log_files:
        logger.info("[loss_curves] skip: logs/ exists but has no *.log or *.json files")
        return

    plt = _try_matplotlib()
    if plt is None:
        logger.info("[loss_curves] skip: matplotlib not installed (pip install -e '.[report]')")
        return

    OUT.mkdir(exist_ok=True)
    summary_lines = ["# Loss summary", ""]
    wrote_any = False

    for path in log_files:
        raw_text = path.read_text(errors="ignore")
        text = raw_text
        if path.suffix == ".json":
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                logger.info("[loss_curves] skip %s: not valid JSON", path.name)
                continue
            if not isinstance(payload, list):
                logger.info("[loss_curves] skip %s: expected a list of {\"data\": ...} objects", path.name)
                continue
            text = "\n".join(str(entry.get("data", "")) for entry in payload if isinstance(entry, dict))

        steps, losses, step_times = [], [], []
        for match in LOG_LINE_RE.finditer(text):
            step_str, loss_str, step_time_str = match.groups()
            try:
                loss_val = float(loss_str)
            except ValueError:
                continue
            if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
                continue  # non-finite, skip rather than plot garbage
            steps.append(int(step_str))
            losses.append(loss_val)
            if step_time_str:
                step_times.append(float(step_time_str))

        if not losses:
            logger.info("[loss_curves] skip %s: no 'step=N loss=X' lines matched", path.name)
            continue

        wrote_any = True
        run_name = path.stem
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, losses, ".", alpha=0.3, markersize=3, label="raw")
        ax.plot(steps, _moving_average(losses), linewidth=2, label="moving avg (20 steps)")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"{run_name} training loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"loss_{run_name}.png", dpi=150)
        plt.close(fig)

        first_100 = losses[:100]
        last_100 = losses[-100:]
        mean_first = sum(first_100) / len(first_100)
        mean_last = sum(last_100) / len(last_100)
        mean_step_time = sum(step_times) / len(step_times) if step_times else None

        summary_lines.append(f"## {run_name}")
        summary_lines.append(f"- steps parsed: {len(losses)}")
        summary_lines.append(f"- mean loss, first 100 steps: {mean_first:.6f}")
        summary_lines.append(f"- mean loss, last 100 steps: {mean_last:.6f}")
        if mean_step_time is not None:
            summary_lines.append(f"- mean step time: {mean_step_time:.2f}s ({len(step_times)} samples)")
        summary_lines.append(f"- plot: `report_assets/loss_{run_name}.png`")
        summary_lines.append("")
        logger.info(
            "[loss_curves] %s: %d steps, mean loss %.6f -> %.6f",
            run_name,
            len(losses),
            mean_first,
            mean_last,
        )

    if wrote_any:
        (OUT / "loss_summary.md").write_text("\n".join(summary_lines) + "\n")
        logger.info("[loss_curves] wrote report_assets/loss_summary.md")
    else:
        logger.info("[loss_curves] skip: none of logs/*.log|*.json had a usable loss line")


def metrics_table() -> None:
    """(b) Read comparison CSVs under results/** (column names may vary) and
    write a mean-over-prompts-and-seeds table per method as md + csv + LaTeX."""
    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        logger.info("[metrics_table] skip: no results/ directory")
        return
    csv_files = sorted(results_dir.rglob("*.csv"))
    if not csv_files:
        logger.info("[metrics_table] skip: no CSVs under results/")
        return

    rows = []
    for path in csv_files:
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    logger.info("[metrics_table] skip %s: no header row", path.name)
                    continue
                method_col = _find_col(reader.fieldnames, ["method"])
                cosine_col = _find_col(reader.fieldnames, ["direction_cosine", "direction cosine", "cosine"])
                motion_col = _find_col(reader.fieldnames, ["motion_ratio", "motion ratio"])
                pixel_col = _find_col(reader.fieldnames, ["pixel_diff", "pixel_difference", "pixel diff"])
                if method_col is None:
                    logger.info("[metrics_table] skip %s: no method column found", path.name)
                    continue
                for row in reader:
                    rows.append(
                        {
                            "method": row.get(method_col),
                            "direction_cosine": row.get(cosine_col) if cosine_col else None,
                            "motion_ratio": row.get(motion_col) if motion_col else None,
                            "pixel_diff": row.get(pixel_col) if pixel_col else None,
                        }
                    )
        except Exception as exc:
            logger.info("[metrics_table] skip %s: %s", path.name, exc)

    if not rows:
        logger.info("[metrics_table] skip: parsed 0 usable rows from %d CSV(s)", len(csv_files))
        return

    agg: Dict[str, Dict[str, object]] = {}
    for row in rows:
        method = row["method"] or "unknown"
        bucket = agg.setdefault(
            method, {"n": 0, "direction_cosine": [], "motion_ratio": [], "pixel_diff": []}
        )
        bucket["n"] += 1
        for key in ("direction_cosine", "motion_ratio", "pixel_diff"):
            value = row[key]
            if value not in (None, ""):
                try:
                    bucket[key].append(float(value))
                except ValueError:
                    pass

    headers = ["method", "n", "direction_cosine (mean)", "motion_ratio (mean)", "pixel_diff (mean)"]
    table_rows = []
    for method, stats in sorted(agg.items()):
        def mean_or_todo(key: str) -> str:
            values = stats[key]
            return f"{sum(values) / len(values):.4f}" if values else "TODO(user)"

        table_rows.append(
            [method, str(stats["n"]), mean_or_todo("direction_cosine"), mean_or_todo("motion_ratio"), mean_or_todo("pixel_diff")]
        )

    OUT.mkdir(exist_ok=True)

    md_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    md_lines += ["| " + " | ".join(r) + " |" for r in table_rows]
    (OUT / "metrics_table.md").write_text("\n".join(md_lines) + "\n")

    with (OUT / "metrics_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)

    latex_lines = [
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(h.replace("_", r"\_") for h in headers) + r" \\",
        "\\midrule",
    ]
    latex_lines += [" & ".join(r) + r" \\" for r in table_rows]
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "metrics_table.tex").write_text("\n".join(latex_lines) + "\n")

    logger.info(
        "[metrics_table] wrote metrics_table.{md,csv,tex} from %d row(s) across %d CSV(s)",
        len(rows),
        len(csv_files),
    )


def dataset_stats() -> None:
    """(c) Manifest stats from data/latents (local) or, if an HF token
    exists, a read-only download of manifest.jsonl from hf_dataset_repo."""
    candidates = []
    data_dir = ROOT / "data" / "latents"
    if data_dir.is_dir():
        candidates.append(data_dir)
        candidates.extend(p for p in sorted(data_dir.iterdir()) if p.is_dir())

    manifest_path = None
    for cand in candidates:
        p = cand / "manifest.jsonl"
        if p.exists():
            manifest_path = p
            break

    if manifest_path is None:
        token = _hf_token()
        if token is None:
            logger.info(
                "[dataset_stats] skip: no local data/latents*/manifest.jsonl and no HF_TOKEN for a read-only download"
            )
            return
        try:
            import yaml
            from huggingface_hub import hf_hub_download

            data_version = hf_dataset_repo = None
            for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                if cfg.get("data_version") and cfg.get("hf_dataset_repo"):
                    data_version = cfg["data_version"]
                    hf_dataset_repo = cfg["hf_dataset_repo"]
                    break
            if not data_version or not hf_dataset_repo:
                logger.info("[dataset_stats] skip: couldn't find data_version/hf_dataset_repo in configs/*.yaml")
                return
            downloaded = hf_hub_download(
                repo_id=hf_dataset_repo,
                filename=f"{data_version}/manifest.jsonl",
                repo_type="dataset",
                token=token,
            )
            manifest_path = Path(downloaded)
            logger.info("[dataset_stats] downloaded manifest from %s/%s (read-only)", hf_dataset_repo, data_version)
        except Exception as exc:
            logger.info("[dataset_stats] skip: HF read-only download failed (%s)", exc)
            return

    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not entries:
        logger.info("[dataset_stats] skip: manifest at %s is empty", manifest_path)
        return

    kept = [e for e in entries if e.get("kept", True)]
    prompts = sorted({e["prompt"] for e in kept if "prompt" in e})
    seeds_per_prompt: Dict[str, set] = {}
    for e in kept:
        seeds_per_prompt.setdefault(e.get("prompt"), set()).add(e.get("seed"))
    motion_scores = [e["motion_score"] for e in entries if "motion_score" in e]

    train_prompts: set = set()
    test_prompts: set = set()
    train_path = ROOT / "prompts" / "train.txt"
    test_path = ROOT / "prompts" / "test.txt"
    if train_path.exists():
        train_prompts = {line.strip() for line in train_path.read_text().splitlines() if line.strip()}
    if test_path.exists():
        test_prompts = {line.strip() for line in test_path.read_text().splitlines() if line.strip()}
    overlap = train_prompts & test_prompts

    OUT.mkdir(exist_ok=True)
    lines = ["# Dataset statistics", "", f"- manifest: `{manifest_path}`"]
    lines.append(f"- total manifest entries: {len(entries)}, kept (motion filter passed): {len(kept)}")
    lines.append(f"- distinct prompts (kept): {len(prompts)}")
    if seeds_per_prompt:
        counts = [len(v) for v in seeds_per_prompt.values()]
        lines.append(
            f"- seeds per prompt (kept): min={min(counts)}, median={statistics.median(counts)}, max={max(counts)}"
        )
    if motion_scores:
        lines.append(
            f"- motion_score over {len(motion_scores)} clips: "
            f"min={min(motion_scores):.3f}, mean={statistics.mean(motion_scores):.3f}, max={max(motion_scores):.3f}"
        )
    lines.append(f"- train.txt prompts: {len(train_prompts)}, test.txt prompts: {len(test_prompts)}")
    if overlap:
        lines.append(f"- **WARNING: {len(overlap)} prompt(s) overlap between train and test**: {sorted(overlap)[:5]}")
    else:
        lines.append("- train/test prompts: disjoint (assertion holds)")
    (OUT / "dataset_stats.md").write_text("\n".join(lines) + "\n")
    logger.info("[dataset_stats] wrote dataset_stats.md (%d kept clips, %d distinct prompts)", len(kept), len(prompts))

    if motion_scores:
        plt = _try_matplotlib()
        if plt is None:
            logger.info("[dataset_stats] motion-score histogram skipped: matplotlib not installed")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(motion_scores, bins=30)
            ax.set_xlabel("motion_score")
            ax.set_ylabel("count")
            ax.set_title("Motion score distribution")
            fig.tight_layout()
            fig.savefig(OUT / "motion_score_hist.png", dpi=150)
            plt.close(fig)
            logger.info("[dataset_stats] wrote report_assets/motion_score_hist.png")


def workflow_diagram() -> None:
    """(d) A clean block diagram (matplotlib only) of the pipeline, with the
    conv_mirror reference path drawn separately. Also writes docs/workflow.mmd."""
    plt = _try_matplotlib()
    if plt is None:
        logger.info("[workflow_diagram] skip: matplotlib not installed (pip install -e '.[report]')")
    else:
        import matplotlib.patches as mpatches

        OUT.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 6.5)
        ax.axis("off")

        def box(x, y, w, h, text, fc="#dfe7f5"):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#333333", linewidth=1
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

        def arrow(x1, y1, x2, y2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))

        box(0.3, 4.5, 1.6, 1, "prompts")
        box(2.2, 4.5, 1.9, 1, "teacher\n(frozen ModelScope)")
        box(4.4, 4.5, 1.8, 1, "forward latents")
        box(6.5, 4.5, 1.8, 1, "flip frame axis")
        box(8.6, 4.5, 2.0, 1, "reversed targets")

        box(2.2, 2.5, 1.9, 1, "student\n(LoRA, trainable)")
        box(4.4, 2.5, 1.8, 1, "inference\n(generate)")
        box(6.5, 2.5, 2.1, 1, "evaluation\nvs matched target")

        arrow(1.9, 5.0, 2.2, 5.0)
        arrow(4.1, 5.0, 4.4, 5.0)
        arrow(6.2, 5.0, 6.5, 5.0)
        arrow(8.3, 5.0, 8.6, 5.0)
        arrow(3.15, 4.5, 3.15, 3.5)
        arrow(4.1, 3.0, 4.4, 3.0)
        arrow(6.2, 3.0, 6.5, 3.0)
        arrow(9.6, 4.5, 8.0, 3.5)

        box(2.2, 0.4, 1.9, 1, "conv_mirror\n(flip temporal conv\nkernels, no training)", fc="#f5e3c8")
        arrow(3.15, 2.5, 3.15, 1.4)
        arrow(4.1, 0.9, 6.5, 2.7)

        ax.text(0.3, 5.9, "revT2V workflow", fontsize=13, weight="bold")
        ax.text(2.2, 1.6, "reference path (conv_mirror, training-free)", fontsize=8, style="italic", color="#7a5b24")

        fig.tight_layout()
        fig.savefig(OUT / "workflow_diagram.png", dpi=150)
        plt.close(fig)
        logger.info("[workflow_diagram] wrote report_assets/workflow_diagram.png")

    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    mermaid = """flowchart LR
    P[prompts] --> T[teacher: frozen ModelScope]
    T --> FL[forward latents]
    FL --> FX[flip frame axis]
    FX --> RT[reversed targets]
    RT --> S[student: LoRA, trainable]
    S --> INF[inference: generate]
    INF --> EV[evaluation vs matched target]
    RT -. reference .-> EV

    subgraph mirror [conv_mirror reference path, training-free]
      CO[conv_mirror: flip temporal conv kernels]
    end
    S -. compared against .-> CO
    CO -. reference .-> EV
"""
    (docs_dir / "workflow.mmd").write_text(mermaid)
    logger.info("[workflow_diagram] wrote docs/workflow.mmd")


def qualitative_figures() -> None:
    """(e) For each comparison video under results/**, export a frame sheet
    PNG (frames 0/5/10/15). Method labels are read from a sibling .json if
    present; never guessed or renamed."""
    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        logger.info("[qualitative_figures] skip: no results/ directory")
        return
    videos = [v for v in sorted(results_dir.rglob("*.mp4")) + sorted(results_dir.rglob("*.gif")) if OUT not in v.parents]
    if not videos:
        logger.info("[qualitative_figures] skip: no mp4/gif found under results/")
        return

    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio
        except ImportError:
            logger.info("[qualitative_figures] skip: imageio not installed")
            return

    plt = _try_matplotlib()
    if plt is None:
        logger.info("[qualitative_figures] skip: matplotlib not installed (pip install -e '.[report]')")
        return

    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for video_path in videos:
        try:
            reader = imageio.get_reader(video_path)
            # Some plugins report nframes=inf and never raise a clean
            # StopIteration, so cap the read instead of exhausting it.
            frames = []
            for i, frame in enumerate(reader):
                frames.append(frame)
                if i >= 199:
                    break
            reader.close()
        except Exception as exc:
            logger.info("[qualitative_figures] skip %s: couldn't decode (%r)", video_path.name, exc)
            continue
        if not frames:
            logger.info("[qualitative_figures] skip %s: zero frames decoded", video_path.name)
            continue

        indices = [i for i in (0, 5, 10, 15) if i < len(frames)] or [0]
        labels = None
        labels_path = video_path.with_suffix(".json")
        if labels_path.exists():
            try:
                labels = json.loads(labels_path.read_text())
            except Exception:
                labels = None

        fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
        axes = [axes] if len(indices) == 1 else axes
        for ax, idx in zip(axes, indices):
            ax.imshow(frames[idx])
            ax.set_title(f"frame {idx}")
            ax.axis("off")
        title = video_path.relative_to(results_dir).as_posix()
        if labels:
            title += f"  (columns, as documented: {labels})"
        else:
            title += "  (method/column order undocumented -- not guessed)"
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        out_name = "frames_" + video_path.relative_to(results_dir).as_posix().replace("/", "_") + ".png"
        fig.savefig(frames_dir / out_name, dpi=150)
        plt.close(fig)
        written += 1
        logger.info("[qualitative_figures] wrote report_assets/frames/%s", out_name)

    if not written:
        logger.info("[qualitative_figures] skip: 0 of %d video(s) produced a frame sheet", len(videos))


def copy_curated_artifacts() -> None:
    """(f) Copy final comparison mp4/gif under results/** into
    report_assets/videos/ (copy, never move; skip anything over 20 MB)."""
    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        logger.info("[copy_curated_artifacts] skip: no results/ directory")
        return
    media = [m for m in sorted(results_dir.rglob("*.mp4")) + sorted(results_dir.rglob("*.gif")) if OUT not in m.parents]
    if not media:
        logger.info("[copy_curated_artifacts] skip: no mp4/gif under results/")
        return

    dest_dir = OUT / "videos"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied, skipped_big = 0, 0
    for path in media:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 20:
            skipped_big += 1
            logger.info("[copy_curated_artifacts] skip %s: %.1f MB > 20 MB limit", path.name, size_mb)
            continue
        dest_name = path.relative_to(results_dir).as_posix().replace("/", "__")
        shutil.copy2(path, dest_dir / dest_name)
        copied += 1
    logger.info(
        "[copy_curated_artifacts] copied %d file(s) into report_assets/videos/ (%d skipped for size)",
        copied,
        skipped_big,
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    steps = [loss_curves, metrics_table, dataset_stats, workflow_diagram, qualitative_figures, copy_curated_artifacts]
    for step in steps:
        try:
            step()
        except Exception as exc:  # never let one asset's failure kill the run
            logger.info("[%s] FAILED, skipping (%s)", step.__name__, exc)
    logger.info("Done. See report_assets/ for generated files.")


if __name__ == "__main__":
    main()

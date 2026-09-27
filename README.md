# revT2V

IE 643 course project: distilling a video diffusion model to generate
**reverse-time** video directly (not by generating forward and reversing).

- Teacher: [`ali-vilab/text-to-video-ms-1.7b`](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) (ModelScope T2V), frozen.
- Student: the same U-Net + a LoRA adapter on its temporal attention layers, trained on reverse-time targets.
- Three methods, selected via `--method`: `baseline`, `attn_rotation` (primary — Wang et al., *Generative Inbetweening*, ICLR 2025), `motion_prior` (secondary — Jeon et al., *Motion Prior Distillation*, ICLR 2026).

New to this repo? Start with **[STRUCTURE.md](STRUCTURE.md)** (what everything is) and **[LEARNING.md](LEARNING.md)** (what order to implement things in).

## Setup

```bash
git clone <this repo>
cd revT2V
pip install torch --index-url <the build matching your CUDA/driver>  # see note below
pip install -e ".[dev]"
pytest  # should fail with NotImplementedError, not ImportError, until you implement the scaffold
```

`torch` is intentionally not a dependency of this package (see `pyproject.toml`) — Kaggle and Colab ship a preinstalled build matched to their GPU driver, and installing a different one on top routinely breaks CUDA. On Kaggle/Colab you can usually skip the `pip install torch` line entirely and go straight to `pip install -e .`.

## Workflow

Code locally in VS Code, push to git, pull and run on Kaggle (training) or Colab (testing/demo):

```
VS Code (edit) --push--> git remote --pull--> Kaggle (scripts/train.py, GPU, 12h sessions)
                                          \--> Colab (scripts/serve.py + Cloudflare tunnel)
```

Neither Kaggle nor Colab gives you persistent storage across sessions, so a private Hugging Face Hub repo is the bridge: `scripts/train.py` pulls the latest checkpoint from the Hub on startup and pushes back periodically (and once at the end), so restarting a Kaggle session — planned or not — resumes instead of starting over. `scripts/build_dataset.py` does the same for dataset shards. See `revt2v/utils.py`'s `CheckpointManager` and `LEARNING.md`'s "how the workflow works" section for the full picture.

Compute is free-tier only: Kaggle T4/P100 (~30 GPU-hours/week, 12h/session) for training; Colab (whatever GPU you're given) for serving a trained checkpoint behind a `cloudflared` quick tunnel for a live demo — see `notebooks/kaggle_train.ipynb` and `notebooks/colab_test.ipynb`.

## Repo layout

```
revt2v/       library code — teacher, student, data, infer, methods/
scripts/      CLI entry points — build_dataset, train, evaluate, serve
configs/      one YAML per method + shared defaults
prompts/      train.txt / test.txt prompt lists (held-out, no overlap)
notebooks/    kaggle_train, colab_test
tests/        CPU-only unit tests, no model download
results/      generated videos, eval metrics (gitignored)
```

(`p.py` at the repo root is an ad hoc scratch script for exploring the
teacher pipeline — see STRUCTURE.md's "Root files" section. There's no
`notebooks/scratch.ipynb`; that role moved to `p.py`.)

Full file-by-file guide: [STRUCTURE.md](STRUCTURE.md). Implementation order and what "done" looks like at each step: [LEARNING.md](LEARNING.md).

## What's implemented vs. what you write

Infrastructure (fully implemented): `revt2v/utils.py`, all of `scripts/`'s CLI/orchestration, the notebooks, prompts, configs.

Learning scaffold (you write the bodies — every function has a docstring with shapes/dtypes and numbered TODOs, no solution code): `revt2v/teacher.py`, `student.py`, `data.py`, `infer.py`, `methods/*.py`, and the metric functions in `scripts/evaluate.py`. (There's no `losses.py` — noise sampling and MSE loss are written directly in `scripts/train.py` and `methods/*.py`.)

## Versions this was written against

Check these against what's actually installed before trusting an exact API signature — diffusers in particular has moved things across releases:

- `diffusers>=0.27`, `transformers>=4.38`, `accelerate>=0.28`, `peft>=0.10`, `huggingface_hub>=0.21` (see `pyproject.toml` for the full pin list)
- Torch: whatever Kaggle/Colab preinstalls (not pinned by this project)
- Python: `>=3.10`

If `scripts/evaluate.py`'s `compute_motion_consistency` ends up using OpenCV for optical flow, note the version you used here too — it isn't a listed dependency.

# revT2V

**Distilling Video Diffusion Models for Reverse-Time Video Generation** — IE 643 (Deep Learning) course project, IIT Bombay.

A frozen ModelScope T2V (`ali-vilab/text-to-video-ms-1.7b`) **teacher** generates normal forward-time video from a text prompt. This project studies how to get the same model to generate the same scene **running backwards in time**, and distills that behaviour into a small **student** (LoRA on the frozen teacher) so the arrow of time can be flipped without a second full model.

**Core finding:** the arrow of time in this U-Net lives in its **temporal convolutions** (`TemporalConvLayer`'s `Conv3d`s), not in temporal attention. Flipping every temporal conv kernel in time turns the teacher into an exact, training-free time-mirror (`conv_oracle`) — this is the project's reference/upper bound, not a deliverable. Rotating or injecting temporal *attention* alone changes motion direction but not pose/structure, and only matters downstream of those direction-aware convs. Blending conv kernels partially (e.g. 50/50 with their flip) breaks the model outright — flips are all-or-nothing.

## Methods

| id | what it does | trained? | role |
|---|---|---|---|
| `conv_oracle` | flips every temporal `Conv3d` kernel in time; no LoRA, no training | no (training-free) | **oracle / upper bound** — architecturally equivalent to generating from time-flipped noise and playing it backward |
| `conv_lora` | LoRA on all 88 temporal `Conv3d`s + 68 temporal attention `to_v`/`to_out.0` Linears, trained with ε-MSE + a mirror loss against the (unflipped) teacher's own time-flipped prediction | yes (1000 steps) | **main method** (TA-approved) — see [Known limitations](#known-limitations): its first full run showed a flat loss, so it is **not** a demonstrated success yet |
| `attn_lora` | same loss/setup as `conv_lora`, but LoRA on the 68 temporal attention Linears only, no conv targets | yes | **baseline**, isolates what attention-only LoRA can do without touching the convs |
| `attn_injection` | two forward passes per step: capture stock attention on `flip(x_t)`, inject it (rotated 180°) into the real pass's own `to_v`/`to_out.0` | zero-shot (no LoRA needed to run; a LoRA on top is trainable) | exploratory — see `revt2v/methods/attn_injection.py` docstring for why this is a no-op at `transformer_in` and only does something real downstream of the temporal convs |

An earlier method, `attn_rotation` (self-rotation + LoRA on temporal attention only, no conv involvement), was tried and trained but was superseded by `attn_injection`/`conv_lora` once the conv-kernel finding came in; its code was removed from the repo. See `docs/REPORT_NOTES.md` §6 for that experiment's outcome.

## Repo map

See [STRUCTURE.md](STRUCTURE.md) for the full folder-by-folder breakdown. Short version:

| folder | purpose |
|---|---|
| `revt2v/` | model code — teacher loading, student/LoRA building, data, and `methods/` |
| `scripts/` | CLI entry points — `train.py`, `evaluate.py`, `build_dataset.py`, `conv_rank_check.py`, `serve.py` |
| `configs/` | one YAML per method/run |
| `server/` | FastAPI inference backend, meant to run on Colab (GPU) |
| `client/` | React + Vite frontend |
| `tests/` | CPU-only pytest suite |
| `notebooks/` | dataset building and method-comparison notebooks |
| `kaggle/` | Kaggle launcher notebook for training on Kaggle's free GPU |
| `prompts/` | train/test prompt lists |
| `results/` | generated videos, local checkpoint copies, eval output (gitignored except `.gitkeep`) |
| `report_assets/` | generated figures/tables for the written report (see `scripts/make_report_assets.py`) |

## Environment

- Python: `>=3.10` (per `pyproject.toml`); torch is **not** pinned in `pyproject.toml` on purpose — Kaggle/Colab ship a preinstalled torch build matched to their GPU/driver, and installing a different one on top routinely breaks CUDA. Install torch yourself (or trust the notebook environment) before `pip install -e .`.
- Node: 20+ (for `client/`).
- Exact torch/diffusers/transformers/peft versions and hardware used for the actual training runs (Kaggle T4) and the final evaluation (Colab T4): `TODO(user)` — fill in from the Kaggle/Colab run logs once dropped into `logs/`.
- Key pinned deps (see `pyproject.toml`): `diffusers>=0.27`, `transformers>=4.38`, `accelerate>=0.28`, `peft>=0.21` (needs Conv3d LoRA support), `fastapi>=0.110`, `uvicorn>=0.29`.

## Reproduce

```bash
pip install -e .
pytest -q   # CPU-only, no GPU or model download needed
```

**1. Build the dataset** (needs a GPU; produces reversed-video latents + text embeddings):

```bash
python scripts/build_dataset.py --prompts prompts/train.txt --test-prompts prompts/test.txt
```

See `notebooks/build_dataset.ipynb` for the notebook version.

**2. Train a method** (Kaggle, GPU — see `kaggle/nb/revt2v.ipynb`, which drives this same script):

```bash
python scripts/train.py --config configs/conv_lora.yaml
python scripts/train.py --config configs/attn_lora.yaml
python scripts/train.py --config configs/attn_injection.yaml
```

`scripts/conv_rank_check.py` recommends a LoRA rank from the temporal convs' energy spectrum before training (`--target` energy fraction, default 0.9).

**3. Evaluate** (Colab, GPU — see `notebooks/compare_methods.ipynb` / `notebooks/final_eval.ipynb`):

```bash
python scripts/evaluate.py --prompts prompts/test.txt --methods conv_oracle conv_lora attn_lora attn_injection
```

Scores FVD/CLIP/optical-flow/VBench-style metrics against the matched (ground-truth reversed) target.

**4. Run the server** (Colab, GPU; see `server/README.md` for the full tunnel setup):

```bash
pip install -e .
uvicorn server.app:app --reload --port 8000
```

**5. Run the client:**

```bash
cd client && npm install && npm run dev
```

Works fully offline against an in-browser mock without a GPU (`VITE_USE_MOCK=true`, the default).

## Links

- GitHub: https://github.com/silentlooop/revT2V
- HF model checkpoints: `silentlooop/revt2v-ckpt` (`TODO(user)`: confirm/paste the HF repo URL — `https://huggingface.co/silentlooop/revt2v-ckpt`)
- HF dataset: `silentlooop/revt2v-data` (`TODO(user)`: confirm/paste the HF repo URL — `https://huggingface.co/datasets/silentlooop/revt2v-data`)
- Live demo / server URL: `TODO(user)` — Colab tunnel URLs are ephemeral, paste the current one if you want this filled in

## Known limitations

- `conv_oracle` is **architecturally equivalent** to flipping the input noise in time and playing the teacher's normal forward output backward — it is not a learned result, and is the oracle/upper bound every trained method is compared against, not a deliverable in itself. Its exactness (`unet_flipped(z) == flip(unet(flip(z)))`) is asserted in `tests/test_conv_mirror.py`, but the stronger end-to-end numeric check (video-level mean pixel difference vs. the matched target) is run in `notebooks/final_eval.ipynb` — until that output exists, treat "exact mirror" as **unverified**, not proven, at the full-pipeline level.
- The trained students (`conv_lora`, `attn_lora`, `attn_injection`+LoRA) are meant to **approximate** the oracle without needing the kernel flip at inference time. `conv_lora`'s first full 1000-step run showed a **flat loss curve** — see `docs/REPORT_NOTES.md` §6 for the numbers once `logs/conv_lora.log` is dropped in and `scripts/make_report_assets.py` is run. Until a results file shows otherwise, **"the conv_lora student works" is not a supported claim** (see the claims table in `docs/REPORT_NOTES.md`).
- Partial kernel blends (α·W + (1-α)·flip(W) for α ∈ {0.25, 0.5}) were tried and **break the model** — flips must be all-or-nothing per layer. This is enforced in code (no blending parameter exists in `conv_mirror.py`).
- `attn_injection` reduces algebraically to plain, un-rotated attention when applied at `transformer_in` (nothing direction-aware upstream yet) — it only does something different from the base model when applied downstream of the direction-aware temporal convs (`rotate_layers="up_attn1"`, the default).

# Report notes

Facts pre-filled from the repo (configs, manifests, code), mapped to the IE 643 report template. `TODO(user)` marks anything that needs a number from a log or the final evaluation that isn't in the repo yet — never filled with an invented value.

---

## 1. Introduction

- ModelScope T2V (`ali-vilab/text-to-video-ms-1.7b`) is a frozen teacher, forward-time video generation.
- Goal: generate the same scene running backwards in time, without a second full model — distill reverse-time behaviour into a small adapter (LoRA) on the same frozen weights.
- `TODO(user)`: 1-2 sentences of motivation/applications for the written intro (not something the repo states).

## 2. Project Workflow

- prompts → teacher (frozen) → forward latents → flip frame axis → reversed targets (the training signal) → student (LoRA, trainable) → inference (generate) → evaluation vs. matched target.
- `conv_oracle` is a separate reference path: teacher weights with every temporal `Conv3d` kernel flipped, no training, no LoRA — used only as the oracle/upper bound the trained students are compared against, not a deliverable itself.
- Diagram: `report_assets/workflow_diagram.png` (generate with `python scripts/make_report_assets.py`). Mermaid source: `docs/workflow.mmd`.

## 3. Literature Survey

One line each (verify details against the source before citing in the final writeup — only the arXiv IDs below are repo-verified; see `docs/references.bib`):

- **ModelScope T2V** (arXiv:2308.06571) — the frozen teacher model used throughout; forward text-to-video diffusion with a spatio-temporal U-Net.
- **LoRA** (arXiv:2106.09685) — low-rank adapter fine-tuning; the mechanism every trained method in this repo uses instead of full fine-tuning.
- **Generative Inbetweening** — `TODO(user)`: verify authors/venue/arXiv id, one line on relevance (frame interpolation via generative video models).
- **VTG (Video Temporal Grounding)** — `TODO(user)`: verify, one line on relevance.
- **Motion Prior Distillation** — `TODO(user)`: verify, one line on relevance.
- **CVFI** — `TODO(user)`: verify, one line on relevance.
- **UniTemp** (arXiv:2606.18702) — generation-order / time-reversed-content work; note this is about the *order generation happens in*, not time-reversed *content* — distinguish carefully from this project's goal.
- **VMC** — `TODO(user)`: verify, one line on relevance.
- **BAgger** — `TODO(user)`: verify, one line on relevance.
- **ViBiDSampler** — `TODO(user)`: verify, one line on relevance.
- **VBench** (arXiv:2311.17982) — benchmark suite for video generation; this repo uses its "dynamic degree" metric (see `scripts/evaluate.py`), optional dependency (`pip install --no-deps vbench easydict`), skipped with a warning if not importable.
- **FVD** (arXiv:1812.01717) and its content bias — Fréchet distance between I3D (Kinetics-400) features; `scripts/evaluate.py` notes it's unreliable in absolute terms below 50 videos (directional only at this project's scale).
- **CLIPScore** (arXiv:2104.08718) — `scripts/evaluate.py`'s `clip_score`: `100 * CLIP cosine(frame, prompt)`, averaged over frames, using `openai/clip-vit-large-patch14`.
- **RAFT** (arXiv:2003.12039) / **Farneback** — optical flow; `scripts/evaluate.py`'s `flow_cosine` metric uses RAFT (torchvision) if available, else OpenCV Farneback, always at 256x256.

## 4. Proposed Approach

### 4.1 Before prep-presentation review

- `attn_rotation`: self-rotation + LoRA on temporal attention only, no conv involvement. Trained. Superseded once the conv-kernel finding (below) came in; code since removed from the repo (was in `revt2v/methods/attn_rotation.py` + `configs/attn_rotation.yaml`, both deleted). `TODO(user)`: pull its outcome/numbers from whatever log or notes exist outside this repo, if you want to report them — none are checked in here.
- `conv_oracle` (originally "convo method"): the training-free exact time-mirror via conv-kernel flipping. This produced the project's key finding: **the arrow of time lives in the temporal convolutions**, not in temporal attention.

### 4.2 After prep-presentation review

- `attn_injection`: cross-model attention injection (capture on `flip(x_t)` with stock attention, inject rotated 180° into the real pass's own `to_v`/`to_out.0`). Zero-shot capable; a LoRA on top is trainable. Only has an effect downstream of the direction-aware temporal convs (default `rotate_layers: up_attn1`) — at `transformer_in` it provably reduces to plain attention (see `revt2v/methods/attn_injection.py` module docstring).
- `conv_lora` (main, TA-approved): LoRA on all 88 temporal `Conv3d`s + 68 temporal attention `to_v`/`to_out.0` Linears, trained with ε-MSE + a mirror loss against the (unflipped) teacher's own time-flipped prediction.
- `attn_lora`: the same loss/setup as `conv_lora`, attention-only LoRA target, as a baseline to isolate what attention alone can do without the convs.

## 5. Dataset Details

From the manifest (`dataset2` on `silentlooop/revt2v-data`, read via `scripts/make_report_assets.py`'s `dataset_stats()` — rerun it for current numbers, these are a snapshot):

- Storage: latents + text embeddings per clip (`revt2v/data.py`'s `build_dataset_record`: `latents`, `reversed_latents`, `prompt_embeds`, `negative_prompt_embeds`), saved as one `.pt` shard per clip plus a `manifest.jsonl` (`shard`, `prompt`, `seed`, `motion_score`, `kept`).
- Prompt style: short, single-scene, physically-grounded prompts chosen so a *correct* time reversal is visually obvious (e.g. "orange juice being poured into a glass", "Slow motion of a soap bubble popping").
- Motion filter: `scripts/build_dataset.py` computes `motion_score` (mean Farneback optical-flow magnitude on a 128px grayscale clip) and drops clips below `--min-motion` (default 0.15) — recorded as `kept: false` in the manifest, not deleted outright.
- Snapshot numbers (from `report_assets/dataset_stats.md`): 447 manifest entries, 416 kept; 147 distinct kept prompts; seeds per prompt: min 1 / median 3 / max 3; `motion_score` over 447 clips: min 0.047, mean 1.831, max 16.108.
- Train/test split: `prompts/train.txt` (149 prompts) / `prompts/test.txt` (10 prompts), asserted disjoint by `scripts/build_dataset.py --test-prompts` and by `tests/test_scripts.py::test_prompt_files`. Note: 149 train-file prompts but only 147 distinct *kept* prompts in the manifest — some train prompts may have 0 kept clips after the motion filter; `TODO(user)`: confirm which 2, if it matters for the writeup.
- `TODO(user)`: `--seeds-per-prompt` actually used for this dataset build (CLI default 3, matches the manifest's median-3 seeds/prompt, consistent with the default).

## 6. Experiments (every one, including negative results)

| experiment | outcome |
|---|---|
| Zero-shot self-rotation (no training) | `TODO(user)`: outcome/numbers not in this repo — pull from notes if reporting it |
| `attn_rotation` (self-rotation + LoRA, temporal attention only) | trained; superseded, code removed from repo once the conv-kernel finding landed. `TODO(user)` for numbers. |
| Teacher-map injection, zero-shot (`attn_injection` with no LoRA) | runs, used as `attn_injection`'s zero-shot fallback when no checkpoint exists (see `revt2v.infer.MethodBank`) |
| **Partial conv-kernel blends** (α·W + (1−α)·flip(W), α ∈ {0.25, 0.5}) | **broke the model outright** — flips must be all-or-nothing per layer; no blend parameter exists anywhere in `conv_mirror.py` by design |
| `conv_oracle` (all temporal conv kernels flipped, no training) | exact time-mirror by construction (`unet_flipped(z) == flip(unet(flip(z)))`), asserted in `tests/test_conv_mirror.py`; full end-to-end numeric exactness check (video-level mean pixel diff vs. matched target) runs in `notebooks/final_eval.ipynb` §1 — **unverified at that level until that output exists** |
| `conv_lora` (main, 1000 steps, LoRA on 88 convs + 68 attn Linears) | **training completed, but the first full run showed a flat loss curve** — not a demonstrated success. Numbers: `TODO(user)` from `logs/conv_lora.log` via `scripts/make_report_assets.py`'s `loss_curves()`. At inference, the checkpoint loads and generates videos without error (verified in `notebooks/compare_methods.ipynb`), but that is not evidence of *correct* reverse-time behaviour — see the claims table below. |
| `attn_lora` (baseline, attention-only LoRA) | `TODO(user)`: training outcome/numbers not yet in `logs/` |
| `attn_injection` + LoRA (trained) | `TODO(user)`: training outcome/numbers not yet in `logs/` |

Training procedure (shared by `conv_lora`/`attn_lora`/`attn_injection`, from `configs/*.yaml`): seed 42, `ali-vilab/text-to-video-ms-1.7b`, 16 frames @ 256x256, 50 inference steps / guidance 9.0 for sampling during training; LoRA rank 16 / alpha 16 / dropout 0; loss = ε-MSE + `mirror_loss_weight=1.0` × MSE(ε_student(x_t), flip(ε_teacher(flip(x_t)))); AdamW, lr 1e-4; batch size 1, grad-accum 8 (effective batch 8); 1000 steps; 10% unconditional dropout (`uncond_prob: 0.1`); checkpoints every 100 steps, pushed to `silentlooop/revt2v-ckpt/<method>/` every 600s. `attn_injection` additionally sets `rotate_layers: up_attn1` (9 layers) and runs ~30-40% slower per step (extra no-grad capture pass per micro-batch).

Hardware: Kaggle free-tier T4 for training, Colab free-tier T4 for evaluation (per `README.md`/this project's stated setup). Exact torch/diffusers/CUDA versions actually used: `TODO(user)` — fill in once Kaggle/Colab run logs are dropped into `logs/`.

Checkpoint locations: `silentlooop/revt2v-ckpt/{conv_lora,attn_lora,attn_injection}/latest.pt` (never the repo root).

## 7. Results

- Loss curves: `report_assets/loss_<run>.png` + `report_assets/loss_summary.md` — `TODO(user)`: generate by dropping Kaggle logs into `logs/*.log` and running `python scripts/make_report_assets.py`. **No numbers exist in this repo yet.**
- Metrics table (FVD / CLIP score / VBench dynamic degree / flow cosine, per method, mean over prompts and seeds): `report_assets/metrics_table.{md,csv,tex}` — `TODO(user)`: generate from `notebooks/final_eval.ipynb`'s output CSVs under `results/final/`, then rerun `make_report_assets.py`.
- Frame sheets / qualitative comparisons: `report_assets/frames/*.png`, curated videos in `report_assets/videos/` — `TODO(user)`: populate `results/final/` via `notebooks/final_eval.ipynb` first.
- Dataset statistics: `report_assets/dataset_stats.md`, `report_assets/motion_score_hist.png` (already generated from the live manifest — see §5 above for the current snapshot).
- Workflow diagram: `report_assets/workflow_diagram.png`.
- rho diagnostic (per trained student, per fixed timestep): `results/final/rho_diagnostic.csv` — `TODO(user)`: generate via `notebooks/final_eval.ipynb` §4.
- `conv_rank_check.py` summary: `results/final/rank_check.txt` — `TODO(user)`: generate via `notebooks/final_eval.ipynb` §4.

## 8. Plan for Novelty Assessment / Future Work

- Antisymmetric temporal adapter (structural constraint instead of a learned-from-scratch LoRA).
- Mirror-only, direction-aware distillation (separate the mirror loss from the ε-MSE term more cleanly).
- Few-step distillation of the oracle (TA-approved direction).
- FVD at scale (current sample count is too small for FVD's absolute numbers to be meaningful — see §3/§6).
- Longer training (1000 steps produced a flat loss for `conv_lora`'s first run; more steps, a different LR, or a different mirror-loss weight are untried).

## 9. Conclusion

`TODO(user)`: write once §6/§7 numbers exist — the honest version of this section depends on whether `conv_lora`'s loss issue was fixed by the final run or not.

---

## Claims we can and cannot make

| claim | evidence file | status |
|---|---|---|
| The arrow of time lives in the temporal convolutions, not temporal attention | `revt2v/methods/conv_mirror.py` docstring + `tests/test_conv_mirror.py` (unet-level exactness, unit test) | **verified** (at the unit/architectural level) |
| `conv_oracle` is an exact time-mirror of the teacher | `tests/test_conv_mirror.py` (unet level, synthetic input) | **unverified** at the full video-generation level until `notebooks/final_eval.ipynb` §1's video-level mean-pixel-diff output exists |
| Partial conv-kernel blends break the model | design decision encoded directly in `conv_mirror.py` (no blend parameter exists) | **verified** (as a design constraint; the specific broken-output evidence itself isn't a checked-in artifact — `TODO(user)` if you want to cite the actual broken samples) |
| `attn_injection` is a no-op at `transformer_in` | `revt2v/methods/attn_injection.py` docstring, algebraic argument; `tests/test_attn_injection.py` | **verified** |
| The conv_lora student works (produces correct reverse-time video) | — | **NOT supported** — its first full run showed a flat loss curve; it loads and generates without crashing (`notebooks/compare_methods.ipynb`), which is not evidence of correctness |
| The attn_lora / attn_injection+LoRA students work | — | **NOT supported** — no training-outcome log or metrics file exists in this repo yet |

# STRUCTURE.md — a beginner's guide to this repo

If you're new to how an ML research repo is laid out, this file is for
you. It explains the big picture, then walks every file. For *what order*
to implement things in, see [LEARNING.md](LEARNING.md) instead — this file
is a map, that one's an itinerary.

## Big picture

The project has five stages, run in this order:

1. **Teacher generates data.** A frozen, pretrained text-to-video model
   (`ali-vilab/text-to-video-ms-1.7b`) generates a normal, forward-time
   video for each training prompt. We keep its internal *latents* (a
   compressed representation, not pixels) rather than the pixels
   themselves, and also flip them to get a reverse-time *target*.
2. **Student trains.** A small trainable adapter (LoRA) is attached to a
   copy of the teacher's network, and trained so that, given the same kind
   of noisy-latent input the teacher was trained on, it predicts toward
   the *reversed* target instead of the forward one.
3. **Student generates directly.** At inference time, the student starts
   from pure noise and denoises straight into a reverse-time video — it
   never generates forward and flips the result.
4. **Evaluation** compares the student's direct reverse-time output
   against the reversed teacher output on prompts neither was tuned on.
5. **Serving** exposes a trained student behind a small web API so it can
   be demoed from a browser.

```
prompts/train.txt                                    prompts/test.txt
      |                                                      |
      v                                                      v
teacher.py --(forward gen)--> data.py --(flip)--> .pt shards on disk/HF
                                                        |
                                                        v
                                          student.py + methods/*
                                          (scripts/train.py loop)
                                                        |
                                                        v
                                        LoRA checkpoint --(HF Hub)--> .pt
                                                        |
                        +-------------------------------+-------------------+
                        v                                                   v
              infer.py (scripts/evaluate.py)                    infer.py (scripts/serve.py)
              -> videos + FVD/CLIP/consistency metrics           -> mp4 served over HTTP
```

Prompts flow in text; everything in the middle is *latents* (small tensors,
not pixels) until something needs to be watched or scored, at which point
`teacher.py`'s VAE decodes latents back to pixels.

## File-by-file

### `revt2v/` — the library

#### `revt2v/__init__.py`
Marks `revt2v` as an importable Python package and lists its submodules.
Nothing to implement. Imported by: everything that does `import revt2v` or
`from revt2v import ...`.

#### `revt2v/utils.py` — fully implemented infrastructure
Shared helpers every other file leans on, so they don't each reinvent
"how do I load a config" or "how do I save a checkpoint."
- `seed_everything(seed)` — makes runs reproducible.
- `load_config(path, overrides)` / `add_config_args(parser)` — YAML config
  loading with `--set key=value` CLI overrides.
- `runtime_environment()`, `default_data_root()`, `default_checkpoint_dir()`,
  `default_results_dir()` — pick sensible paths depending on whether you're
  on Kaggle, Colab, or your own machine.
- `save_video`, `save_comparison_video` — write an mp4 (or a side-by-side
  comparison mp4) from a tensor/array in whatever layout a pipeline handed
  back.
- `get_hf_token()` — finds your Hugging Face token from Kaggle Secrets,
  Colab userdata, or the `HF_TOKEN` env var, in that order.
- `CheckpointManager` — the class that bridges local checkpoints and a
  private HF Hub repo (see "Data & artifacts" below).
- `push_file_to_hub`, `list_hub_files` — lower-level Hub helpers,
  used directly by `scripts/build_dataset.py` for dataset shards.

Imports from: nothing in this repo (it's the foundation). Imported by:
nearly everything else. Runs: everywhere (local, Kaggle, Colab). Status:
fully implemented, 0 TODOs.

#### `revt2v/teacher.py` — learning scaffold (5 functions, 18 TODOs)
Loading and running the frozen ModelScope T2V pipeline: `load_teacher`,
`encode_prompt`, `generate_forward_video`, `decode_latents_to_video`,
`encode_video_to_latents`. Imports from: `torch`, `diffusers` (not other
revt2v files). Imported by: `data.py`, `infer.py`, `scripts/build_dataset.py`,
`scripts/evaluate.py`, ad hoc scratch scripts (see `p.py`). Runs: Kaggle/Colab
only (needs a GPU). Touch it: LEARNING.md step 1.

#### `revt2v/student.py` — learning scaffold (4 functions, 7 TODOs)
Wraps the teacher's U-Net with a LoRA adapter (`build_student`), runs one
forward pass (`predict_noise`), and extracts/loads just the LoRA weights
for cheap checkpoints (`lora_state_dict`, `load_lora_state_dict`). Imports
from: `torch`, `peft` (not other revt2v files, though it *takes* a
teacher pipeline as an argument). Imported by: `scripts/train.py`,
`infer.py`. Runs: Kaggle (training) and Colab (loading a checkpoint for
inference). Touch it: LEARNING.md step 3.

#### `revt2v/data.py` — learning scaffold (5 functions/classes, 9 TODOs)
Turns teacher output into training examples: `flip_latents_time_axis`
(the core "reverse-time target" operation — pure tensor math, no model
needed), `build_dataset_record` (calls into `teacher.py` for one prompt),
`LatentDataset` (a `torch.utils.data.Dataset` reading the `.pt` files back
for training). Imports from: `teacher.py`, `torch`. Imported by:
`scripts/build_dataset.py`, `scripts/train.py`, `scripts/evaluate.py`.
Runs: Kaggle. Touch it: LEARNING.md step 2.

#### `revt2v/infer.py` — learning scaffold (2 functions, 11 TODOs)
The only entry point into model code that `scripts/evaluate.py` and
`scripts/serve.py` use: `load_student_for_inference` (rebuild a student
and load a trained checkpoint into it) and `generate` (the reverse
diffusion sampling loop, method-agnostic — whatever a method's `apply()`
did to the student at training time is still in effect here). Imports
from: `student.py`, `teacher.py`, `methods/`, `utils.py`. Imported by:
`scripts/evaluate.py`, `scripts/serve.py`. Runs: Kaggle (eval) and Colab
(serving). Touch it: LEARNING.md step 4.

`generate` also takes optional initial `latents`, so different methods can
start from the same noise. `MethodBank` holds every inference method on ONE
shared U-Net (fits a T4): `teacher`, `conv_oracle`, and, if their
checkpoints exist, `attn_rotation` and `conv_student`. It switches by
toggling LoRA adapters, attention processors, and conv flips, restoring the
clean teacher after each call. `MethodBank.matched_target(prompt, z)` =
`reverse(teacher(latents=flip(z)))`, the per-sample reference for a reverse
generator started from `z`. Checkpoints load from `<method>/latest.pt` on
the Hub, falling back to the original root `latest.pt` (never for
`conv_mirror`).

#### `revt2v/methods/__init__.py`
Documents the shared `apply()`/`loss()` interface every method implements
and exposes `METHODS`, the `{name: class}` lookup `scripts/train.py` and
`scripts/serve.py` use for `--method`. Fully implemented — it's just
wiring, not model logic. Imports from: the two method files below.
Imported by: `scripts/train.py`, `scripts/evaluate.py`, `scripts/serve.py`.

#### `revt2v/methods/attn_rotation.py` — learning scaffold (3 items, 9 TODOs)
Attention-rotation method: `RotatedTemporalAttnProcessor` (a diffusers
attention-processor class implementing the 180-degree attention rotation)
plus `AttentionRotation` (`apply` installs it on the U-Net's temporal
self-attention layers, `loss` is epsilon loss with an optional ablation
term). Imports from: `torch`. Runs: Kaggle. Touch it: LEARNING.md
step 5. Read the module docstring before coding — the correct operation
is subtle to get right (see LEARNING.md step 5).

#### `revt2v/methods/conv_mirror.py` — temporal-conv mirroring
Flips every temporal `Conv3d` kernel (inside `TemporalConvLayer`, 88 of
them, kernel `(3, 1, 1)`) along time: `find_temporal_convs`,
`flip_temporal_convs` (per block: down/mid/up; flip again to undo; never a
blend), `ConvMirror` (`apply`/`loss`/`lora_targets`). Two modes via
`use_trained_weights`:
- `false` → **oracle** (`conv_oracle` in `MethodBank`): teacher with all
  convs flipped, no LoRA, no training. Exactly `flip(teacher(flip(z)))`,
  i.e. equivalent to reversing the teacher's output — use it as a
  reference/upper bound only.
- `true` → **conv student**: LoRA on the 88 temporal Conv3d of the
  unflipped teacher (`attention_lora_targets` is an empty hook for adding
  temporal attention later), loss = ε-MSE + `mirror_loss_weight` ·
  MSE to `flip(ε_teacher(flip(x_t)))`. Optional `flip_blocks` flips some
  blocks before training; it's saved in the checkpoint, re-applied at
  inference, and any other flip on a trained student is refused (it would
  double-reverse time). Imports from: `torch`, `diffusers`, `data.py`.
  Runs: Kaggle (training), Colab (comparison).

### `scripts/` — command-line entry points (all fully implemented infra)

Each script only does argument parsing, config loading, logging, and
looping — the actual model math is entirely delegated to `revt2v/`.

#### `scripts/build_dataset.py`
Runs the teacher over every prompt in a prompts file and saves one `.pt`
record per prompt to disk (and optionally a private HF Hub *dataset* repo,
for resuming across Kaggle sessions). Skips prompts whose shard already
exists locally or on the Hub. Imports from: `revt2v.data`, `revt2v.teacher`,
`revt2v.utils`. Runs: Kaggle. Called from: `notebooks/kaggle_train.ipynb`.

#### `scripts/train.py`
The training loop: builds the student, resumes a checkpoint if one exists
(local, then the Hub), iterates the dataset, computes the method's loss,
steps the optimizer, and periodically checkpoints (local + Hub push). Noise
is added via a `diffusers.DDPMScheduler` built from the teacher's scheduler
config — not `pipeline.scheduler` directly, since that's whatever fast
sampler the pipeline ships for inference. Imports from: `revt2v.data`,
`revt2v.student`, `revt2v.teacher`, `revt2v.methods`, `revt2v.utils`,
`diffusers.DDPMScheduler`. Runs: Kaggle. Called from:
`notebooks/kaggle_train.ipynb`.

#### `scripts/evaluate.py`
Loads a trained checkpoint, generates on every held-out prompt, generates
the reversed-teacher reference for comparison, saves side-by-side videos,
and computes/aggregates the four metrics (whose *computation* is a
learning scaffold — see above). Imports from: `revt2v.data`,
`revt2v.infer`, `revt2v.teacher`, `revt2v.utils`. Runs: Kaggle (or Colab,
if you'd rather evaluate there).

#### `scripts/serve.py`
A minimal FastAPI app: `POST /generate`, `GET /jobs/{id}`, `GET /health`,
videos served as static files, generation running on a background worker
thread so requests don't block on GPU work. Imports from: `revt2v.infer`,
`revt2v.methods`, `revt2v.utils`. Runs: Colab. Called from:
`notebooks/colab_test.ipynb`.

### `configs/` — fully implemented YAML, one TODO field each

Two self-contained configs, one per method in use: `attn_rotation.yaml`
(also the `--config` default) and `conv_mirror.yaml`, loaded by
`scripts/train.py`/`evaluate.py` via `--config`. (The earlier `baseline`
and `motion_prior` methods and their configs were removed; they're in git
history.)
`attn_rotation.yaml`'s `target_modules` names the temporal attention layers
to give LoRA, scoped to temporal attention
only (e.g. a regex like `r".*temp_attentions.*\.(to_q|to_k|to_v|to_out\.0)$"`)
— `revt2v.student.build_student` warns at runtime if a match falls outside
`temp_attentions`. Read by: `revt2v.utils.load_config`.
`conv_mirror.yaml` leaves `target_modules: null` (filled in by
`ConvMirror.lora_targets`) and adds `use_trained_weights`, `flip_blocks`,
`mirror_loss_weight`, and `hf_subfolder: conv_mirror` (Hub folder for its
checkpoint; configs without it keep using the root `latest.pt`).

### `prompts/` — fully implemented data files

`train.txt` (154 prompts) and `test.txt` (30 prompts) — one prompt per
line, no overlap between the two files. Sourced from VBench's
`human_action` and `subject_consistency` prompt categories plus a handful
of added object/animal-motion prompts, covering people, vehicles, and
animals so training and evaluation see varied motion types. Read by:
`scripts/build_dataset.py` (`train.txt`) and `scripts/evaluate.py`
(`test.txt`).

### `notebooks/` — fully implemented, run remotely

- `kaggle_train.ipynb` — clone/pull, install, load `HF_TOKEN` from Kaggle
  Secrets, GPU check, run `build_dataset.py` then `train.py`. Runs: Kaggle.
- `colab_test.ipynb` — clone/pull, install, load `HF_TOKEN` from Colab
  userdata, start `serve.py` in the background, open a `cloudflared`
  tunnel, hit the API with `requests`. Runs: Colab.
- `compare_methods.ipynb` — clone/pull, install, `HF_TOKEN` from Colab
  secrets, load `MethodBank` once; exactness checks (U-Net level and
  `conv_oracle` vs `matched_target` video level); then for each prompt and
  seed, every method from the same noise, saved as mp4s + side-by-side,
  scored vs the matched target (flow direction cosine, motion ratio with
  < 0.3 = frozen, mean pixel diff), summary table, frame sheets, zip.
  Runs: Colab (GPU).

There's no dedicated scratch notebook — free exploration (load the teacher,
print temporal attention layer names, sanity-check the flip-latents-vs-
flip-frames equivalence) happens in an ad hoc script instead; see `p.py`
under "Root files" below.

### `tests/` — fully implemented, run locally (CPU-only, no download)

Three files exercising the pure-math and small-toy-module parts of the
scaffold, so you get fast feedback without needing a GPU or downloading
the teacher: `test_data.py` (latent flip + dataset item shapes),
`test_attn_rotation.py` (rotated attention vs. a from-scratch reference, on
a tiny real `diffusers.Attention` layer with random weights),
`test_conv_mirror.py` (Conv3d discovery by class, flip-twice = identity,
exact per-block flips, toy time-mirror, double-flip guard; plus one GPU
test, skipped without CUDA, checking the real U-Net's
`unet_flipped(z) == flip(unet(flip(z)))` and printing the max diff), `test_student.py`
(only LoRA parameters get `requires_grad=True`, on a tiny toy U-Net). Run with
`pytest` from the repo root — as of the scaffold's initial state, every one
of these fails with `NotImplementedError` until you fill in the
corresponding function; that's expected, not a bug. There's no dedicated
`test_losses.py` — loss math has no shared module to unit-test in
isolation; it's exercised indirectly wherever a method's `.loss()` is
called.

### `results/`

Where generated videos and `metrics.json` land (`scripts/evaluate.py`,
`scripts/serve.py`). Gitignored except for a `.gitkeep` — this is
regenerated output, not something to commit.

### Root files

- `pyproject.toml` — package metadata and dependencies (`pip install -e .`
  makes `revt2v` importable everywhere without path hacks). Fully implemented.
- `.gitignore` — excludes weights, videos, datasets, caches, OS junk.
- `README.md` — project overview, setup, and the workflow diagram.
- `LEARNING.md` — the implementation order (read this to decide what to
  work on next).
- `STRUCTURE.md` — this file.
- `p.py` — ad hoc, untracked scratch script (not part of the `revt2v`
  package) standing in for the originally-planned `notebooks/scratch.ipynb`:
  load the teacher and inspect it (e.g. print `pipeline.unet` to find
  temporal attention layer names for `target_modules`). Edit freely; not
  meant to be a permanent, polished file.

## Concepts glossary

- **Package / `__init__.py`** — a directory becomes an importable Python
  package once it has an `__init__.py`; `revt2v/` is one, so `from revt2v
  import teacher` works from anywhere after installing it.
- **Editable install** (`pip install -e .`) — installs the package as a
  link to this source directory instead of copying it, so edits to `.py`
  files take effect immediately without reinstalling.
- **Latent** — a compressed representation of an image/video frame that a
  VAE produces; diffusion models generate in this compressed space (32x32
  here, vs. 256x256 pixels) because it's far cheaper than working in
  pixel space directly.
- **VAE (Variational Autoencoder)** — the frozen network with `.encode`
  (pixels -> latents) and `.decode` (latents -> pixels) used here; applied
  independently per frame, which is *why* flipping latents equals encoding
  reversed frames.
- **U-Net** — the network that actually does the denoising step in
  diffusion; `UNet3DConditionModel` here adds temporal layers on top of
  the usual image U-Net so it can reason across frames, not just within one.
- **Temporal attention** — the specific layers inside the U-Net that let
  information flow between different frames (as opposed to spatial layers,
  which only look within one frame); the only layers this project's
  `attn_rotation` method touches.
- **LoRA (Low-Rank Adaptation)** — instead of fine-tuning a whole (huge,
  frozen) weight matrix, add a small trainable low-rank update next to it;
  drastically fewer trainable parameters, much less VRAM/compute.
- **CFG (Classifier-Free Guidance)** — running the model twice per
  denoising step (once conditioned on the prompt, once unconditional) and
  extrapolating between them to sharpen how strongly the output follows
  the prompt.
- **Scheduler / timesteps** — the object controlling how much noise is
  present at each step of the diffusion process, both when adding noise
  during training and when removing it during sampling.
- **fp16** — 16-bit floating point; halves memory and roughly doubles
  throughput vs. fp32 on GPUs that support it, at some numerical precision
  cost — standard for running (and often training) diffusion models on
  consumer/free-tier GPUs.
- **Checkpoint** — a saved snapshot of trainable weights (here: just the
  small LoRA weights, not the frozen base model) plus training state
  (step, optimizer), so a run can resume.
- **HF Hub (Hugging Face Hub)** — used two ways here: as the source of the
  pretrained teacher model, and as this project's own private storage for
  dataset shards and checkpoints, bridging Kaggle and Colab sessions that
  share no disk.
- **Cloudflare (quick) tunnel** — a free, no-signup way to expose a
  `localhost` port (here, Colab's `serve.py`) as a public HTTPS URL,
  what `colab_test.ipynb` uses for the live demo link.
- **FVD (Frechet Video Distance)** — a distributional-similarity metric
  between two sets of videos in a pretrained video-classifier feature
  space; lower means the generated videos "look like" the reference set.
- **CLIP score** — cosine similarity between a CLIP text embedding and a
  CLIP image embedding, used here both for text-video alignment and (frame
  vs. next frame) for temporal consistency.

## Data & artifacts

| What | Where | Format | Shape |
|---|---|---|---|
| Latent dataset shard | `data/latents/*.pt` (local), mirrored to an HF `dataset` repo | dict via `torch.save` | `latents`/`reversed_latents`: `(4, F, 32, 32)` fp16; `prompt_embeds`/`negative_prompt_embeds`: `(77, 1024)` fp16 |
| Training checkpoint | `results/checkpoints/<method>/latest.pt` (local), mirrored to an HF `model` repo (root `latest.pt`, or `<hf_subfolder>/latest.pt`, e.g. `conv_mirror/latest.pt`) | dict via `torch.save`: `{"step", "model", "optimizer", "metadata"}` | `model` is a LoRA-only state dict (megabytes, not gigabytes) |
| Evaluation output | `results/<method>/` | `NNN_compare.mp4` side-by-side videos + `metrics.json` | videos: `(F, H, 2W, C)` uint8 (student left, reversed teacher right) |
| Served video | `results/serve/<job_id>.mp4` | mp4, served at `/videos/<job_id>.mp4` | `(F, H, W, C)` uint8 |

Batched training tensors add a leading batch dimension, e.g. latents become
`(B, 4, F, 32, 32)` once a `DataLoader` collates them — see
`data.flip_latents_time_axis`'s docstring for exactly which axis is the
frame axis in each case.

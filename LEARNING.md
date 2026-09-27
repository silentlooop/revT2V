# LEARNING.md — implementation curriculum

Do these in order. Each step only depends on the ones before it. Every
scaffolded function has full TODOs in its docstring — this file tells you
*which* functions to tackle when, why, and how to know you got it right.
See [STRUCTURE.md](STRUCTURE.md) for what every file is if you need
orientation first.

## How the workflow works

You write code locally in VS Code. Nothing here runs a GPU locally — you
push to git, then pull and run on whichever free GPU fits the task:

```
VS Code (edit code)
     |  git push
     v
git remote (GitHub)
     |  git pull
     v
     +------------------------+------------------------+
     |                        |
Kaggle (training)        Colab (serving/demo)
scripts/build_dataset.py  scripts/serve.py
scripts/train.py          + cloudflared tunnel
```

Kaggle and Colab both wipe local disk between sessions (Kaggle also caps
sessions at 12 hours; you get ~30 GPU-hours/week total). Neither can mount
Google Drive from a Kaggle kernel, so there's no shared persistent disk
between "where I trained" and "where I'm serving." A **private Hugging
Face Hub repo** is the bridge: `revt2v.utils.CheckpointManager` pulls the
latest checkpoint on startup and pushes back periodically (plus once at
the end of a run), and `scripts/build_dataset.py` does the same for
dataset shards. Practically: every notebook cell that runs `train.py` or
`build_dataset.py` is safe to re-run after a session dies — it picks up
where it left off.

Each step below names which notebook (if any) to run it from.

## Step 1 — `teacher.py`

**Concepts first:** what a diffusion pipeline's `__call__` does under the
hood (sampling loop, classifier-free guidance, scheduler timesteps); what a
VAE's encode/decode does and why it doesn't need to know about time; fp16
and `torch.no_grad()`/`requires_grad_(False)` and why a frozen model still
needs both.

**Files/TODOs:** all of `revt2v/teacher.py` — `load_teacher`,
`encode_prompt`, `generate_forward_video`, `decode_latents_to_video`,
`encode_video_to_latents`.

**Where:** a scratch script (see `p.py` at the repo root — there's no
dedicated scratch notebook), on Kaggle or Colab (GPU required — this won't
run comfortably on CPU).

**Verify:** load the teacher and generate a clip end to end — you should
get a real video you can watch, a printed list of module names containing
"temp" (you'll need these in step 3), and a small number (not huge) from a
flip-latents-vs-flip-frames comparison — that's the empirical check that
`data.flip_latents_time_axis` (step 2) will be mathematically sound.

**Expected ballpark** (T4, fp16, 16 frames, 256x256, 50 steps) — these are
rough, your numbers will vary: teacher load ~1-2 min, one clip ~20-40s,
peak VRAM a few GB for inference alone.

## Step 2 — `data.py` + build the dataset

**Concepts first:** why flipping already-encoded latents on the frame axis
equals encoding reversed pixel frames (per-frame VAE, no temporal mixing —
you just verified this empirically in step 1); what `torch.utils.data.Dataset`
needs to implement.

**Files/TODOs:** `revt2v/data.py` — `flip_latents_time_axis`,
`build_dataset_record`, `LatentDataset`.

**Verify:** `pytest tests/test_data.py` (CPU-only, no download — this
should pass once `flip_latents_time_axis` and `LatentDataset` are
correct, independent of everything else). Then, on Kaggle:
`python scripts/build_dataset.py --prompts prompts/train.txt --hf-repo <you>/revt2v-dataset`
(or via `notebooks/kaggle_train.ipynb`, section 5) — start with
`--limit 5` to sanity-check a handful of records before committing GPU
hours to all 154 training prompts.

**Expected ballpark:** ~20-40s/prompt at 50 inference steps -> the full
154-prompt training set is on the order of an hour or two of GPU time;
budget accordingly against the weekly 30-hour cap. Each `.pt` record is
small (a few MB — mostly the fp16 latents and two 77x1024 embeddings).

## Step 3 — `student.py` + `baseline` -> first training run

**Concepts first:** what LoRA actually inserts into a linear layer and why
only those extra parameters need gradients; the standard diffusion
training objective (sample noise, corrupt at a random timestep, predict
the noise back out); why this project needs no teacher forward pass at
train time (the teacher's only job was building the dataset in step 2).

**Files/TODOs:** `revt2v/student.py` (all functions), `revt2v/methods/baseline.py`
(`apply`, `loss`). There's no dedicated `losses.py` — noise sampling and
MSE loss are written directly in `scripts/train.py` and `methods/*.py`.

**Verify, in order:**
1. `pytest tests/test_student.py` — CPU-only, no download, uses a tiny
   toy "U-Net" instead of the real one, checks that *only* LoRA parameters
   have `requires_grad=True` after `build_student`.
2. On Kaggle, set `target_modules` in `configs/baseline.yaml` to the
   layer names you found in step 1's scratch script, scoped to temporal
   attention only (e.g. a regex like
   `r".*temp_attentions.*\.(to_q|to_k|to_v|to_out\.0)$"` — a bare name like
   `"to_v"` would also match spatial/cross-attention; `build_student` warns
   at runtime if that happens), then:
   `python scripts/train.py --config configs/baseline.yaml --set num_steps=50 --hf-repo <you>/revt2v-checkpoints`
   — a short 50-step smoke run. Loss should be a finite number that
   generally trends down, not `nan` or flat at a huge value.

**Expected ballpark:** LoRA fine-tuning the frozen U-Net at rank 8, batch
size 1, fp16, on a T4 — order 1-2s/step; VRAM depends heavily on whether
you can enable gradient checkpointing on the U-Net (worth doing if you hit
OOM). A full `num_steps: 1000` run is on the order of 30-60 minutes.

## Step 4 — `infer.py` + `evaluate.py` metrics -> first numbers

**Concepts first:** the reverse diffusion sampling loop (start from noise,
iteratively denoise using the scheduler, classifier-free guidance
combining a conditional and unconditional prediction); what each eval
metric is actually measuring (FVD: distributional similarity in a video
feature space; CLIP score: does the video match the text; temporal
consistency: does it flicker; motion consistency: does the *motion*, not
just the content, match the reversed teacher).

**Files/TODOs:** `revt2v/infer.py` (`load_student_for_inference`,
`generate`), the four `compute_*` functions in `scripts/evaluate.py`.

**Verify:** `python scripts/evaluate.py --config configs/baseline.yaml --prompts prompts/test.txt --limit 5 --hf-repo <you>/revt2v-checkpoints`
against the checkpoint from step 3. You should get `results/baseline/metrics.json`
and a handful of `*_compare.mp4` side-by-side videos you can actually
watch — look at those before trusting the numbers, especially this early.

**Expected ballpark:** sampling one clip is roughly the same cost as one
teacher forward generation (same number of denoising steps) — tens of
seconds per prompt on a T4.

## Step 5 — `attn_rotation` (primary method)

**Concepts first:** re-read `revt2v/methods/attn_rotation.py`'s module
docstring closely — the "flip q and k after projecting from the unflipped
input, don't flip v or the output" distinction is the entire method, and
the easy-to-write wrong version (flip input, attend, flip output back) is
mathematically a no-op. Understand *why* before writing code — it'll save
you a confusing debugging session where training "works" but does nothing.

**Files/TODOs:** `revt2v/methods/attn_rotation.py` — `RotatedTemporalAttnProcessor.__call__`, `AttentionRotation.apply`, `AttentionRotation.loss`.

**Verify:** `pytest tests/test_attn_rotation.py` first (CPU-only, no
download, uses a tiny real `diffusers.Attention` layer) — this must pass
before you touch the real model, since it's checking exactly the
easy-to-get-backwards math above. Then on Kaggle:
`python scripts/train.py --config configs/attn_rotation.yaml --hf-repo <you>/revt2v-checkpoints`,
then evaluate the same way as step 4 with `--config configs/attn_rotation.yaml`.

**Compare against baseline:** run `scripts/evaluate.py` for both methods
on the same `prompts/test.txt` and look at the side-by-side videos and
`metrics.json` next to each other — does rotation produce more convincing
reverse motion than plain fine-tuning, even at a smaller LoRA rank?

## Step 6 — `motion_prior` (secondary method)

**Concepts first:** the closed-form relationship between a predicted
noise, a noisy latent, and an estimated clean latent (`x0`) at a given
diffusion timestep — you'll need this to compute `predicted_x0` before the
auxiliary loss in this method means anything.

**Files/TODOs:** `revt2v/methods/motion_prior.py` — `apply`, `loss`,
`motion_consistency_penalty`. Start with `prior_weight: 0.0` (the config
default) so this method trains identically to baseline while you get the
mechanics working, then wire up `predicted_x0` and raise `prior_weight`
above 0 as an ablation.

**Verify:** train and evaluate the same way as steps 3-5, comparing
`prior_weight: 0.0` vs. a small nonzero value (e.g. `0.1`) on the same
prompts.

## Step 7 — serve via Colab + Cloudflare

**Concepts first:** nothing new conceptually — this step exercises the
same `infer.generate` you wrote in step 4, just behind an HTTP API instead
of a script loop.

**Files/TODOs:** none — `scripts/serve.py` is fully implemented
infrastructure that calls your `infer.py`.

**Verify:** run `notebooks/colab_test.ipynb` end to end, pointing
`REVT2V_HF_REPO` at whichever checkpoint you want to demo. You should get
a public `https://*.trycloudflare.com` URL, a `{"status": "ok"}` from
`/health`, and a playable video back from a `/generate` + poll `/jobs/{id}`
round trip.

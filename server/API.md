# revT2V inference server — API contract

Implemented in `server/app.py`, matching **`client/src/api/CONTRACT.md`**
exactly (that file + `client/src/api/types.ts` are the source of truth for
every request/response shape — this file only covers what's server-specific
and not in the contract: auth, env vars, and operational notes).

Poll `/jobs/{id}` every **~0.7s** while a job is `queued`/`running` (the
frontend's `useJob` hook does this already).

## Auth

If `API_KEY` is set on the server (`server/colab_server.ipynb` always sets
one — user-supplied or a random `secrets.token_urlsafe(16)`), every endpoint
except `/health` requires a header `X-API-Key: <key>` (`401` if missing or
wrong). The frontend sends this from `VITE_API_KEY` — see `client/.env.example`.

## One job = one student method

`POST /generate` takes a single `student_method` (not a list). Every job
always runs and returns three videos from the **same initial noise**:
`teacher` (forward, normal), `student` (the requested method), and
`teacher_reversed` (just the teacher's own frames played backwards — not a
second generation pass: reversing a video's frame order also flips its flow
field's frame order and negates every vector, so `teacher_reversed`'s
`VideoOut.flow` is derived from the teacher's, not recomputed). `metrics`
compares `student` against `teacher_reversed`.

## Endpoints

- `GET /health` — never requires `X-API-Key`.
- `GET /methods` — ids: exactly `attn_injection`, `conv_oracle`, `conv_lora`
  (never `attn_lora`/`attn_rotation` — those are research-only, MethodBank
  only). `attn_injection` is always `available: true` (zero-shot until a
  LoRA for the injection mechanism is loaded); `conv_oracle` is always
  `available: true` (training-free); `conv_lora` depends on a checkpoint
  existing in `REVT2V_HF_REPO`.
- `POST /generate` — `400` for: empty/>500-char prompt; `steps` outside
  `[1, 100]`; `num_frames` not in `{8, 16, 24}`; unknown `student_method`;
  a known but `available: false` `student_method` (`detail` includes its
  reason).
- `GET /jobs/{job_id}` — `404` for an unknown id.
- `GET /jobs/{job_id}/result` — `404` unknown id, `409` if not yet `done`.
- `GET /files/{path}` — static mount serving every `video_url`/frame PNG
  referenced in a result. Videos are H.264 MP4 (yuv420p), 8 fps, 256×256.

## Metrics notes (beyond what CONTRACT.md's `PairMetrics` table says)

- `fvd` is always `NaN` for a single live pair — Fréchet distance needs a
  batch of samples to be meaningful (see `scripts/evaluate.py` for the
  set-level number across the whole test set); `notes.fvd` explains this.
  The frontend's own `fmtNum` already renders `NaN` as `—`.
- `clip.*` is real (CLIP ViT-L/14, loaded lazily once and shared across
  jobs).
- `dynamic_degree.*` is the real VBench score if VBench is installed
  (`pip install --no-deps vbench easydict`); otherwise it falls back to a
  flow-magnitude-threshold proxy with `notes.dynamic_degree` saying so
  explicitly — never silently presented as the real metric.
- `optical_flow.*` is real (RAFT if `torchvision`'s weights load, else
  OpenCV Farneback — same fallback `scripts/evaluate.py` uses).

## Env vars

`REVT2V_HF_REPO` (default `silentlooop/revt2v-ckpt`), `REVT2V_CHECKPOINT_DIR`,
`REVT2V_RESULTS_DIR`, `API_KEY` — see the top of `app.py` for defaults.

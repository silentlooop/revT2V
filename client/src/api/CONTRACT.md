# revT2V API contract

The frontend talks to the backend only through `src/api/client.ts`. Implement these endpoints in FastAPI and set `VITE_USE_MOCK=false` and `VITE_API_BASE_URL=<your url>`. The TypeScript source of truth for every shape is `src/api/types.ts`.

- All bodies are JSON (`Content-Type: application/json`).
- Errors: return a non-2xx status with `{"detail": "human readable message"}`. FastAPI's `HTTPException` does this by default.
- CORS: allow the frontend origin (e.g. `http://localhost:5173`), methods `GET, POST`, header `Content-Type`.
- Media URLs may be absolute (`https://…`) or relative to the API (`/media/job_x/teacher.mp4`). The client prefixes relative URLs with `VITE_API_BASE_URL`.
- Videos must be browser-playable **H.264 MP4** (`yuv420p`), served with CORS headers. The player seeks the `<video>` frame by frame, so `crossOrigin="anonymous"` must work.

---

## `GET /health`

Polled every 8 s. If this request fails, the UI switches to offline mode and shows the precomputed gallery.

```json
{
  "online": true,
  "gpu": "Tesla T4",
  "model_loaded": true,
  "queue_length": 0,
  "version": "0.1.0"
}
```

## `GET /methods`

The composer and the method lists are built from this. Nothing is hardcoded in the UI.

```json
[
  {
    "id": "conv_lora",
    "label": "conv_lora",
    "kind": "main",
    "description": "LoRA on temporal convs + attention",
    "trained": true,
    "available": true,
    "checkpoint": "silentlooop/revt2v-ckpt/conv_lora"
  }
]
```

| field | type | notes |
|---|---|---|
| `id` | string | sent back as `student_method` |
| `kind` | `"main" \| "trained" \| "training_free"` | controls the method tag |
| `available` | bool | `false` → greyed out and not selectable |
| `checkpoint` | string? | shown for information only |

Expected ids:

| id | kind | what it does |
|---|---|---|
| `attn_injection` | `trained` | capture the base U-Net's own attention on the time-flipped latent (stock attention, no LoRA), inject it rotated 180° against the real pass's own V/output — a to_v/to_out.0 LoRA trained for this mechanism is used if one exists, else zero-shot |
| `conv_oracle` | `training_free` | flip every temporal conv kernel in time (exact time mirror of the teacher) |
| `conv_lora` | `main` | LoRA on temporal convs + attention, ε-MSE + mirror loss |

## `POST /generate`

Request body (`GenerateRequest`):

```json
{
  "prompt": "white smoke rising from a burning incense stick, black background",
  "student_method": "conv_lora",
  "seed": 42,
  "steps": 25,
  "cfg": 9.0,
  "num_frames": 16,
  "negative_prompt": "watermark, text"
}
```

Response: `202` (or `200`) with

```json
{ "job_id": "job_8f3a2c" }
```

The backend must run the **teacher and the student from the same seed and initial noise** so the comparison is fair.

## `GET /jobs/{job_id}`

Polled every ~0.7 s while a job is pending, and again after a page refresh (the job id is stored in localStorage).

```json
{
  "id": "job_8f3a2c",
  "status": "running",
  "queue_pos": 0,
  "current_model": "teacher",
  "step": 12,
  "total_steps": 25,
  "elapsed_s": 41.3,
  "created_at": "2026-10-07T10:21:00Z",
  "request": { "...": "the GenerateRequest" },
  "error": null
}
```

| field | type | notes |
|---|---|---|
| `status` | `"queued" \| "running" \| "done" \| "error"` | |
| `queue_pos` | int | jobs ahead of this one; `0` once running |
| `current_model` | `"teacher" \| "student"` | which model is denoising right now |
| `step` / `total_steps` | int | diffusion step of the **current** model (resets to 0 when switching to the student) |
| `elapsed_s` | float | seconds since the job was created |
| `error` | string? | set when `status = "error"` |

Return `404` for unknown ids (for example after a backend restart). After 5 consecutive failures the UI shows the error and a retry button.

## `GET /jobs/{job_id}/result`

Only valid when `status = "done"` (otherwise `409`).

```json
{
  "job_id": "job_8f3a2c",
  "request": { "...": "the GenerateRequest" },
  "time_taken_s": 212.4,
  "teacher": { "...VideoOut" },
  "student": { "...VideoOut" },
  "teacher_reversed": { "...VideoOut (optional)" },
  "metrics": { "...PairMetrics" }
}
```

### `VideoOut`

```json
{
  "video_url": "/media/job_8f3a2c/teacher.mp4",
  "frames": ["/media/job_8f3a2c/teacher_00.png", "/media/job_8f3a2c/teacher_05.png", "/media/job_8f3a2c/teacher_10.png", "/media/job_8f3a2c/teacher_15.png"],
  "frame_indices": [0, 5, 10, 15],
  "fps": 8,
  "width": 256,
  "height": 256,
  "num_frames": 16,
  "flow": {
    "grid_w": 8,
    "grid_h": 8,
    "frames": [[[0.0, -3.1], [0.2, -2.9]]]
  }
}
```

- `flow.frames` has `num_frames - 1` entries, one per frame transition. Each entry is a flat list of `grid_w × grid_h` `[dx, dy]` vectors in pixels at 256×256, row-major. Average the dense Farneback/RAFT flow inside each grid cell. It drives the arrow overlay.
- `teacher_reversed` is optional. If present, the "teacher reversed" toggle appears. It is simply the teacher frames in reverse order (reversing the flow means flipping frame order and negating vectors).

### `PairMetrics`

```json
{
  "fvd": 246.0,
  "clip": { "teacher": 31.2, "student": 30.6 },
  "dynamic_degree": { "teacher": 0.71, "student": 0.64 },
  "optical_flow": {
    "teacher_mag": 1.75,
    "student_mag": 1.61,
    "direction_cosine": 0.81,
    "frozen": false
  },
  "notes": {
    "fvd": "Single-pair FVD is an approximation; see set-level FVD.",
    "clip": null,
    "dynamic_degree": null,
    "optical_flow": null
  }
}
```

| metric | definition | better |
|---|---|---|
| `fvd` | FVD between the student video and the **reversed** teacher video (I3D features). One pair, so approximate. | lower |
| `clip.*` | mean over frames of 100 × cos(CLIP image emb, CLIP text emb) | higher |
| `dynamic_degree.*` | VBench dynamic degree (RAFT-based), 0–1 | higher (≈ teacher) |
| `optical_flow.teacher_mag / student_mag` | mean flow magnitude (px/frame) | ≈ equal |
| `optical_flow.direction_cosine` | cosine between student flow and time-reversed teacher flow (flip frame order, negate vectors), over all frames and pixels | → +1 |
| `optical_flow.frozen` | `student_mag < 0.3 × teacher_mag` | `false` |

`notes.*` strings are optional. When present they are shown under the metric.

## `GET /results`

Set-level results for the Findings section. Shape: `Results` in `types.ts`. If you'd rather keep these numbers static, don't implement this endpoint and edit `src/content/results.ts` instead. The mock serves that file. To always use the static file, make `api.getResults` return `RESULTS` directly.

```json
{
  "test_set": { "name": "revT2V-test", "num_prompts": 20, "seeds_per_prompt": 3 },
  "columns": [
    { "key": "fvd", "label": "FVD", "better": "lower", "hint": "…" }
  ],
  "leaderboard": [
    { "method": "conv_lora", "fvd": 246, "clip": 30.6, "dynamic_degree": 0.64, "flow_cosine": 0.81 }
  ],
  "loss_curves": [
    { "method": "conv_lora", "label": "conv_lora student", "steps": [0, 50], "total": [0.42, 0.39], "components": [{ "name": "ε-MSE", "values": [0.3, 0.28] }] }
  ],
  "qualitative": [
    { "prompt": "…", "seed": 42, "videos": { "teacher": "/media/qual/smoke/teacher.mp4", "attn_injection": "…", "conv_oracle": "…", "conv_lora": "…" } }
  ],
  "findings": [
    { "id": "f1", "method": "conv_oracle", "verdict": "positive", "title": "…", "body": "…" }
  ]
}
```

`better` can be `"higher"`, `"lower"` or `"one"` (best when closest to 1.0). Use `null` for a metric value that doesn't exist yet. Set `todo: true` on any row to show a TODO sticker.

---

### Minimal FastAPI skeleton

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory="media"), name="media")

@app.get("/health")
def health(): ...

@app.get("/methods")
def methods(): ...

@app.post("/generate", status_code=202)
def generate(req: GenerateRequest): ...   # enqueue, return {"job_id": ...}

@app.get("/jobs/{job_id}")
def job(job_id: str): ...

@app.get("/jobs/{job_id}/result")
def result(job_id: str): ...
```

On Colab, expose the server with a tunnel (e.g. `cloudflared` or `ngrok`) and put the public URL in `VITE_API_BASE_URL`.

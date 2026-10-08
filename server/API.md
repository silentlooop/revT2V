# revT2V inference server — API contract

Implemented in `server/app.py`. Poll `/jobs/{id}` every **1–2 seconds**
while a job is `queued`/`running`.

All bodies are JSON. Errors are a non-2xx status with `{"detail": "..."}`.
CORS allows any origin. If `API_KEY` is set on the server, every endpoint
except `/health` requires a header `X-API-Key: <key>` (401 if missing/wrong).

## `GET /health`

Never requires `X-API-Key`.

```json
{ "status": "ok", "gpu": "Tesla T4", "uptime": 812.4, "queue_length": 0 }
```

`gpu` is `null` on a CPU-only machine.

## `GET /methods`

```json
[
  {
    "id": "attn_injection",
    "name": "Attention injection (zero-shot)",
    "description": "Captures the base U-Net's own attention on the time-flipped latent (stock attention, no LoRA) and re-applies it, rotated 180°, against the real trajectory's own V/output projections — a to_v/to_out.0 LoRA trained for this mechanism is used if one exists on the Hub, otherwise this runs zero-shot.",
    "available": true,
    "reason": null
  },
  {
    "id": "conv_oracle",
    "name": "Conv rotation (oracle)",
    "description": "Every temporal conv kernel flipped in time — an exact, training-free mirror of the teacher.",
    "available": true,
    "reason": null
  },
  {
    "id": "conv_lora",
    "name": "Conv LoRA",
    "description": "LoRA on temporal convs + attention, ε-MSE + mirror loss (main method).",
    "available": false,
    "reason": "no checkpoint for 'conv_lora' found at silentlooop/revt2v-ckpt"
  }
]
```

Ids: exactly `attn_injection`, `conv_oracle`, `conv_lora` — nothing else is
selectable. `conv_oracle` is always `available: true` (training-free).
`attn_injection` is also always `available: true`: it runs zero-shot (`name`
reads `"Attention injection (zero-shot)"`) until a LoRA trained for the
injection mechanism exists on the Hub, at which point it's loaded
automatically and `name` switches to `"Attention injection (LoRA)"`.
`conv_lora` depends on a checkpoint existing in the configured Hub repo.
`teacher` and `matched_target` are generated internally as comparison
references when `compare_with_target: true` (see below) but are never
themselves selectable.

## `POST /generate`

```json
{
  "prompt": "a red ball bouncing on a wooden floor",
  "methods": ["conv_lora", "conv_oracle"],
  "seed": 42,
  "steps": 25,
  "cfg": 9.0,
  "negative_prompt": "watermark, text",
  "compare_with_target": true
}
```

Only `prompt` and `methods` are required — `seed` (random if omitted),
`steps` (25), `cfg` (9.0), `negative_prompt` (`"watermark, text"`), and
`compare_with_target` (`false`) all default. Every requested method runs
from the **same initial noise**, derived deterministically from `seed`, so
outputs are directly comparable.

Response, `202`:

```json
{ "job_id": "job_8f3a2c1b90", "position": 0 }
```

`position` is jobs ahead of this one in the queue at submission time (racy
by nature — the worker may already have started the next job by the time
you read it).

Errors (`400`): prompt empty or over 500 characters; `steps` outside
`[10, 50]`; an unknown method id; a known but currently `available: false`
method id (`detail` includes its `reason`).

## `GET /jobs/{job_id}`

```json
{
  "status": "running",
  "progress": { "current_method": "conv_lora", "step": 14, "total_steps": 25 },
  "results": [
    {
      "method": "conv_oracle",
      "video_url": "/files/job_8f3a2c1b90/conv_oracle.mp4",
      "thumbnail_url": "/files/job_8f3a2c1b90/conv_oracle_thumb.jpg",
      "seconds": 38.2,
      "metrics": {
        "flow_cosine_vs_target": 0.94,
        "motion_ratio_vs_teacher": 1.01,
        "frozen": false,
        "mean_pixel_diff_vs_target": 0.021
      }
    }
  ],
  "error": null
}
```

`status`: `queued | running | done | error`. `results` fills in as each
requested method finishes — a `running` job can already have earlier
methods' results while later ones are still generating. `metrics` is `null`
unless the request had `compare_with_target: true` (and is always `null` for
the `matched_target` method itself, which is the reference the metrics are
computed against). `404` for an unknown `job_id`.

## `GET /files/{path}`

Serves the video/thumbnail files referenced in `results` (a static file
mount — `path` is whatever `video_url`/`thumbnail_url` gave you, e.g.
`job_8f3a2c1b90/conv_oracle.mp4`). Videos are H.264 MP4 (yuv420p), 8 fps,
256×256, 16 frames; thumbnails are the first frame as JPEG.

## Metrics, when `compare_with_target: true`

- `flow_cosine_vs_target` — direction agreement (optical flow) between the
  method's output and `matched_target`, the correct reverse-time reference.
  Closer to 1 is better.
- `motion_ratio_vs_teacher` — mean flow magnitude of the output divided by
  the teacher's. `frozen: true` when this is `< 0.3` (the method produced
  near-static output instead of reversed motion).
- `mean_pixel_diff_vs_target` — mean absolute pixel difference (0–1 scale)
  against `matched_target`.

These reuse `scripts/evaluate.py`'s `optical_flow`/`compute_flow_metrics`
(OpenCV Farneback, no extra model download).

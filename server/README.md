# revT2V inference server

Runs every method (teacher, training-free oracles, and the trained LoRA
students) on one GPU, behind a small FastAPI job API. See `API.md` for the
contract and `app.py` for the implementation — it reuses
`revt2v.infer.MethodBank` and `scripts/evaluate.py` directly, no duplicated
model code.

## Run it on Colab (the supported way)

1. Open `colab_server.ipynb` in Colab.
2. **Runtime > Change runtime type > GPU** (a T4 is enough).
3. Add a Colab secret named `HF_TOKEN` (left sidebar, key icon) with read
   access to `silentlooop/revt2v-ckpt`, and turn on notebook access for it.
4. **Runtime > Run all.**
5. Step 6 prints a `SERVER URL` (a `*.trycloudflare.com` link) and an
   `API KEY`. Paste both into the frontend's settings.
6. Leave the notebook running (step 8's cell) while you use the frontend —
   closing the tab or letting Colab disconnect kills the server and the
   tunnel. **The URL changes every time the notebook restarts**; re-run from
   step 5 (or step 2, if the whole runtime reset) and paste the new one.

`GET /methods` tells you which trained methods actually loaded — a missing
checkpoint shows up as `available: false` there, not as a crash.

## Run it locally (for development, no tunnel)

```bash
pip install -e .
uvicorn server.app:app --reload --port 8000
```

Needs a GPU reachable from wherever you run this (`revt2v.infer.load_teacher`
defaults to `cuda`). `REVT2V_HF_REPO`, `REVT2V_CHECKPOINT_DIR`,
`REVT2V_RESULTS_DIR`, and `API_KEY` are all read from the environment — see
the top of `app.py` for defaults.

## Tests (no GPU needed)

```bash
pytest tests/test_server.py -v
```

These exercise the job queue, validation, auth, and error recovery against a
fake method registry — the real GPU-backed registry (`RealRegistry` in
`app.py`) is never touched by these.

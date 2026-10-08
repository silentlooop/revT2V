# Repo structure

| folder | what's in it |
|---|---|
| `revt2v/` | The model code: teacher loading, student/LoRA building, data (latent dataset, time-flip), and `methods/` (attn_injection, conv_lora; `conv_mirror.py` holds the training-free conv_oracle flip mechanics, shared by conv_lora). Nothing here talks to a network or a notebook. |
| `scripts/` | CLI entry points: `train.py`, `evaluate.py` (FVD/CLIP/flow/VBench scoring), `build_dataset.py`, `conv_rank_check.py`, `serve.py`. Not a package — loaded by path. |
| `configs/` | One YAML per method/run (`attn_injection.yaml`, `conv_lora.yaml`, `attn_lora.yaml`). `conv_oracle` needs no training/config. |
| `server/` | The Colab-hosted inference backend (`app.py`), the Colab launcher notebook (`colab_server.ipynb`), and its docs (`API.md`, `README.md`). See `server/README.md` to run it. |
| `client/` | The Vite + React frontend. See `client/README.md`. |
| `tests/` | Pytest suite — all CPU-only, no GPU or model download required. |
| `notebooks/` | Ad hoc Jupyter notebooks (comparing methods, pushing datasets/model cards to the Hub). |
| `kaggle/` | Kaggle kernel launchers (training + dataset building on Kaggle's GPU). |
| `prompts/` | `train.txt` / `test.txt` prompt lists for dataset building and evaluation. |
| `results/` | Generated videos, checkpoints copied locally, eval output — gitignored except a `.gitkeep`. |

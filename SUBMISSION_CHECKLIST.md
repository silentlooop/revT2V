# Submission checklist

## What I (the agent) did

- Audited the repo: tests, secrets, file sizes, scaffolding leftovers. Tagged `submission-v1` locally.
- Wrote `README.md` (root — did not exist before).
- Wrote `scripts/make_report_assets.py` and ran it (`report_assets/` now has real dataset stats from the live manifest; everything else is skip-logged until you drop in logs/results).
- Wrote `notebooks/final_eval.ipynb` for the Colab run.
- Wrote `docs/REPORT_NOTES.md` and `docs/references.bib`.
- Wrote `scripts/package_submission.sh` and built `submission.zip` (4 MB, 121 files — rerun the script after you fill in the TODOs below so the zip picks up the real results).

## What you must do by hand before submitting

1. **Drop Kaggle training logs into `logs/`** — e.g. `logs/conv_lora.log`, `logs/attn_lora.log`, `logs/attn_injection.log` (plain text with `step=N loss=X step_time=Ys` lines, or Kaggle's JSON log export). Without these, `report_assets/loss_summary.md` and the loss-curve PNGs don't exist and §6/§7 of the report have no real loss numbers.
2. **Run `notebooks/final_eval.ipynb` on Colab** (GPU, `HF_TOKEN` secret with read access to `silentlooop/revt2v-ckpt` and `silentlooop/revt2v-data`). Start with `QUICK = True` to sanity-check, then set `QUICK = False` for the real report run. This produces `results/final/` (exactness report, comparison videos, metrics CSVs, the rho diagnostic, `conv_rank_check` output) and `results/final.zip`.
3. **Re-run `python scripts/make_report_assets.py`** after step 2 — it will now also produce `report_assets/metrics_table.{md,csv,tex}` and the qualitative frame sheets from `results/final/`.
4. **Fill every `TODO(user)`** in `README.md` (environment versions, HF repo URLs, demo URL) and `docs/REPORT_NOTES.md` (loss numbers, metrics table, conclusion, the literature entries with no verified arXiv id, dataset §5's 149-vs-147-prompts discrepancy if you care to chase it down).
5. **Push to GitHub** (`git push origin main`) — confirm `https://github.com/silentlooop/revT2V` is current and public/accessible to whoever grades it.
6. **Add the real links** in `README.md`'s Links section (HF model/dataset repo URLs, and a live server URL if you're demoing — Colab tunnel URLs are ephemeral, so only fill this in right before presenting).
7. **Write the actual report text** from `docs/REPORT_NOTES.md` into whatever document template IE 643 requires (Introduction, Workflow, Literature Survey, Proposed Approach 4.1/4.2, Dataset Details, Experiments, Results, Novelty Assessment plan, Conclusion, References) — `REPORT_NOTES.md` is bullet points mapped to those headings, not final prose.
8. **Rebuild `submission.zip`**: `bash scripts/package_submission.sh`, after steps 1-3 above so it includes the real `results/final/` and `logs/`.

## Pre-submit checklist

- [ ] Secrets scan clean: `grep -rE "hf_[A-Za-z0-9]{20,}" . --include="*" | grep -v .git` — repo working tree and `main` history are clean; a real token was found only in a local-only Claude Code session-checkpoint ref (never pushed to GitHub) — **revoke that HF token anyway**, since it was written to disk at some point.
- [ ] `pytest -q` passes except the one known, pre-existing failure (`test_scripts.py::test_prompt_files` — stale fixture vs. the current `prompts/test.txt`, 5 extra prompts; decide whether to update the test's hardcoded list or trim the prompt file, then fix it — left untouched here since it's inside the "report-asset scripts and docs only" scope, not a model-behavior change, but it's a loose end).
- [ ] README links filled in (not `TODO(user)`).
- [ ] `submission.zip` size is reasonable (no accidental large binary) — check with `unzip -l submission.zip | tail -1`.
- [ ] `https://github.com/silentlooop/revT2V` opens and shows the pushed commit.
- [ ] Every `TODO(user)` you can resolve before the deadline is resolved; the rest are visible markers, not silently dropped.

## Things flagged, not changed (ML logic — out of scope for this pass)

- `conv_lora`'s first full training run showed a flat loss curve (per your own description) — not fixed, not silently claimed to work. Documented honestly in `README.md` and `docs/REPORT_NOTES.md`'s claims table.
- `STRUCTURE.md`'s `kaggle/` line was stale (said it built datasets; it only trains now) — fixed, since that's a doc correction, not a logic change.
- No other `NotImplementedError`/`TODO(`/stale-scaffold findings in the code.

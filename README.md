# H&M Personalized Fashion Recommendations

**Difficulty:** 9/10 &middot; **Task type:** Hybrid recommender

This repo is one of 22 Liora MLE portfolio projects. Phase 1 scaffold (EDA + verified references + baseline + advanced model scripts + IMRaD manuscript + presentation) is complete. Model execution happens in the main learning session (not committed here).

## Repo layout

```
brief.md          Project summary and methodology plan
data/README.md    Dataset source and download command (data files gitignored)
notebooks/        EDA notebook (raw, awaiting execution)
reports/          Verified academic references
src/              Baseline + advanced model scripts (runnable)
manuscripts/      4-5k word IMRaD write-up
deliverables/     Self-contained HTML presentation
checkpoint.json   Phase-1 status manifest
```

## Data

Raw data is **not committed** (see `.gitignore`). See `data/README.md` for the dataset source and exact download command.

## How to reproduce

```bash
git clone https://github.com/Sandyyy123/liora-13-hm-fashion.git
cd liora-13-hm-fashion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # if present
# Follow data/README.md to download the dataset, then:
jupyter lab notebooks/
python src/model_baseline.py
python src/model_advanced.py
```

## Status

Phase 1 scaffold complete (code-only) as of 2026-05-08. Model execution and Phase 2+ work (RAG / fine-tuning / agent layer) are tracked in the main portfolio.

## License

MIT - see [LICENSE](LICENSE).

---

*Part of a 22-project Liora MLE portfolio. See companion repos: liora-01 through liora-22.*

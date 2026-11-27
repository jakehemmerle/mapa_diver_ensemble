# MAPA + DIVER-1 ensemble for Neuroprobe (within-session)

Code, frozen design, results and technical note for the **MAPA+DIVER-1 ensemble** entry on the
[Neuroprobe](https://neuroprobe.dev) within-session leaderboard.

| Within-session, official test split | Mean AUROC |
|---|---:|
| **MAPA+DIVER-1 ensemble (this repo)** | **0.7037** |
| MAPA (our reproduction) | 0.6955 |
| MAPA (leaderboard) | 0.6953 |
| DIVER-1 0.1 s tiny, frozen (leaderboard) | 0.6777 |

Gain over MAPA: +0.0082, 6/6 subjects improved, exact sign-flip p = 0.016, session bootstrap 95% CI
[+0.0036, +0.0132]. The design was fixed on permitted non-evaluation sessions, vetoed once on half of the
official validation split (+0.0083, Go), then scored once on the official test split.

**Method.** Two released encoders, both frozen: MAPA (last layer, closed-form ridge, one penalty per cell
chosen on validation AUROC) and DIVER-1 0.1 s tiny (linear head, DIVER's own trainer, early stopping on
validation AUROC). Each member's scores are mapped through the empirical CDF of its own label-free
validation scores, then averaged with equal weights. No pretraining was done here.

The full protocol, amendment, disclosures and per-session/per-task results are in
[`paper/note.pdf`](paper/note.pdf) (source `paper/note.tex`).

## Layout

| Path | Contents |
|---|---|
| `common/` | splits (official Neuroprobe generators, tiers P/V/T), ridge readout, statistics, prediction files, test-label lock |
| `ensemble/` | members (MAPA kernels, DIVER-1 head, CNN), fusion (`fuse.py`), phases E0–E5 (`run.py`), controls |
| `classical/` | the classical-feature track (did not qualify; kept for the record) |
| `server/` | freeze (`overnight_freeze.py`), confirmation (`overnight_confirm.py`), single-class amendment, setup/sync |
| `frozen-designs/within.json` | the frozen within-session design (config `cc49ff955fa0ae88`) |
| `submission/` | the leaderboard folder submitted to `azaho/neuroprobe` |
| `01-…08-*.md` | plans, rule verdicts and audits written before and during the runs |

## Reproducing

Requirements: Linux, Python 3.11, [uv](https://docs.astral.sh/uv/), one CUDA GPU with ≥ 8 GB for the
DIVER-1 head, and the [Brain Treebank](https://braintreebank.dev) recordings.

1. Environment: `uv sync --locked` (`pyproject.toml`, `uv.lock`; recorded from the environment used for
   every run). Upstreams at pinned revisions (`ensemble/paths.py`): MAPA `dc582b2`, Neuroprobe `f9b0842`
   (identical package code to current `main`), DIVER-1 `25638eb`; DIVER-1 weights `diver_ieeg.pt`
   (sha256 `b8120937…864c`). `server/setup.sh` shows the layout under `$NPEXP_ROOT`
   (`data_dev/`, `src/`, `work/`, `results/`); it assumes an existing reproduction directory for data
   and checkpoints, so adapt its data steps to your copy of the dataset.
2. Design (Tier P, permitted sessions only): `python -m ensemble.run e0|e1|e2|e3 ...` and
   `python -m classical.run c0..c3 ...`; freeze with `python -m server.overnight_freeze`.
3. Confirmation: `python -m server.overnight_confirm within` runs the Tier V veto and, only on Go, the
   single Tier T score. Tier T test labels are locked by `common/lock.py` and every access is logged.

`PYTHONPATH` must point at the repository root. Server-specific details (GPU UUIDs in
`server/overnight_confirm.py`, unit names, paths) reflect the machine the runs used; change them for yours.

## Notes

- Within-session only. Neuroprobe PR #21 proposes temporal-block within-session splits; these results use
  the current splits.
- This is a snapshot of an internal repository (Tier T at internal commit `50932eb`); history is not
  included.
- MAPA (Apache-2.0) and DIVER-1 (MIT) are used through their released code and weights; cite their papers.

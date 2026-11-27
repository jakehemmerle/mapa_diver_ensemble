# MAPA+DIVER-1: Fusing Frozen Spectral and Waveform Encoders for Neuroprobe Within-Session Decoding

Code, frozen design, audit trail, and technical note ([`paper/note.pdf`](paper/note.pdf)) for the `MAPA+DIVER-1 ensemble` entry on the [Neuroprobe](https://neuroprobe.dev) within-session leaderboard ([insight-neuro/neuroprobe#22](https://github.com/insight-neuro/neuroprobe/pull/22)).

| Model (official within-session test split, 12 sessions / 360 cells) | Mean AUROC |
|---|---:|
| **MAPA+DIVER-1 ensemble (this repository)** | **0.70374** |
| MAPA (our reproduction) | 0.69554 |
| DIVER-1 0.1 s tiny frozen (our member) | 0.67907 |
| Ridge6 (classical reference, excluded) | 0.67602 |
| MAPA (leaderboard) | 0.6953 |
| DIVER-1 0.1 s tiny frozen (leaderboard) | 0.6777 |

The ensemble beats our MAPA reproduction by **+0.00820 AUROC** with **6/6 subjects** and **10/12 sessions** improved (exact one-sided sign-flip *p* = 0.0156, session bootstrap 95% CI [+0.0036, +0.0132], minimum leave-one-subject-out Δ = +0.0064).

## Key Findings

Current iEEG foundation models rely on contrasting signal views: MAPA encodes multi-scale STFT magnitudes (discarding waveform phase), whereas DIVER-1 encodes raw 500 Hz time-domain voltage patches. Using a three-tier holdout protocol with no pretraining and no learned fusion weights, we establish four empirical findings (full details in [`paper/note.pdf`](paper/note.pdf)):

1. **Hand-crafted phase and waveform features add zero signal to MAPA (Δ = +0.000039).**
   Stacking classical low-frequency waveform, theta/alpha phase, high-gamma timing, and shaft-covariance blocks onto MAPA produced virtually zero change on the 13-session Tier P design split (and fusing MAPA with Ridge6 scored −0.00009). Recovering what MAPA's magnitude-only spectrogram discards requires a *learned* raw-waveform encoder, not classical signal-processing features.

2. **A much weaker frozen raw-waveform encoder still lifts every subject (+0.0082, 6/6 subjects).**
   Although standalone frozen DIVER-1 trails MAPA by 0.0165 AUROC (0.6791 vs. 0.6955), a parameter-free 50/50 rank fusion improves all 6/6 evaluation subjects. On Tier P, it also beats a 5-bootstrap MAPA self-ensemble (−0.0062 vs. MAPA) by +0.0116, ruling out simple variance reduction.

3. **Complementarity splits cleanly along task physics (spectral power vs. waveform/phase).**
   - **Gains concentrate in language, word-timing, and auditory-onset tasks:** `word_index` (+0.0345), `delta_volume` (+0.0216), `word_length` (+0.0206), `word_part_speech` (+0.0180), `word_gap` (+0.0169), `gpt2_surprisal` (+0.0139), `onset` (+0.0131), `word_head_pos` (+0.0103), `speech` (+0.0057), and `pitch` (+0.0008).
   - **All four visual tasks and steady-state `volume` dip slightly:** `global_flow` (−0.0022), `frame_brightness` (−0.0040), `face_num` (−0.0064), `local_flow` (−0.0090), and `volume` (−0.0106).
   - With *n* = 6 subjects, no individual task survives Holm correction across 15 tests (smallest *p*_Holm = 0.234), so this split is directional evidence pointing toward dual-view (spectral + waveform/phase) iEEG foundation models rather than 15 isolated per-task claims.

4. **MAPA-unseen session gap and rank-calibration lesson.**
   - MAPA was pretrained on 12 of the 13 Tier P sessions (all except `5:0`). On MAPA-held-out session `5:0`, `MAPA+DIVER-1` gained **+0.0180** (0.7306 vs. 0.7126) compared with +0.0037 across the 12 MAPA-pretrained Tier P sessions — anticipating why the ensemble gained more on MAPA-unseen Tier T (+0.0082) than on the Tier P mean (+0.0055).
   - Calibrating held-out scores via out-of-fold training scores (`train_oof_cdf`) failed (−0.0075) because OOF score distributions were 2–3× wider than held-out scores, underweighting MAPA. Switching to the label-free validation-score empirical CDF (`select_cdf`) fixed the calibration, and a learned out-of-fold logistic stacker (+0.0023 over uniform fusion) did not clear the pre-specified +0.003 adoption bar.

## Method & Protocol

Both encoders reuse their released checkpoints and upstream session-level preprocessing unchanged, with no pretraining or fine-tuning:

- **MAPA (ViT-Small + dual ridge):** block-12 representations (`mapa12`) with closed-form, trace-normalized dual kernel ridge regression; one scalar penalty λ per cell chosen by validation AUROC.
- **DIVER-1 (0.1 s tiny + linear head):** frozen 500 Hz voltage-patch encoder with a linear head trained using DIVER's `Trainer` and early stopping on validation AUROC.
- **Label-free rank fusion:** each member's scores are mapped through the empirical CDF of its own unlabeled validation scores (`select_cdf` in [`ensemble/fuse.py`](ensemble/fuse.py)) and averaged with fixed 50/50 weights.

The design was selected on 13 permitted non-evaluation sessions (**Tier P**), vetted once on the later half of official validation (**Tier V**, Δ = +0.00834, 5/6 subjects up, Go), and scored once on official test (**Tier T**) behind [`common/lock.py`](common/lock.py). See [`paper/note.pdf`](paper/note.pdf) for the full protocol, Tier P disclosures, Tier V single-class amendment, and per-session/per-task tables.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Technical note ([`note.pdf`](paper/note.pdf), [`note.tex`](paper/note.tex), `refs.bib`, `PUBLICATION.bib`) |
| `docs/` | Pre-run research notes, rules verdicts, plans, audits ([`docs/README.md`](docs/README.md)), and staging records (`docs/staging/`) |
| `common/` | Official Neuroprobe splits, tiers P/V/T, dual ridge readout, statistics, prediction I/O, and test-label lock |
| `ensemble/` | Frozen members, label-free CDF fusion (`fuse.py`), phases E0–E5 (`run.py`), controls, and tests |
| `classical/` | Classical-feature negative-result track (`run.py`, `features.py`, controls, and tests) |
| `analysis/` | Decision report and leaderboard aggregation helpers |
| `server/` | Freeze (`overnight_freeze.py`), confirmation (`overnight_confirm.py`), single-class amendment, setup, and sync |
| `frozen-designs/` | Immutable frozen within-session config (`within.json`, hash `cc49ff955fa0ae88`) |
| `results/` | Immutable audit trail: hash-chained `ledger.csv` (381 rows), `test_access.log`, confirmation/amendment receipts, E1–E5 summaries, parity receipt, and `submission/export_t_within.py` |
| `submission/` | Leaderboard submission folder for `insight-neuro/neuroprobe#22` |

## Reproducing

You need Linux, Python 3.11, [uv](https://docs.astral.sh/uv/), one CUDA GPU with at least 8 GB of memory for the DIVER-1 head, and the [Brain Treebank](https://braintreebank.dev) recordings.

1. Install the locked environment with `uv sync --locked` (`pyproject.toml` and `uv.lock` pin the exact environment used for every run).
2. Fetch the upstream repositories at the revisions pinned in [`ensemble/paths.py`](ensemble/paths.py): MAPA `dc582b2`, Neuroprobe `f9b0842` (same package code as `main`), and DIVER-1 `25638eb` (with weights `diver_ieeg.pt` matching the SHA-256 in `ensemble/paths.py`).
3. Set up `$NPEXP_ROOT` (`data_dev/`, `src/`, `work/`, `results/`). [`server/setup.sh`](server/setup.sh) links the evaluation recordings from an existing copy of the dataset, and [`server/download_dev.py`](server/download_dev.py) downloads the 13 permitted Tier P recordings.
4. Run the Tier P design stage with `python -m ensemble.run` and `python -m classical.run`, then freeze with `python -m server.overnight_freeze`.
5. Run `python -m server.overnight_confirm within` to execute the Tier V veto and — only on Go — the single Tier T test score behind `common/lock.py`.

Set `PYTHONPATH` to the repository root. GPU IDs, systemd unit names, and paths in `server/overnight_confirm.py` reflect our machine and should be adapted for yours.

## Notes

- Results cover the within-session regime only (cross-subject returned Stop on Tier P; no cross-session or cross-subject claim is made).
- Open maintainer PR `azaho/neuroprobe#21` (temporal-block within-session splits) postdates this run; results follow `train_test_splits.py` at `f9b0842`.
- This repository is a public snapshot of an internal repository; the Tier T score came from internal commit `50932eb`.
- We use MAPA (Apache-2.0) and DIVER-1 (MIT) through their released code and checkpoints; please cite their papers.

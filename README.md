# MAPA+DIVER-1 ensemble for Neuroprobe within-session decoding

This repository holds the code, frozen design, results and technical note for the MAPA+DIVER-1 ensemble
entry on the [Neuroprobe](https://neuroprobe.dev) within-session leaderboard. The leaderboard pull request
is [insight-neuro/neuroprobe#22](https://github.com/insight-neuro/neuroprobe/pull/22).

| Model, official test split | Mean AUROC |
|---|---:|
| MAPA+DIVER-1 ensemble, this repository | **0.7037** |
| MAPA, our reproduction | 0.6955 |
| MAPA, leaderboard entry | 0.6953 |
| DIVER-1 0.1 s tiny frozen, leaderboard entry | 0.6777 |

The ensemble beats our MAPA reproduction by 0.0082 AUROC. All 6 subjects improved. The exact one-sided
sign-flip test over subjects gives p = 0.016, and the session bootstrap 95% interval is +0.0036 to +0.0132.

We fixed the design on the permitted non-evaluation sessions. We then ran one veto check on half of the
official validation split, which passed with a gain of 0.0083. Finally, we scored the frozen design once on
the official test split.

## Method

The ensemble has two members. Both encoders are the released checkpoints, and both stay frozen.

- MAPA uses its last encoder layer and a closed-form ridge readout. One ridge penalty per cell is chosen
  by validation AUROC.
- DIVER-1 0.1 s tiny uses a linear head trained with DIVER's own trainer. Early stopping uses validation
  AUROC.

The fusion maps each member's scores through the empirical CDF of that member's unlabeled validation scores.
It then averages the two members with equal weights. We did no pretraining.

The technical note, [`paper/note.pdf`](paper/note.pdf), gives the full protocol, the amendment, the
disclosures and the results per session and per task. Its source is `paper/note.tex`.

## Layout

| Path | Contents |
|---|---|
| `common/` | Splits from the official Neuroprobe generators, tiers P, V and T, the ridge readout, statistics, prediction files and the test-label lock |
| `ensemble/` | Members, fusion in `fuse.py`, ensemble phases E0 to E5 in `run.py`, and controls |
| `classical/` | The classical-feature track, which did not qualify and is kept for the record |
| `server/` | Freeze in `overnight_freeze.py`, confirmation in `overnight_confirm.py`, the single-class amendment, setup and sync |
| `frozen-designs/within.json` | The frozen within-session design, config `cc49ff955fa0ae88` |
| `submission/` | The leaderboard folder submitted to Neuroprobe |
| `01-*.md` to `08-*.md` | Plans, rule verdicts and audits written before and during the runs |

## Reproducing

You need Linux, Python 3.11, [uv](https://docs.astral.sh/uv/), one CUDA GPU with at least 8 GB of memory
for the DIVER-1 head, and the [Brain Treebank](https://braintreebank.dev) recordings.

1. Install the environment with `uv sync --locked`. The `pyproject.toml` and `uv.lock` files record the
   environment used for every run.
2. Fetch the upstream code at the revisions pinned in `ensemble/paths.py`. MAPA is at `dc582b2`, Neuroprobe
   at `f9b0842` and DIVER-1 at `25638eb`. Neuroprobe `f9b0842` has the same package code as current `main`.
   The DIVER-1 weights file is `diver_ieeg.pt`, with the sha256 listed in `ensemble/paths.py`.
3. Set up the data directory. `server/setup.sh` shows the layout under `$NPEXP_ROOT`, with the folders
   `data_dev/`, `src/`, `work/` and `results/`. The script links the evaluation recordings from an existing
   reproduction directory, and `server/download_dev.py` downloads the 13 permitted recordings. Point the
   linking step at your copy of the dataset.
4. Run the design stage on the permitted sessions with `python -m ensemble.run` and
   `python -m classical.run`. Freeze the design with `python -m server.overnight_freeze`.
5. Run `python -m server.overnight_confirm within`. It runs the validation veto and, only if the veto
   passes, the single test score. `common/lock.py` locks the test labels and logs every access.

Set `PYTHONPATH` to the repository root. The GPU IDs in `server/overnight_confirm.py`, the systemd unit
names and the paths come from the machine we ran on. Change them for yours.

## Notes

- The results cover the within-session regime only.
- Neuroprobe PR #21 proposes temporal-block within-session splits. These results use the current splits.
- This repository is a snapshot of an internal repository. The test score came from internal commit
  `50932eb`, and the history is not included.
- We use MAPA, licensed Apache-2.0, and DIVER-1, licensed MIT, through their released code and weights.
  Please cite their papers.

"""Server smoke test: one evaluation session, one task, a few hundred train and val rows.

    python -m classical.smoke [--session btbank3_0] [--task speech] [--n-train 300] [--n-select 150]
                              [--max-shafts 2] [--workers 1]

Uses a Tier V within-session cell and touches training and validation rows only: the
official test rows are not part of a Tier V cell and no "eval" label is requested.
Features go to work/smoke/, the prediction file to results/smoke/preds/ (so the real
prediction tree never holds a partial cell), and one "info" row goes to the real ledger.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time

import numpy as np

import common
from classical import features
from classical.run import base_arm
from common import ledger, predictions, ridge, splits
from common.splits import Cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="btbank3_0")
    parser.add_argument("--task", default="speech")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-select", type=int, default=150)
    parser.add_argument("--max-shafts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    opts = parser.parse_args()

    cell = Cell("V", "within", opts.session, opts.session, opts.task, 0)
    started = time.time()
    full = splits.get_cell(cell)
    print(f"cell {cell.key()} tier V: train {len(full.rows['train'])}, select {len(full.rows['select'])}, "
          f"eval {len(full.rows['eval'])} rows (eval untouched); built in {time.time() - started:.1f} s")
    counts = {"train": opts.n_train, "select": opts.n_select}
    first = {part: np.argsort(full.rows[part], kind="stable")[:count] for part, count in counts.items()}
    take = {part: full.rows[part][index] for part, index in first.items()}
    labels = {part: full.labels(part)[index] for part, index in first.items()}
    onsets = {part: full.onsets(part)[index] for part, index in first.items()}
    subset = splits.CellSplit(cell=cell, rows=take, rates=full.rates, _onsets=onsets,
                              _labels={part: (lambda part=part: labels[part]) for part in take})

    feature_root = common.work_root() / "smoke" / "features"
    shutil.rmtree(feature_root, ignore_errors=True)   # the smoke store is keyed to this run's rows
    started = time.time()
    features.extract(opts.session, {("window", "bipolar"): ("base6",)}, onsets=np.concatenate(list(onsets.values())),
                     root=feature_root, workers=opts.workers, max_shafts=opts.max_shafts)
    extract_s = time.time() - started
    store = features.Store(opts.session, "window", "bipolar", feature_root)
    n_channels, n_windows = len(store.names), len(store.onsets)
    recording = features.recording(opts.session)
    tiles = recording.n_samples / recording.rate / features.DEFAULTS["norm_tile_stride_s"]
    print(f"features: {n_windows} windows x {n_channels} bipolar channels ({opts.max_shafts} shafts) + {tiles:.0f} "
          f"normalization tiles in {extract_s:.1f} s wall with {opts.workers} worker(s) -> "
          f"{extract_s * opts.workers / n_channels / ((n_windows + tiles) / 1000):.2f} core-s per channel per 1000 windows")

    started = time.time()
    matrices = features.cell_matrices(cell, "base6", "window", "bipolar", root=feature_root, onsets=onsets)
    fit = ridge.fit_predict([features.kernel_set(matrices)], labels["train"], labels["select"],
                            oof_blocks=subset.oof_blocks())
    arm = base_arm()
    results = common.results_root() / "smoke"
    finite = np.isfinite(fit["scores"]["train"])
    path = predictions.write(
        cell, arm.member, {"train": take["train"][finite], "select": take["select"]},
        {"train": fit["scores"]["train"][finite], "select": fit["scores"]["select"]},
        {"commit": ledger.commit(), "config_hash": arm.hash(), "lam": fit["lam"], "smoke": True,
         "max_shafts": opts.max_shafts, "n_train": int(len(take["train"]))}, results)
    back = predictions.read(cell, arm.member, results)
    assert set(back.scores) == {"train", "select"} and "eval" not in back.rows
    assert np.array_equal(back.rows["select"], take["select"])
    recomputed = ridge.auroc(back.scores["select"], labels["select"])
    oof = ridge.auroc(back.scores["train"], labels["train"][finite])
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    children_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2
    print(f"fit: dim {matrices['train'].shape[1]}, lam {fit['lam']:.3g} ({fit['lam_pin'] or 'interior'}), "
          f"select AUROC {fit['select_auroc']:.4f}, out-of-fold train AUROC {oof:.4f}, {time.time() - started:.1f} s")
    print(f"prediction file {path}: parts {sorted(back.scores)}, rows {[len(v) for v in back.rows.values()]}, "
          f"select AUROC recomputed from file {recomputed:.4f}, meta {json.dumps(back.meta, sort_keys=True)[:200]}")
    print(f"peak RSS: parent {peak_gib:.2f} GiB, largest worker {children_gib:.2f} GiB")
    ledger.append({"tier": "V", "regime": "within", "phase": "smoke", "kind": "info", "arm": arm.member,
                   "config_hash": arm.hash(), "score": fit["select_auroc"],
                   "extra": {"smoke": True, "part": "select", "cell": cell.key(), "n_train": int(len(take["train"])),
                             "n_select": int(len(take["select"])), "max_shafts": opts.max_shafts}}, common.results_root())
    return 0


if __name__ == "__main__":
    sys.exit(main())

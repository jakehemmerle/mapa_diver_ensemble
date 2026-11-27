"""Shared export loop for the trained members (DIVER head, CNN).

Out-of-fold and seed design (the cheapest that still feeds fusion rung (b)):

- seed 0 trains one model per cell on all training rows plus five models on the purged
  out-of-fold blocks; its file ``<member>`` carries train (out-of-fold), select and eval;
- seeds 1 and 2 train the full model only; their files ``<member>@s1``, ``<member>@s2``
  carry select and eval and exist to report seed spread.

That is 8 fits per cell instead of 18 for three seeds of everything. The stacker is fit
on seed 0's out-of-fold scores and applied to seed 0's select/eval scores.

On the select part these members choose the stopping epoch by AUROC and nothing else.
Eval labels are never read here; the trainer's "test" loader is given the select rows.
The within V/T one-class-select amendment uses a fixed training prefix instead.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from common import confirm_fallback, ledger, predictions, stats

SEEDS = (0, 1, 2)


def seed_member(member: str, seed: int) -> str:
    return member if seed == SEEDS[0] else f"{member}@s{seed}"


def groups(cells):
    ordered = sorted(cells, key=lambda c: (c.train_session, c.target_session))
    return [(pair, list(group)) for pair, group in
            itertools.groupby(ordered, key=lambda c: (c.train_session, c.target_session))]


def _has_train(cell, member: str, root: Path) -> bool:
    return "train" in predictions.read(cell, member, root).scores


def pending(backend, cells, root: Path, seeds=SEEDS, available_only: bool = False, oof: bool = True):
    """Cells still lacking a seed's file (or, with ``oof``, seed 0's out-of-fold training part);
    with ``available_only`` those whose recordings exist."""
    from ensemble import paths
    first = seed_member(backend.name, seeds[0])
    todo = [c for c in cells if any(not predictions.exists(c, seed_member(backend.name, s), root) for s in seeds)
            or (oof and not _has_train(c, first, root))]
    if available_only:
        have = {s: paths.recording(s).is_file() for c in todo for s in (c.train_session, c.target_session)}
        todo = [c for c in todo if have[c.train_session] and have[c.target_session]]
    return todo


def export(backend, tier: str, regime: str, root: Path, *, splits=None, seeds=SEEDS, shard=(0, 1),
           available_only: bool = False, oof: bool = True, reuse_full: bool = False) -> int:
    """Train and export ``backend`` for every pending cell of a tier and regime.
    ``shard=(i, n)`` takes every n-th session pair, for one process per GPU. With
    ``oof=False`` seed 0's file carries select and eval only (no out-of-fold training
    scores, so it cannot enter fusion); a later run with ``oof=True`` redoes those cells.
    ``reuse_full`` keeps an existing seed-0 file's select/eval scores and meta (the quick
    export's full model) and only adds the five out-of-fold fits."""
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    if regime not in backend.regimes:
        raise SystemExit(f"{backend.name} has no {regime} protocol (supported: {backend.regimes})")
    todo = pending(backend, splits.list_cells(tier, regime), root, seeds, available_only, oof)
    written = 0
    for index, ((train, target), cells) in enumerate(groups(todo)):
        if index % shard[1] != shard[0]:
            continue
        pair = backend.open_pair(train, target, regime)
        for cell in cells:
            started = time.monotonic()
            split = splits.get_cell(cell)
            y_train, y_select = np.asarray(split.labels("train")), np.asarray(split.labels("select"))
            fallback = (confirm_fallback.for_select(cell, backend.name, y_select)
                        if backend.name == "diver" and 0 in seeds else None)
            x_train = pair.train(split.onsets("train"))
            x_select, x_eval = pair.target(split.onsets("select")), pair.target(split.onsets("eval"))
            for seed in seeds:
                receipt = fallback if seed == 0 else None
                fit_options = {} if receipt is None else {"fixed_epochs": receipt["epochs"]}
                reused = None
                if seed == seeds[0] and oof and reuse_full and predictions.exists(cell, seed_member(backend.name, seed), root):
                    reused = predictions.read(cell, seed_member(backend.name, seed), root)
                    if receipt is not None and reused.meta.get("confirm_fallback") != receipt:
                        reused = None
                if reused is None:
                    model = backend.fit(x_train, y_train, x_select, y_select, seed=1000 * seed, **fit_options)
                    scores = {"select": model.predict(x_select), "eval": model.predict(x_eval)}
                else:
                    scores = {part: reused.scores[part] for part in ("select", "eval")}
                if seed == seeds[0] and oof:
                    held_scores = np.full(len(y_train), np.nan)
                    for block, (fit_pos, held_pos) in enumerate(split.oof_blocks(), start=1):
                        inner = backend.fit(x_train[fit_pos], y_train[fit_pos], x_select, y_select,
                                            seed=1000 * seed + block, **fit_options)
                        held_scores[held_pos] = inner.predict(x_train[held_pos])
                    scores["train"] = held_scores
                scores = {part: np.asarray(value, dtype=np.float32) for part, value in scores.items()}
                oof_note = "5 purged time blocks, seed 0 only" if "train" in scores else None
                if reused is not None:
                    meta = dict(reused.meta, oof=oof_note, oof_commit=ledger.commit(), oof_autocast=backend.autocast)
                else:
                    meta = {"commit": ledger.commit(), "seed": seed,
                            "early_stop": backend.early_stop if receipt is None else None,
                            "best_epoch": model.best_epoch,
                            "selected_on_select": ["stopping epoch (AUROC)"] if receipt is None else [],
                            "select_auroc": stats.auroc(scores["select"], y_select) if receipt is None else None,
                            "trainer_select_auroc": model.select_auroc, "pair": pair.meta,
                            "oof": oof_note, "autocast": backend.autocast,
                            "config_hash": ledger.config_hash({"member": backend.name, **backend.config})}
                    if receipt is not None:
                        meta.update(confirm_fallback=receipt, epochs_run=receipt["epochs"])
                predictions.write(cell, seed_member(backend.name, seed), split.rows, scores, meta, root)
            written += 1
            print(f"[{backend.name}] {cell.tier}/{cell.regime} {cell.key()} "
                  f"{time.monotonic() - started:.0f} s", flush=True)
        backend.close_pair(train, target)
    return written


def plan(backend, tier: str, regime: str, root: Path, *, splits=None, seeds=SEEDS,
         available_only: bool = False, oof: bool = True, reuse_full: bool = False) -> dict:
    """Cells that would run, missing inputs and a time estimate, for ``--dry-run``."""
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    if regime not in backend.regimes:
        return {"cells": 0, "pairs": [], "missing": [], "hours": 0.0, "fits": 0,
                "note": f"{backend.name} has no {regime} protocol"}
    todo = pending(backend, splits.list_cells(tier, regime), root, seeds, available_only, oof)
    fits_per_cell = len(seeds) - (1 if reuse_full else 0) + (splits.N_OOF_BLOCKS if oof else 0)
    seconds, sessions = 0.0, set()
    for (train, target), cells in groups(todo):
        sessions |= {train, target}
        for cell in cells:
            n_train = len(splits.get_cell(cell).rows["train"])
            seconds += fits_per_cell * backend.seconds_per_train_row[regime] * n_train
    seconds += sum(backend.session_minutes(s) for s in sessions) * 60
    return {"cells": len(todo), "pairs": [pair for pair, _ in groups(todo)], "fits": fits_per_cell * len(todo),
            "missing": sorted({str(p) for s in sessions for p in backend.missing_inputs(s)}),
            "hours": seconds / 3600, "sessions": sorted(sessions)}

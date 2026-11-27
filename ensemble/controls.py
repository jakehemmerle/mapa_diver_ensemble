"""Controls that run with the ensemble candidate (plan phase E3).

- self-ensemble: M1 (``mapa12``) refit on five bootstrap resamples of its training
  trials and averaged; the gain over M1 is what variance reduction alone buys;
- leave-one-member-out: a member contributing under +0.001 is dropped;
- shuffled stacker: the stacker fit on permuted training labels must match the uniform mean;
- subject concentration: the gain with each subject removed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import ledger, predictions, stats
from ensemble import fuse

SELF_ENSEMBLE_MARGIN = 0.003
LOMO_MINIMUM = 0.001
SHUFFLE_TOLERANCE = 0.002
N_BOOT = 5


def _splits(splits):
    if splits is None:
        from common import splits as module
        return module
    return splits


def boot_members(base: str = "mapa12", n_boot: int = N_BOOT) -> list[str]:
    return [f"{base}@boot{b}" for b in range(n_boot)]


def export_bootstraps(tier: str, regime: str, root: Path, *, base: str = "mapa12", n_boot: int = N_BOOT,
                      seed: int = 0, splits=None, skip_existing: bool = True) -> int:
    """Refit ``base`` on bootstrap resamples of each cell's training rows.

    A resample draws training rows with replacement and takes the matching rows and
    columns of the stored kernel (features keep the standardization of the full
    training set). Each replicate chooses its own ridge penalty on select. Its
    out-of-fold scores, kept for the rows it drew, are its rank-normalization reference.
    """
    from common import mapa_features, ridge
    from common.ridge import KernelSet
    from ensemble.members import mapa
    splits = _splits(splits)
    written = 0
    for cell in splits.list_cells(tier, regime):
        names = boot_members(base, n_boot)
        if skip_existing and all(predictions.exists(cell, name, root) for name in names):
            continue
        split = splits.get_cell(cell)
        kernel = mapa_features.kernels(cell, mapa.TAP[base], splits=splits)
        y_train, y_select = np.asarray(split.labels("train")), np.asarray(split.labels("select"))
        blocks = split.oof_blocks()
        for b, name in enumerate(names):
            rng = np.random.default_rng(fuse._cell_seed(cell, seed * 1000 + b))
            draw = np.sort(rng.integers(0, len(y_train), len(y_train)))
            resampled = KernelSet(train=kernel.train[np.ix_(draw, draw)],
                                  cross={part: block[:, draw] for part, block in kernel.cross.items()})
            inner = [(np.flatnonzero(np.isin(draw, fit)), np.flatnonzero(np.isin(draw, held))) for fit, held in blocks]
            fit = ridge.fit_predict([resampled], y_train[draw], y_select, oof_blocks=inner)
            unique, first = np.unique(draw, return_index=True)
            rows = {"train": np.asarray(split.rows["train"])[unique], "select": split.rows["select"],
                    "eval": split.rows["eval"]}
            scores = {"train": fit["scores"]["train"][first], "select": fit["scores"]["select"],
                      "eval": fit["scores"]["eval"]}
            scores = {part: np.asarray(value, dtype=np.float32) for part, value in scores.items()}
            meta = {"commit": ledger.commit(), "control": "self-ensemble bootstrap", "base": base,
                    "replicate": b, "seed": seed, "lam": fit["lam"],
                    "select_auroc": ridge.auroc(scores["select"], y_select),
                    "config_hash": ledger.config_hash({"control": "bootstrap", "base": base, "n": n_boot, "seed": seed})}
            predictions.write(cell, name, rows, scores, meta, root)
        written += 1
    return written


def self_ensemble(tier: str, regime: str, root: Path, *, base: str = "mapa12", members=None,
                  splits=None, sessions=None) -> stats.Comparison:
    """Uniform fusion of the bootstrap replicates compared with the single base member."""
    members = list(members or boot_members(base))
    name = fuse.fuse(members, "a", tier, regime, root, splits=splits, sessions=sessions,
                     name=f"ctl_self__{base}x{len(members)}")
    return stats.compare(name, base, tier, regime, root, splits=splits, sessions=sessions, descriptive=True)


def leave_one_member_out(members, rung: str, tier: str, regime: str, root: Path, *, splits=None) -> dict:
    """Contribution of each member: full ensemble minus the ensemble without it."""
    members = sorted(members)
    full = fuse.fuse(members, rung, tier, regime, root, splits=splits)
    out = {}
    for member in members:
        rest = [m for m in members if m != member]
        reduced = rest[0] if len(rest) == 1 else fuse.fuse(rest, rung, tier, regime, root, splits=splits)
        delta = stats.compare(full, reduced, tier, regime, root, splits=splits, descriptive=True).delta
        out[member] = {"contribution": delta, "without": reduced, "keep": bool(delta >= LOMO_MINIMUM)}
    return out


def shuffled_stacker(members, tier: str, regime: str, root: Path, *, splits=None, seed: int = 0) -> dict:
    """Stacker fit on shuffled training labels against the uniform mean of the same members."""
    members = sorted(members)
    uniform = fuse.fuse(members, "a", tier, regime, root, splits=splits)
    shuffled = fuse.fuse(members, "b", tier, regime, root, splits=splits, shuffle_seed=seed,
                         name=fuse.fused_name(members, "b", prefix="ctl_shuf"))
    comparison = stats.compare(shuffled, uniform, tier, regime, root, splits=splits, descriptive=True)
    return {"comparison": comparison, "delta": comparison.delta,
            "passed": bool(abs(comparison.delta) <= SHUFFLE_TOLERANCE), "tolerance": SHUFFLE_TOLERANCE}


def subject_concentration(comparison: stats.Comparison) -> dict:
    """Gain with each subject removed. ``carried_by`` names a subject whose removal takes
    away more than half of a positive gain."""
    loso = comparison.leave_one_subject_out
    carried = [s for s, d in loso.items() if comparison.delta > 0 and d < 0.5 * comparison.delta]
    return {"delta": comparison.delta, "leave_one_subject_out": loso, "carried_by": carried}

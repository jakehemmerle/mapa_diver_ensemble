"""Closed-form multi-kernel ridge readout shared by both tracks.

Each kernel is trace-normalized (``K * n / trace(K)``, i.e. divided by its trace up to
the constant ``n``), the kernels are summed with fixed weights (equal unless given),
and one penalty is chosen from MAPA's 25-point grid by AUROC on the "select" part.
With a single kernel this is MAPA's readout (``evals/neuroprobe/readout.py`` at
dc582b2: ``LAM_MULTS``, penalty ``m * trace(G) / n``, no intercept, smallest penalty
on ties). Nothing else is selected: no kernel weights, no per-kernel penalties.
Frozen confirmation may instead supply a predeclared multiplier for a one-class select part.

Kernels are linear kernels of features standardized with training-row statistics,
so they are centred on the training rows and an intercept would not change AUROC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from sklearn.metrics import roc_auc_score

LAM_MULTS = tuple(np.logspace(-4.0, 4.0, 25))


class Degenerate(ValueError):
    """The cell cannot be fit, or its penalty cannot be selected."""


@dataclass
class KernelSet:
    train: np.ndarray               # (n_train, n_train), linear kernel of train-standardized features
    cross: dict[str, np.ndarray]    # {"select": (n_select, n_train), "eval": (n_eval, n_train)}
    # Optional Gram of the target rows among themselves, same standardization:
    # {"select": (ns, ns), "eval": (ne, ne), "eval_select": (ne, ns)}. Only the random-projection
    # control (classical/controls.py) needs it.
    cross_gram: dict[str, np.ndarray] | None = None


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC as MAPA's readout computes it; NaN when the labels have one class."""
    y = (np.asarray(labels) > 0).astype(int)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def combine(kernels: list[KernelSet], weights=None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Weighted sum of trace-normalized kernels (float64)."""
    if not kernels:
        raise ValueError("no kernels")
    weights = np.ones(len(kernels)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(kernels),) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be non-negative, one per kernel, not all zero")
    n = kernels[0].train.shape[0]
    total = np.zeros((n, n), dtype=np.float64)
    cross: dict[str, np.ndarray] = {}
    for kernel, weight in zip(kernels, weights):
        if kernel.train.shape != (n, n):
            raise ValueError("kernels disagree on the number of training rows")
        trace = float(np.trace(kernel.train))
        if not np.isfinite(trace) or trace <= 0:
            raise Degenerate("kernel with non-positive or non-finite trace")
        scale = weight * n / trace
        total += scale * kernel.train
        for part, block in kernel.cross.items():
            cross[part] = cross.get(part, 0.0) + scale * np.asarray(block, dtype=np.float64)
    if any(set(kernel.cross) != set(cross) for kernel in kernels):
        raise ValueError("kernels disagree on their cross parts")
    return total, cross


def fit_predict(kernels: list[KernelSet], y_train, y_select, oof_blocks=None, weights=None, *,
                fixed_lam: float | None = None, strict: bool = False) -> dict:
    """Fit on the training rows, choose the penalty on "select", score every part.

    ``oof_blocks`` is ``CellSplit.oof_blocks()``: ``(fit_pos, held_pos)`` positions into the
    training rows. When given, ``scores["train"]`` holds out-of-fold scores refit per block
    at the chosen multiplier (the penalty itself is rescaled by the fit block's trace).
    The kernels were standardized with all training rows, a label-free dependence on the
    held block that the kernel interface cannot remove.

    ``fixed_lam`` skips the grid search and must be a grid multiplier for a nonempty,
    one-class select part. No AUROC or grid-selection diagnostics are produced for that
    fit. ``strict`` rejects empty or non-finite inputs and incomplete or degenerate
    out-of-fold fits; a fixed multiplier always enables those checks.
    """
    gram, cross = combine(kernels, weights)
    y = np.asarray(y_train, dtype=np.float64)
    n = gram.shape[0]
    if y.shape != (n,) or n < 2 or y.min() == y.max():
        raise Degenerate("training labels are missing, mismatched or single-class")
    if "select" not in cross:
        raise ValueError("kernels carry no 'select' part")
    strict = strict or fixed_lam is not None
    if strict:
        if not np.isfinite(y).all() or not ((y > 0).any() and (y <= 0).any()):
            raise Degenerate("training labels must be finite and contain both classes")
        if not np.isfinite(gram).all():
            raise Degenerate("non-finite training kernel")
        for part, block in cross.items():
            if block.ndim != 2 or block.shape[1] != n or not block.size or not np.isfinite(block).all():
                raise Degenerate(f"{part} cross-kernel is empty, mismatched or non-finite")
        select = np.asarray(y_select)
        if not select.size or select.shape != (cross["select"].shape[0],) or not np.isfinite(select).all():
            raise Degenerate("select labels are empty, mismatched or non-finite")
        if fixed_lam is not None:
            if fixed_lam not in LAM_MULTS:
                raise ValueError("fixed ridge multiplier must belong to the declared grid")
            if (select > 0).any() and (select <= 0).any():
                raise ValueError("fixed ridge multiplier requires one-class select labels")
    base = float(np.trace(gram)) / n
    if strict and (not np.isfinite(base) or base <= 0):
        raise Degenerate("training kernel has non-positive or non-finite trace scale")
    eigenvalues, vectors = np.linalg.eigh(gram)
    projected = vectors.T @ y
    if strict and (not np.isfinite(eigenvalues).all() or not np.isfinite(projected).all()):
        raise Degenerate("non-finite training eigensystem")
    if fixed_lam is None:
        multipliers = np.asarray(LAM_MULTS)
        alphas = vectors @ (projected[:, None] / (eigenvalues[:, None] + base * multipliers[None, :]))
        select_scores = cross["select"] @ alphas
        grid = np.array([auroc(select_scores[:, i], y_select) for i in range(len(multipliers))])
        if not np.any(np.isfinite(grid)):
            raise Degenerate("select part cannot rank penalties (one class or empty)")
        best = np.nanmax(grid)
        tied = np.flatnonzero(grid == best)
        index = int(tied[0])    # the grid ascends, so this is the smallest tied penalty
        lam, alpha = float(multipliers[index]), alphas[:, index]
        select_auroc, n_tied = float(best), int(tied.size)
        lam_pin = "lo" if index == 0 else ("hi" if index == len(multipliers) - 1 else "")
    else:
        lam = float(fixed_lam)
        alpha = vectors @ (projected / (eigenvalues + base * lam))
        select_auroc = index = n_tied = None
        lam_pin = ""
    scores = {part: block @ alpha for part, block in cross.items()}
    if oof_blocks is not None:
        scores["train"] = _out_of_fold(gram, y, oof_blocks, lam, strict=strict)
    if strict:
        bound = np.finfo(np.float32).max   # predictions are stored in float32
        for part, values in scores.items():
            if not np.isfinite(values).all() or values.min() < -bound or values.max() > bound:
                raise Degenerate(f"{part} scores are non-finite at export precision")
    return {
        "scores": scores,
        "lam": lam,
        "select_auroc": select_auroc,
        "lam_index": index,
        "lam_pin": lam_pin,
        "n_tied": n_tied,
    }


def _out_of_fold(gram: np.ndarray, y: np.ndarray, blocks, multiplier: float, *, strict: bool = False) -> np.ndarray:
    """Refit on each block's fit rows and score its held rows.

    The full Gram is centred on all training rows, which makes the full fit's missing
    intercept harmless. A fit block's rows are not centred, so each refit centres the
    kernel and the labels on its own fit rows (ridge with an unpenalized intercept on
    features centred there) and adds the fit rows' label mean back. Without this the
    constant part of ``y`` is fit through the features and every held block gets its own
    offset: before November 25 the pooled out-of-fold scores were 2-4x wider than held-out
    scores and their AUROC up to 0.1 lower (mapa12 more than ridge6)."""
    scores = np.full(len(y), np.nan)
    for fit, held in blocks:
        fit, held = np.asarray(fit, dtype=np.int64), np.asarray(held, dtype=np.int64)
        if len(fit) < 2 or len(held) == 0:
            if strict:
                raise Degenerate("out-of-fold block has too few fit rows or no held rows")
            continue
        target = y[fit]
        if strict and not ((target > 0).any() and (target <= 0).any()):
            raise Degenerate("out-of-fold fit labels must contain both classes")
        sub = gram[np.ix_(fit, fit)]
        means = sub.mean(axis=0)
        grand = float(means.mean())
        sub = sub - means[None, :] - means[:, None] + grand
        cross = gram[np.ix_(held, fit)]
        cross = cross - cross.mean(axis=1, keepdims=True) - means[None, :] + grand
        offset = float(target.mean())
        penalty = multiplier * float(np.trace(sub)) / len(fit)
        if strict and (not np.isfinite(penalty) or penalty <= 0):
            raise Degenerate("out-of-fold fit kernel has non-positive or non-finite trace scale")
        system = sub + penalty * np.eye(len(fit))
        try:
            alpha = scipy.linalg.solve(system, target - offset, assume_a="pos", check_finite=strict)
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as error:
            if strict:
                raise Degenerate("out-of-fold ridge system could not be solved") from error
            alpha = np.linalg.lstsq(system, target - offset, rcond=None)[0]
        scores[held] = cross @ alpha + offset
    return scores

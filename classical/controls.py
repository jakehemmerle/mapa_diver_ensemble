"""Controls that run beside every classical arm (plan, "Classical track", Controls).

* Shifted sham: a block's features are taken from a window at least 30 s later in the
  same session (circularly), so they keep their distribution and autocorrelation but lose
  their alignment with the labels.
* Phase-randomized waveform: B1/B2 computed from windows whose Fourier phases were
  randomized (same amplitude spectrum). Built at extraction time as ``B1~phaserand`` and
  ``B2~phaserand`` with the in-window pipeline.
* Extra-kernel control: a kernel from a random projection of MAPA frontend features with
  the dimension of the block it stands in for.

Window discipline, the fourth control, is not a function: it is the same arm run with
``window_mode="session"`` on Tier P.
"""
from __future__ import annotations

import numpy as np

from classical import features
from common.ridge import KernelSet
from common.splits import Cell

MIN_SHIFT_S = 30.0
PHASE_BLOCKS = ("B1", "B2")


def shifted_positions(onsets: np.ndarray, positions: np.ndarray, rate: int,
                      min_shift_s: float = MIN_SHIFT_S) -> np.ndarray:
    """For each cached window, the first cached window starting at least ``min_shift_s`` later.

    ``onsets`` are the sorted window starts of the session cache. Time wraps around from the
    last cached window to the first, so every row keeps a partner.
    """
    low, span = int(onsets[0]), int(onsets[-1]) - int(onsets[0]) + 1
    shift = int(round(min_shift_s * rate))
    if span <= 2 * shift:
        raise ValueError(f"cached windows span {span / rate:.0f} s, too short for a {min_shift_s:.0f} s circular shift")
    target = low + (onsets[positions] - low + shift) % span
    partner = np.searchsorted(onsets, target, side="left")
    partner[partner == len(onsets)] = 0
    return partner


def shifted_sham(cell: Cell, block: str, window_mode: str, reference: str, **options) -> tuple[KernelSet, int]:
    """Kernels of ``block`` with every part's features circularly shifted by at least 30 s."""
    def remap(store: features.Store, positions: np.ndarray) -> np.ndarray:
        return shifted_positions(store.onsets, positions, store.meta["rate"])
    return features.kernels_and_dim(cell, block, window_mode, reference, remap=remap, **options)


def phase_randomized(cell: Cell, block: str, reference: str, **options) -> tuple[KernelSet, int]:
    """Kernels of the phase-randomized surrogate of B1 or B2 (in-window pipeline)."""
    if block not in PHASE_BLOCKS:
        raise ValueError(f"the phase surrogate exists for {PHASE_BLOCKS}, not {block}")
    return features.kernels_and_dim(cell, f"{block}~phaserand", "window", reference, **options)


class RandomProjector:
    """Dimension-matched random-projection kernels drawn from a source kernel.

    For features Z (rows = train, select, eval) and a Gaussian matrix P with ``dim`` columns,
    Z P has rows that are jointly Gaussian with covariance (Z Z^T) x I given Z. Sampling
    L G with L L^T = Z Z^T is therefore the same distribution as projecting Z itself, and
    needs only the joint Gram matrix. That requires ``source.cross_gram`` (the target rows'
    Gram among themselves). Without it the target rows are projected onto the span of the
    training rows first (``exact`` is False): an approximation that drops what the target
    rows carry outside that span.
    """

    def __init__(self, source: KernelSet) -> None:
        self.parts = [part for part in ("select", "eval") if part in source.cross]
        self.sizes = [source.train.shape[0]] + [source.cross[part].shape[0] for part in self.parts]
        self.exact = source.cross_gram is not None and all(part in source.cross_gram for part in self.parts)
        if self.exact:
            n = sum(self.sizes)
            joint = np.empty((n, n), dtype=np.float64)
            edges = np.cumsum([0] + self.sizes)
            joint[:edges[1], :edges[1]] = source.train
            for i, part in enumerate(self.parts, start=1):
                block = slice(edges[i], edges[i + 1])
                joint[block, :edges[1]] = source.cross[part]
                joint[:edges[1], block] = source.cross[part].T
                joint[block, block] = source.cross_gram[part]
            if self.parts == ["select", "eval"]:
                select, evaluate = slice(edges[1], edges[2]), slice(edges[2], edges[3])
                joint[evaluate, select] = source.cross_gram["eval_select"]
                joint[select, evaluate] = source.cross_gram["eval_select"].T
            self.factor = _factor(joint)
            self.lift = None
        else:
            self.factor = _factor(np.asarray(source.train, dtype=np.float64))
            pseudo_inverse = np.linalg.pinv(source.train, hermitian=True)
            self.lift = {part: source.cross[part] @ pseudo_inverse for part in self.parts}

    def kernel(self, dim: int, seed: int) -> KernelSet:
        rng = np.random.default_rng(seed)
        projected = self.factor @ rng.standard_normal((self.factor.shape[1], dim)) / np.sqrt(dim)
        train = projected[: self.sizes[0]]
        if self.exact:
            edges = np.cumsum([0] + self.sizes)
            others = {part: projected[edges[i]:edges[i + 1]] for i, part in enumerate(self.parts, start=1)}
        else:
            others = {part: self.lift[part] @ train for part in self.parts}
        return KernelSet(train=train @ train.T, cross={part: values @ train.T for part, values in others.items()})


def _factor(gram: np.ndarray) -> np.ndarray:
    """L with L L^T = gram (positive part), columns restricted to the numerical rank."""
    values, vectors = np.linalg.eigh((gram + gram.T) / 2.0)
    keep = values > 1e-10 * values.max()
    return vectors[:, keep] * np.sqrt(values[keep])

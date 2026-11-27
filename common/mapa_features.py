"""MAPA kernels per cell: tap 0 is MAPA's frontend, tap 12 its last encoder layer.

The kernels are produced by ``ensemble.mapa.kernel_job`` (a storage-bounded GPU job)
and stored label-free under ``<work>/kernels/<store>/<regime>/tap<tap>/<cell key>.npz``.
Store ``P`` serves Tier P; store ``E`` serves Tiers V and T, whose cells share training
rows and draw select/eval rows from the same official validation and test rows.

On the server ``<work>/kernels`` is a symlink to the scratch NVMe
(``/scratch/neuroprobe-experiments/kernels``, ext4). That disk is not mounted again after
a reboot, so every access first checks that the store exists on an ext4 mount other than
``/`` and fails otherwise; nothing here ever falls back to another location. Files are
written and read whole, never memory-mapped.

Features are MAPA's released ones, unchanged: per contact for within- and cross-session
(contacts matched by label), atlas-region means for cross-subject (regions matched by
id). Each feature is standardized with the mean and standard deviation of the cell's
training rows only; the kernel is the linear kernel of those standardized features.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TAPS = (0, 12)
STORE_OF_TIER = {"P": "P", "V": "E", "T": "E"}


class KernelMissing(FileNotFoundError):
    """The GPU kernel job has not produced the kernel this cell needs."""


class KernelStoreUnavailable(RuntimeError):
    """The kernel store is absent or not on its required filesystem (scratch not mounted?)."""


def work_root() -> Path:
    return Path(os.environ.get("NPEXP_ROOT", Path.home() / "runs/neuroprobe-experiments")) / "work"


def _mount(path: Path) -> tuple[str | None, str | None]:
    """Mount point and filesystem type of the mount that holds ``path`` (Linux), else (None, None)."""
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return None, None
    best, kind = "", None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = fields[1].replace("\\040", " ")
        inside = str(path) == mount or str(path).startswith(mount.rstrip("/") + "/")
        if inside and len(mount) > len(best):
            best, kind = mount, fields[2]
    return best or None, kind


def _required_filesystem() -> str:
    return os.environ.get("NPEXP_KERNEL_STORE_FS", "ext4")


def _check_filesystem(path: Path, what: str) -> None:
    required = _required_filesystem()
    if required == "any":
        return
    mount, kind = _mount(path)
    if kind != required or mount == "/":
        raise KernelStoreUnavailable(
            f"{what} {path} is on a {kind} filesystem mounted at {mount}, not {required} on its own disk: "
            "the scratch disk is probably not mounted. Refusing to read or write kernels.")


def require_store(work: Path | None = None) -> Path:
    """The kernel store directory, after checking it exists on the required filesystem.

    ``NPEXP_KERNEL_STORE_FS`` names that filesystem (default ``ext4``, never the ``/``
    mount); ``any`` disables the check and is meant for tests on synthetic data only.
    """
    store = Path(work or work_root()) / "kernels"
    resolved = store.resolve()
    if _required_filesystem() == "any":
        resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise KernelStoreUnavailable(
            f"kernel store {store} -> {resolved} does not exist. If the scratch disk is not mounted, "
            "mount it; kernels are never written anywhere else.")
    _check_filesystem(resolved, "kernel store")
    return store


def kernel_file(store: str, regime: str, key: str, tap: int, work: Path | None = None) -> Path:
    if tap not in TAPS:
        raise ValueError(f"tap must be one of {TAPS}")
    return require_store(work) / store / regime / f"tap{tap}" / f"{key}.npz"


def save(file: Path, *, train: np.ndarray, cross: np.ndarray, train_rows: np.ndarray,
         target_rows: np.ndarray, meta: dict, target: np.ndarray | None = None) -> None:
    """Write one cell's kernels atomically. ``cross`` is (target rows, train rows)."""
    if train.shape != (len(train_rows), len(train_rows)) or cross.shape != (len(target_rows), len(train_rows)):
        raise ValueError(f"kernel shapes {train.shape}, {cross.shape} do not match the row lists")
    arrays = {"train": np.asarray(train, dtype=np.float32), "cross": np.asarray(cross, dtype=np.float32),
              "train_rows": np.asarray(train_rows, dtype=np.int64),
              "target_rows": np.asarray(target_rows, dtype=np.int64),
              "meta": np.array(json.dumps(meta, sort_keys=True, default=str))}
    if target is not None:
        if target.shape != (len(target_rows), len(target_rows)):
            raise ValueError("target kernel shape does not match the target rows")
        arrays["target"] = np.asarray(target, dtype=np.float32)
    if not all(np.isfinite(arrays[k]).all() for k in ("train", "cross")):
        raise ValueError(f"non-finite kernel for {file}")
    # Checked at every write: if the pool went away, creating the directories below
    # would silently put kernels on the root filesystem.
    existing = file.resolve().parent
    while not existing.exists():
        existing = existing.parent
    _check_filesystem(existing, "kernel directory")
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_name(file.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp, file)


def _positions(wanted: np.ndarray, stored: np.ndarray, what: str) -> np.ndarray:
    wanted = np.asarray(wanted, dtype=np.int64)
    if wanted.shape == stored.shape and np.array_equal(wanted, stored):
        return np.arange(len(stored))
    order = np.argsort(stored, kind="stable")
    found = np.searchsorted(stored[order], wanted)
    found[found == len(stored)] = 0
    position = order[found]
    if not np.array_equal(stored[position], wanted):
        raise KeyError(f"{what}: {int((stored[position] != wanted).sum())} of {len(wanted)} rows "
                       "are not in the stored kernel; rerun the kernel job for this cell")
    return position


def kernels(cell, tap: int, *, splits=None, work: Path | None = None):
    """``KernelSet`` for a cell. Raises :class:`KernelMissing` if the job has not run."""
    from common.ridge import KernelSet
    if splits is None:
        from common import splits as splits_module
        splits = splits_module
    file = kernel_file(STORE_OF_TIER[cell.tier], cell.regime, cell.key(), tap, work)
    if not file.is_file():
        raise KernelMissing(f"{file} is absent; run ensemble/launch/mapa_kernels.sh for "
                            f"store {STORE_OF_TIER[cell.tier]}, regime {cell.regime}")
    split = splits.get_cell(cell)
    with np.load(file, allow_pickle=False) as data:
        train_pos = _positions(split.rows["train"], data["train_rows"], f"{cell.key()} train")
        part_pos = {part: _positions(split.rows[part], data["target_rows"], f"{cell.key()} {part}")
                    for part in ("select", "eval")}
        full = data["train"].astype(np.float64)
        identity = np.array_equal(train_pos, np.arange(full.shape[0]))
        train = full if identity else full[np.ix_(train_pos, train_pos)]
        cross_all = data["cross"].astype(np.float64)
        cross = {part: cross_all[np.ix_(pos, train_pos)] for part, pos in part_pos.items()}
        extra = {}
        if "target" in data.files:
            target = data["target"].astype(np.float64)
            sel, ev = part_pos["select"], part_pos["eval"]
            extra["cross_gram"] = {"select": target[np.ix_(sel, sel)], "eval": target[np.ix_(ev, ev)],
                                   "eval_select": target[np.ix_(ev, sel)]}
    return KernelSet(train=train, cross=cross, **extra)


def meta(cell, tap: int, work: Path | None = None) -> dict:
    file = kernel_file(STORE_OF_TIER[cell.tier], cell.regime, cell.key(), tap, work)
    if not file.is_file():
        raise KernelMissing(str(file))
    with np.load(file, allow_pickle=False) as data:
        return json.loads(str(data["meta"]))

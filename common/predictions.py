"""Per-trial score files shared by every ensemble member and the classical stack.

One file per member and cell:
``<root>/preds/<tier>/<regime>/<member>/<target>__<train>__<task>__f<fold>.npz``
with ``row`` (int32 index in the benchmark dataset the part belongs to), ``part``
(uint8: 0 train out-of-fold, 1 select, 2 eval), ``score`` (float32) and ``meta``
(JSON string). Labels are never stored: the writer rejects anything label-like.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARTS = ("train", "select", "eval")
_CODE = {name: code for code, name in enumerate(PARTS)}
_FORBIDDEN_META = ("label", "labels", "y", "y_true", "target", "targets")


class RowSetMismatch(ValueError):
    """Two members (or a member and the split) scored different rows for a cell."""


@dataclass
class Pred:
    rows: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    meta: dict


def path(cell, member: str, root: Path) -> Path:
    if not member or "/" in member or member.startswith("."):
        raise ValueError(f"invalid member name {member!r}")
    if cell.regime == "within" and cell.tier in ("V", "T"):
        from common.confirm_fallback import PROTOCOL
        root = Path(root) / "amendments" / PROTOCOL
    return Path(root) / "preds" / cell.tier / cell.regime / member / f"{cell.key()}.npz"


def write(cell, member: str, rows: dict, scores: dict, meta: dict, root: Path) -> Path:
    """Write one member's scores for one cell. ``scores`` decides which parts are stored."""
    unknown = set(scores) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}; expected a subset of {PARTS}")
    if not scores:
        raise ValueError("no scores to write")
    bad = [k for k in meta if k.lower() in _FORBIDDEN_META]
    if bad:
        raise ValueError(f"meta keys {bad} look like labels; labels are never stored with predictions")
    row_parts, code_parts, score_parts = [], [], []
    for name in PARTS:
        if name not in scores:
            continue
        r = np.asarray(rows[name])
        s = np.asarray(scores[name], dtype=np.float64)
        if r.ndim != 1 or s.shape != r.shape:
            raise ValueError(f"{member} {cell.key()} {name}: rows {r.shape} and scores {s.shape} differ")
        if not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"{name}: rows must be integers, got {r.dtype}")
        if len(r) and (r.min() < 0 or r.max() > np.iinfo(np.int32).max):
            raise ValueError(f"{name}: row index out of int32 range")
        if len(np.unique(r)) != len(r):
            raise ValueError(f"{member} {cell.key()} {name}: duplicate rows")
        if not np.isfinite(s).all():
            raise ValueError(f"{member} {cell.key()} {name}: non-finite scores")
        row_parts.append(r.astype(np.int32))
        code_parts.append(np.full(len(r), _CODE[name], dtype=np.uint8))
        score_parts.append(s.astype(np.float32))
    full_meta = dict(meta, member=member, tier=cell.tier, regime=cell.regime,
                     train_session=cell.train_session, target_session=cell.target_session,
                     task=cell.task, fold=int(cell.fold))
    out = path(cell, member, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, row=np.concatenate(row_parts), part=np.concatenate(code_parts),
                     score=np.concatenate(score_parts),
                     meta=np.array(json.dumps(full_meta, sort_keys=True, default=_jsonable)))
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out


def read(cell, member: str, root: Path) -> Pred:
    return read_file(path(cell, member, root))


def read_file(file: Path) -> Pred:
    """Decode one score export, including original files audited for an amended run."""
    with np.load(file, allow_pickle=False) as data:
        if set(data.files) != {"row", "part", "score", "meta"}:
            raise ValueError(f"{file}: unexpected arrays {sorted(data.files)}")
        row, part, score = data["row"], data["part"], data["score"]
        meta = json.loads(str(data["meta"]))
    rows, scores = {}, {}
    for name, code in _CODE.items():
        mask = part == code
        if mask.any():
            rows[name] = row[mask].astype(np.int64)
            scores[name] = score[mask].astype(np.float64)
    return Pred(rows, scores, meta)


def exists(cell, member: str, root: Path) -> bool:
    return path(cell, member, root).is_file()


def align(scores: np.ndarray, rows: np.ndarray, reference_rows: np.ndarray, what: str) -> np.ndarray:
    """Reorder ``scores`` (given for ``rows``) to ``reference_rows``; raise if the row sets differ."""
    rows = np.asarray(rows, dtype=np.int64)
    reference_rows = np.asarray(reference_rows, dtype=np.int64)
    if rows.shape == reference_rows.shape and np.array_equal(rows, reference_rows):
        return np.asarray(scores)
    order, ref_order = np.argsort(rows, kind="stable"), np.argsort(reference_rows, kind="stable")
    if rows.shape != reference_rows.shape or not np.array_equal(rows[order], reference_rows[ref_order]):
        only = np.setdiff1d(rows, reference_rows)
        missing = np.setdiff1d(reference_rows, rows)
        raise RowSetMismatch(
            f"{what}: row sets differ ({len(rows)} vs {len(reference_rows)} rows; "
            f"{len(only)} unexpected, first {only[:5].tolist()}; {len(missing)} missing, "
            f"first {missing[:5].tolist()})")
    out = np.empty(len(rows), dtype=np.asarray(scores).dtype)
    out[ref_order] = np.asarray(scores)[order]
    return out


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

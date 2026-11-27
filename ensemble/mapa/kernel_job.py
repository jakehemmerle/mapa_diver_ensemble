"""Storage-bounded job that turns MAPA features into per-cell kernels (taps 0 and 12).

    python -m ensemble.mapa.kernel_job --store P|E --regimes within,cross_session,cross_subject
                                       [--dry-run] [--skip-parity] [--parity-only [--parity-tasks ...]]

Store ``P`` holds Tier P kernels, store ``E`` those of the 12 evaluation sessions (shared
by Tiers V and T; built from the Tier T cells' rows, never from a label).

Per-contact features (within- and cross-session) are about 65-75 GiB per session, so the
plan keeps at most two sessions' per-contact features on disk: encode, reduce to kernels,
delete. Cross-subject uses region means: the reproduction's caches where they exist
(read-only), our own encodes otherwise. Nothing here reads a label except the parity
step, which reads training and validation labels of one evaluation session.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import ledger, mapa_features
from ensemble import paths
from ensemble.mapa import encode_session
from ensemble.mapa.features import Features, OwnFeatures, ReproFeatures, shared_columns

TIER_OF_STORE = {"P": "P", "E": "T"}
VRAM_NEED_MIB = 6144
ACCUMULATOR_BUDGET = 2 * paths.GIB       # float64 kernel accumulators held on the device per pass
ASSUMED_FLOPS = 1.0e12                   # float32 matmul rate assumed for time estimates (not measured)
ENCODE_MIN = {True: 12.0, False: 5.0}    # minutes per session: with / without per-contact taps
PREPARE_MIN = 6.0
ASSUMED_CONTACTS, ASSUMED_REGIONS = 120, 20
WIDTH = {0: 348, 12: 19968}
# Parity with the reproduction's cross-subject readout (validation quantities only):
# (a) with the penalty forced to the reproduction's choice, every validation AUROC agrees
#     within 1e-4; (b) where our chosen penalty differs, the readout's validation AUROCs at
#     the two penalties differ by less than 2e-3 (a near-tie);
# (c) a fit that fails (a) is still accepted if the same fit moves by at least as much as
#     its gap when nothing but dtype or device changes (this device float32, CPU float32,
#     CPU float64 products), the gap is under 2e-3, and at most 2 of the fits need it.
# Pass needs (b) for every fit and (a) or (c) for every fit.
PARITY_FORCED_TOLERANCE = 1e-4
PARITY_TIE_TOLERANCE = 2e-3
PARITY_SENSITIVITY_GAP = 2e-3
PARITY_SENSITIVITY_MAX_FITS = 2
PARITY_CRITERION = ("(a) at the reproduction's penalty every validation AUROC within 1e-4; (b) where the chosen "
                    "penalty differs, validation AUROCs at the two penalties differ by < 2e-3; (c) a fit failing (a) "
                    "is accepted only if its value at the reproduction's penalty moves by at least the gap when only "
                    "dtype or device changes (device float32, CPU float32, CPU float64), the gap is < 2e-3, and at "
                    "most 2 fits need (c)")
PARITY_TARGET = "btbank3_0"


def _splits():
    from common import splits
    return splits


# --------------------------------------------------------------------------- locations

def own_features(session: str) -> Path:
    return paths.scratch() / "mapa" / "features" / session


def repro_region_cache(session: str) -> Path:
    subject, trial = paths.parse_session(session)
    return paths.repro() / "work/features/mapa_cs" / f"enc_s{subject}_t{trial}_mapa_cs.pt"


def prepared_dir(session: str) -> tuple[Path, bool]:
    """Prepared band caches: the reproduction's if complete (read-only), else ours.
    The flag says whether the directory exists."""
    subject, trial = paths.parse_session(session)
    theirs = paths.repro() / "work/prepared" / f"s{subject}_t{trial}"
    if (theirs / "preprocessing.json").is_file():
        return theirs, True
    ours = paths.scratch() / "mapa" / "prepared" / f"s{subject}_t{trial}"
    return ours, (ours / "preprocessing.json").is_file()


def region_features(session: str, store: str) -> Features | None:
    if (own_features(session) / "meta.json").is_file():
        return OwnFeatures(own_features(session))
    if store == "E" and repro_region_cache(session).is_file():
        return ReproFeatures(repro_region_cache(session), session, paths.parse_session(session)[0])
    return None


# --------------------------------------------------------------------------- planning

def pair_cells(store: str, regime: str, train: str, target: str):
    return [c for c in _splits().list_cells(TIER_OF_STORE[store], regime)
            if c.train_session == train and c.target_session == target]


def pending(store: str, regime: str, train: str, target: str) -> bool:
    return any(not mapa_features.kernel_file(store, regime, c.key(), tap).is_file()
               for c in pair_cells(store, regime, train, target) for tap in mapa_features.TAPS)


AVAILABLE_ONLY = False


def _pairs(tier: str, regime: str) -> list[tuple[str, str]]:
    """Session pairs of a regime; with ``--available-only`` those whose recordings exist."""
    found = _splits().pairs(tier, regime)
    if AVAILABLE_ONLY:
        found = [p for p in found if all(paths.recording(s).is_file() for s in p)]
    return found


def build_plan(store: str, regimes: list[str]) -> list[tuple]:
    """Ordered operations. At most two sessions hold per-contact features at any point."""
    splits = _splits()
    tier = TIER_OF_STORE[store]
    for regime in regimes:
        for session in {s for pair in splits.pairs(tier, regime) for s in pair}:
            paths.parse_session(session)       # refuses 6_0 and partial sessions
    within = {t for _, t in _pairs(tier, "within") if pending(store, "within", t, t)} if "within" in regimes else set()
    directed = [p for p in _pairs(tier, "cross_session") if pending(store, "cross_session", *p)] \
        if "cross_session" in regimes else []
    if directed and MAX_HELD < 2:
        raise SystemExit("cross_session kernels need the per-contact features of two sessions at once; "
                         "they cannot run with --max-held 1. Run within and cross_subject now and "
                         "cross_session with --max-held 2 when the disk allows it.")
    cross_subject = [p for p in _pairs(tier, "cross_subject") if pending(store, "cross_subject", *p)] \
        if "cross_subject" in regimes else []
    cached = (lambda s: store == "E" and repro_region_cache(s).is_file())
    ops: list[tuple] = []
    # Cross-subject pairs fully served by the reproduction's region caches need no encode.
    for anchor, target in cross_subject:
        if cached(anchor) and cached(target):
            ops.append(("cross_subject", anchor, target))
    own_cs = [(a, t) for a, t in cross_subject if not (cached(a) and cached(t))]
    cs_targets = {t for _, t in own_cs if not cached(t)}
    cs_anchors = {a for a, _ in own_cs if not cached(a)}
    remaining = {frozenset(p) for p in directed}
    involved = within | {s for pair in remaining for s in pair} | cs_targets | cs_anchors
    subject = lambda s: paths.parse_session(s)[0]
    anchor_subjects = {subject(a) for a in cs_anchors}
    done: set[str] = set()

    def degree(s):
        return sum(s in pair for pair in remaining)

    def visit(s, elec):
        ops.append(("encode", s, elec))
        if s in done:          # re-encoded only to serve as a cross-session partner or hub
            return
        if s in within:
            ops.append(("within", s))
        for a, t in own_cs:
            if t == s:
                ops.append(("cross_subject", a, t))
        done.add(s)

    def leave(s):
        ops.append(("delete_regions_later", s) if s in cs_anchors else ("delete", s))

    for subj in sorted({subject(s) for s in involved}, key=lambda x: (x not in anchor_subjects, x)):
        group = sorted(s for s in involved if subject(s) == subj)
        # anchors first so that their region features exist before any target is visited
        for s in [s for s in group if s in cs_anchors and degree(s) == 0]:
            visit(s, s in within)
            leave(s)
        hub = None
        while any(degree(s) for s in group):
            if hub is None:
                hub = max(group, key=lambda s: (degree(s), s in cs_anchors, s))
                visit(hub, True)
            partners = sorted({p for pair in remaining if hub in pair for p in pair if p != hub},
                              key=lambda s: (degree(s), s))
            next_hub = None
            for index, partner in enumerate(partners):
                visit(partner, True)
                for train, target in ((hub, partner), (partner, hub)):
                    if (train, target) in directed:
                        ops.append(("cross_session", train, target))
                remaining.discard(frozenset((hub, partner)))
                if index == len(partners) - 1 and degree(partner):
                    next_hub = partner          # stays on disk as the next hub
                else:
                    leave(partner)
            leave(hub)
            hub = next_hub
        for s in group:
            if s not in done:
                visit(s, s in within)
                leave(s)
    for s in sorted(cs_anchors):
        ops.append(("delete", s))
    return _compact(ops)


def _compact(ops: list[tuple]) -> list[tuple]:
    """Anchors keep their (small) region features until the end; only the per-contact
    files are dropped at the point a non-anchor session would be deleted."""
    return [("delete_elec", op[1]) if op[0] == "delete_regions_later" else op for op in ops]


def session_onsets(store: str, regimes: list[str], session: str) -> np.ndarray:
    """Window onsets a session must have encoded: every row any requested cell uses."""
    splits = _splits()
    tier = TIER_OF_STORE[store]
    found = []
    for regime in regimes:
        planned = set(_pairs(tier, regime))
        for cell in splits.list_cells(tier, regime):
            if session not in (cell.train_session, cell.target_session):
                continue
            if (cell.train_session, cell.target_session) not in planned:
                continue
            split = splits.get_cell(cell)
            if cell.train_session == session:
                found.append(split.onsets("train"))
            if cell.target_session == session:
                found += [split.onsets("select"), split.onsets("eval")]
    return np.unique(np.concatenate(found)) if found else np.array([], dtype=np.int64)


def kernel_bytes(store: str, regime: str, train: str, target: str) -> tuple[int, float, int]:
    """Bytes on disk, matmul flops per feature column and cell count for one session pair."""
    splits = _splits()
    total, flops, count = 0, 0.0, 0
    for cell in pair_cells(store, regime, train, target):
        split = splits.get_cell(cell)
        n, m = len(split.rows["train"]), len(split.rows["select"]) + len(split.rows["eval"])
        for tap in mapa_features.TAPS:
            with_target = store == "P" and tap == 0
            total += 4 * (n * n + m * n + (m * m if with_target else 0))
            flops += 2.0 * WIDTH[tap] * (n * n + m * n + (m * m if with_target else 0))
        count += 1
    return total, flops, count


def describe(store: str, regimes: list[str], ops: list[tuple]) -> dict:
    """Disk, time and memory estimates for a plan, printed by ``--dry-run``."""
    lines, on_disk, peak, kernels, minutes, prepared = [], {}, 0.0, 0.0, 0.0, 0.0
    counted: set[str] = set()
    start_free = paths.free_gib()
    for op in ops:
        kind = op[0]
        if kind == "encode":
            _, session, elec = op
            n = len(session_onsets(store, regimes, session))
            _, have_prepared = prepared_dir(session)
            size = encode_session.estimate_gib(n, ASSUMED_CONTACTS, ASSUMED_REGIONS, elec)
            took = ENCODE_MIN[elec] + (0 if have_prepared else PREPARE_MIN)
            prepared += 0 if have_prepared or session in counted else 1.7
            counted.add(session)
            on_disk[session] = size
            minutes += took
            lines.append(f"encode {session:11s} windows={n:6d} per-contact={'yes' if elec else 'no '} "
                         f"~{size:5.1f} GiB ~{took:4.1f} min" + ("" if have_prepared else " (+prepare, 1.7 GiB kept)"))
            if not paths.recording(session).is_file():
                lines[-1] += f"  MISSING {paths.recording(session)}"
        elif kind in ("within", "cross_session", "cross_subject"):
            train, target = (op[1], op[1]) if kind == "within" else (op[1], op[2])
            size, flops, count = kernel_bytes(store, kind, train, target)
            columns = ASSUMED_REGIONS / 2 if kind == "cross_subject" else ASSUMED_CONTACTS
            took = flops * columns / ASSUMED_FLOPS / 60
            kernels += size / paths.GIB
            minutes += took
            lines.append(f"{kind:13s} {train} -> {target}: {count} cells x 2 taps, "
                         f"{size / paths.GIB:5.1f} GiB kernels (pool), ~{took:5.1f} min")
        elif kind == "delete":
            on_disk.pop(op[1], None)
            lines.append(f"delete {op[1]}")
        elif kind == "delete_elec":
            on_disk[op[1]] = encode_session.estimate_gib(13600, 0, ASSUMED_REGIONS, False)
            lines.append(f"delete per-contact features of {op[1]} (region features kept)")
        peak = max(peak, sum(on_disk.values()) + prepared)
    pool = mapa_features.require_store().resolve()
    return {"lines": lines, "peak_gib": peak, "kernel_gib": kernels, "minutes": minutes,
            "free_gib": start_free, "free_after_peak_gib": start_free - peak,
            "pool": str(pool), "pool_free_gib": paths.free_gib(pool)}


# --------------------------------------------------------------------------- execution

def ensure_prepared(session: str) -> Path:
    folder, ready = prepared_dir(session)
    if ready:
        return folder
    subject, trial = paths.parse_session(session)
    paths.require_disk(3, f"preparing {session}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(folder.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    env = dict(os.environ, ROOT_DIR_BRAINTREEBANK=str(paths.data_root()))
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_session.py")),
                    str(subject), str(trial), str(staging / "recording")], check=True, env=env)
    os.replace(staging / "recording", folder)
    staging.rmdir()
    return folder


def run_encode(store: str, regimes: list[str], session: str, elec: bool) -> None:
    out = own_features(session)
    if (out / "meta.json").is_file():
        have = OwnFeatures(out)
        if (have.has("enc12_elec") or not elec) and np.isin(session_onsets(store, regimes, session), have.onsets).all():
            return
        shutil.rmtree(out)
    onsets = session_onsets(store, regimes, session)
    need = encode_session.estimate_gib(len(onsets), ASSUMED_CONTACTS, ASSUMED_REGIONS, elec)
    for stale in out.parent.glob(f"{out.name}.*"):       # staging left by an interrupted encode
        shutil.rmtree(stale)
    paths.require_disk(need, f"encoding {session}")
    held = [p.parent.name for p in (paths.scratch() / "mapa" / "features").glob("*/enc12_elec.npy")]
    if elec:
        if MAX_HELD == 1:
            hold_single_slot(session)
            held = [p.parent.name for p in (paths.scratch() / "mapa" / "features").glob("*/enc12_elec.npy")]
        if len(held) >= MAX_HELD:
            raise SystemExit(f"per-contact features of {held} are on disk; at most {MAX_HELD} session(s) may be held")
    encode_session.encode(session, ensure_prepared(session), out, onsets, elec=elec)


MAX_HELD = 2
_slot = None


def hold_single_slot(session: str) -> None:
    """With ``--max-held 1`` every kernel job on the machine shares one slot for per-contact
    features: wait here until no other job holds a session's features."""
    global _slot
    if _slot is not None:
        return
    import fcntl
    file = paths.scratch() / "mapa" / "held.lock"
    file.parent.mkdir(parents=True, exist_ok=True)
    handle = file.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[slot] waiting for the per-contact slot before encoding {session}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
    _slot = handle


def release_slot() -> None:
    global _slot
    if _slot is not None:
        _slot.close()
        _slot = None


def run_kernels(store: str, regime: str, train: str, target: str, device) -> None:
    import torch
    from ensemble.mapa import gram
    splits = _splits()
    region = regime == "cross_subject"
    if region:
        train_features, target_features = region_features(train, store), region_features(target, store)
    else:
        train_features = OwnFeatures(own_features(train))
        target_features = train_features if target == train else OwnFeatures(own_features(target))
    if train_features is None or target_features is None:
        raise FileNotFoundError(f"features for {train} or {target} are absent")
    cells = pair_cells(store, regime, train, target)
    for tap in mapa_features.TAPS:
        todo = [c for c in cells if not mapa_features.kernel_file(store, regime, c.key(), tap).is_file()]
        if not todo:
            continue
        name = f"enc{tap}" if region else f"enc{tap}_elec"
        if target_features is train_features:
            columns_a = columns_b = list(range(train_features.n_columns(name)))
            shared = (train_features.parcels if region else train_features.elec_labels).tolist()
        else:
            columns_a, columns_b, shared = shared_columns(train_features, target_features, name)
        with_target = store == "P" and tap == 0
        jobs, rows = [], {}
        for cell in todo:
            split = splits.get_cell(cell)
            target_rows = np.concatenate([split.rows["select"], split.rows["eval"]])
            target_onsets = np.concatenate([split.onsets("select"), split.onsets("eval")])
            rows[cell.key()] = (split.rows["train"], target_rows)
            jobs.append(gram.Job(cell.key(), train_features.positions(split.onsets("train")),
                                 target_features.positions(target_onsets), with_target))

        def fetch(i):
            a = train_features.column(name, columns_a[i])
            if target_features is train_features:
                return a, a
            return a, target_features.column(name, columns_b[i])

        meta = {"store": store, "regime": regime, "tap": tap, "feature": name,
                "train_source": train_features.source, "target_source": target_features.source,
                "n_columns": len(columns_a), "columns": shared, "standardized_on": "training rows",
                "flags": {**train_features.flags, **target_features.flags},
                "mapa_revision": paths.REVISIONS["MAPA"], "commit": ledger.commit()}
        for group in gram.passes(jobs, ACCUMULATOR_BUDGET):
            started = time.monotonic()
            gram.accumulate(group, len(columns_a), fetch, device)
            for job in group:
                train_rows, target_rows = rows[job.key]
                mapa_features.save(mapa_features.kernel_file(store, regime, job.key, tap),
                                   train=job.out["train"], cross=job.out["cross"], target=job.out.get("target"),
                                   train_rows=train_rows, target_rows=target_rows, meta=meta)
                job.out = {}
            print(f"[kernels] {regime} {train}->{target} tap{tap}: {len(group)} cells, "
                  f"{len(columns_a)} columns, {time.monotonic() - started:.0f} s", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def remove(session: str, elec_only: bool) -> None:
    folder = own_features(session)
    if not folder.exists():
        return
    if elec_only:
        for file in folder.glob("*_elec.npy"):
            file.unlink()
    else:
        shutil.rmtree(folder)
    print(f"[delete] {'per-contact features of ' if elec_only else ''}{session}", flush=True)


# --------------------------------------------------------------------------- parity

def _parity_grids(cells, train_features, target_features, tap: int, device, products) -> tuple[dict, list]:
    """Validation AUROC over the penalty grid for each cell at one tap (training and
    validation labels only)."""
    from common.ridge import KernelSet
    from ensemble.mapa import gram
    splits = _splits()
    name = f"enc{tap}"
    columns_a, columns_b, shared = shared_columns(train_features, target_features, name)
    jobs = [gram.Job(cell.key(), train_features.positions(splits.get_cell(cell).onsets("train")),
                     target_features.positions(splits.get_cell(cell).onsets("select"))) for cell in cells]
    gram.accumulate(jobs, len(columns_a),
                    lambda i: (train_features.column(name, columns_a[i]), target_features.column(name, columns_b[i])),
                    device, products=products)
    grids = {}
    for cell, job in zip(cells, jobs):
        split = splits.get_cell(cell)
        grids[cell.key()] = penalty_grid([KernelSet(train=job.out["train"], cross={"select": job.out["cross"]})],
                                         split.labels("train"), split.labels("select"))
    return grids, shared


def parity(tasks: list[str] | None, device) -> dict:
    """Recompute MAPA's reproduced cross-subject readout for one evaluation session from
    the reproduction's cached region features, through this job's kernel code and
    ``common.ridge``, and compare validation AUROC and the chosen penalty with the
    reproduction's ``results/cs_S3T0.json``. Training and validation labels only."""
    from common import ridge
    splits = _splits()
    subject, trial = paths.parse_session(PARITY_TARGET)
    reference_file = paths.repro() / "results" / f"cs_S{subject}T{trial}.json"
    reference = json.loads(reference_file.read_text())["cells"]
    anchor = splits.EVAL_ANCHOR
    train_features, target_features = region_features(anchor, "E"), region_features(PARITY_TARGET, "E")
    if not isinstance(train_features, ReproFeatures) or not isinstance(target_features, ReproFeatures):
        raise FileNotFoundError(f"parity needs {repro_region_cache(anchor)} and {repro_region_cache(PARITY_TARGET)}")
    cells = [c for c in pair_cells("E", "cross_subject", anchor, PARITY_TARGET) if not tasks or c.task in tasks]
    import torch
    cpu = torch.device("cpu")
    rows, worst = [], 0.0
    for tap in mapa_features.TAPS:
        name = f"enc{tap}"
        grids, shared = _parity_grids(cells, train_features, target_features, tap, device, torch.float32)
        tap_rows = []
        for cell in cells:
            grid = grids[cell.key()]
            entry = next(v for k, v in reference.items() if k.rsplit("|", 1)[1] == cell.task)
            theirs = entry["cells"][f"{name}|std"]
            ours_index = int(np.flatnonzero(grid == np.nanmax(grid))[0])     # smallest penalty on ties
            their_index = int(np.flatnonzero(np.isclose(ridge.LAM_MULTS, theirs["lam_mult"], rtol=1e-9))[0])
            forced_gap = abs(grid[their_index] - theirs["val"])             # (a) at the reproduction's penalty
            tie_gap = abs(grid[ours_index] - grid[their_index])             # (b) where the choice differs
            row = {"task": cell.task, "tap": tap, "regions": len(shared), "ours_val": float(grid[ours_index]),
                   "ours_lam": ridge.LAM_MULTS[ours_index], "reproduction_val": theirs["val"],
                   "reproduction_lam": theirs["lam_mult"], "lam_equal": ours_index == their_index,
                   "val_at_reproduction_lam": float(grid[their_index]), "forced_gap": forced_gap,
                   "near_tie_gap": tie_gap if ours_index != their_index else 0.0,
                   "abs_gap": abs(grid[ours_index] - theirs["val"]), "n_parcels_reference": entry.get("n_parcels"),
                   "forced_ok": bool(forced_gap <= PARITY_FORCED_TOLERANCE), "needs_c": False}
            tap_rows.append((cell, their_index, row))
        failing = [(cell, index, row) for cell, index, row in tap_rows if not row["forced_ok"]]
        if failing:      # (c): how far the same fits move when only dtype or device changes
            subset = [cell for cell, _, _ in failing]
            variants = {f"{device.type}_float32": None,
                        "cpu_float32": _parity_grids(subset, train_features, target_features, tap, cpu, torch.float32)[0],
                        "cpu_float64": _parity_grids(subset, train_features, target_features, tap, cpu, torch.float64)[0]}
            for cell, index, row in failing:
                values = {key: (row["val_at_reproduction_lam"] if found is None else float(found[cell.key()][index]))
                          for key, found in variants.items()}
                spread = max(values.values()) - min(values.values())
                row["needs_c"] = True
                row["sensitivity"] = {**values, "spread": spread}
                row["c_ok"] = bool(spread >= row["forced_gap"] and row["forced_gap"] < PARITY_SENSITIVITY_GAP)
        for _, _, row in tap_rows:
            row["passed"] = bool((row["forced_ok"] or row.get("c_ok", False)) and row["near_tie_gap"] < PARITY_TIE_TOLERANCE)
            rows.append(row)
            worst = max(worst, row["forced_gap"])
            print(f"[parity] {row['task']:18s} tap{tap:<2d} val at their penalty ours={row['val_at_reproduction_lam']:.6f} "
                  f"reproduction={row['reproduction_val']:.6f} gap={row['forced_gap']:.2e}; penalty "
                  + ("equal" if row["lam_equal"] else f"differs, near-tie gap {row['near_tie_gap']:.2e}")
                  + (f"; (c) sensitivity {row['sensitivity']} -> {'accepted' if row['c_ok'] else 'not accepted'}"
                     if row["needs_c"] else ""), flush=True)
    needs_c = int(sum(r["needs_c"] for r in rows))
    return {"target": PARITY_TARGET, "anchor": anchor, "reference": str(reference_file), "rows": rows,
            "criterion": PARITY_CRITERION, "max_abs_gap": worst,
            "forced_ok": all(r["forced_ok"] for r in rows), "needs_c": needs_c,
            "max_near_tie_gap": max([r["near_tie_gap"] for r in rows] or [0.0]),
            "passed": bool(rows and all(r["passed"] for r in rows) and needs_c <= PARITY_SENSITIVITY_MAX_FITS),
            "lam_equal": int(sum(r["lam_equal"] for r in rows)), "n": len(rows)}


def penalty_grid(kernels, y_train, y_select) -> np.ndarray:
    """Select AUROC at every multiplier of MAPA's 25-point grid (``common.ridge`` arithmetic)."""
    from common import ridge
    gram, cross = ridge.combine(kernels)
    y = np.asarray(y_train, dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh(gram)
    base = float(np.trace(gram)) / gram.shape[0]
    multipliers = np.asarray(ridge.LAM_MULTS)
    alphas = vectors @ ((vectors.T @ y)[:, None] / (eigenvalues[:, None] + base * multipliers[None, :]))
    scores = cross["select"] @ alphas
    return np.array([ridge.auroc(scores[:, i], y_select) for i in range(len(multipliers))])


def run_parity(tasks, device) -> bool:
    report = parity(tasks, device)
    out = paths.results() / "parity" / "mapa_kernels_cs_S3T0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    ledger.append({"tier": "T", "regime": "cross_subject", "phase": "EK", "kind": "info",
                   "arm": "mapa_kernel_parity", "baseline": "reproduction results/cs_S3T0.json",
                   "delta": report["max_abs_gap"], "decision": "pass" if report["passed"] else "fail",
                   "extra": {"quantity": "validation AUROC and penalty; no test label read", "tasks": report["n"] // 2,
                             "lam_equal": report["lam_equal"], "n": report["n"], "criterion": PARITY_CRITERION,
                             "max_near_tie_gap": report["max_near_tie_gap"], "needs_c": report["needs_c"],
                             "sensitivity": {f"{r['task']}|tap{r['tap']}": r["sensitivity"]
                                             for r in report["rows"] if r["needs_c"]}}},
                  paths.results())
    print(f"[parity] max |val AUROC gap| at the reproduction's penalty = {report['max_abs_gap']:.2e} over "
          f"{report['n']} fits (limit {PARITY_FORCED_TOLERANCE}); penalty equal in {report['lam_equal']}; "
          f"largest near-tie gap {report['max_near_tie_gap']:.2e} (limit {PARITY_TIE_TOLERANCE}); "
          f"{report['needs_c']} fit(s) under (c) (limit {PARITY_SENSITIVITY_MAX_FITS}); "
          f"{'PASS' if report['passed'] else 'FAIL'} -> {out}", flush=True)
    if not report["passed"]:
        print("[parity] STOP: a fit fails (b), or fails (a) without the dtype/device sensitivity that (c) "
              "requires, or more fits need (c) than allowed; the kernels do not reproduce MAPA's readout.", flush=True)
    return report["passed"]


# --------------------------------------------------------------------------- shards

def execute(store: str, regimes: list[str], op: tuple, device) -> None:
    kind = op[0]
    if kind == "encode":
        run_encode(store, regimes, op[1], op[2])
    elif kind == "within":
        run_kernels(store, "within", op[1], op[1], device)
    elif kind in ("cross_session", "cross_subject"):
        run_kernels(store, kind, op[1], op[2], device)
    elif kind in ("delete", "delete_elec"):
        remove(op[1], kind == "delete_elec")
        release_slot()


def claim(store: str, session: str, wait: bool):
    """Per-session claim shared by every ``--claim`` job: an flock on /scratch, released when
    the holder finishes the session or dies. ``wait`` blocks (anchors); otherwise None if held."""
    import fcntl
    file = paths.scratch() / "mapa" / "claims" / f"{store}_{session}.lock"
    file.parent.mkdir(parents=True, exist_ok=True)
    handle = file.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
        handle.close()
        return None
    return handle


def claim_groups(ops: list[tuple]) -> list[tuple[str, list[tuple]]]:
    """Split a within/cross_subject plan into per-session groups (encode, kernels, delete).
    The final deletes of cross-subject anchors are dropped: other shards still read their
    (small) region features."""
    groups: list[tuple[str, list[tuple]]] = []
    for op in ops:
        if op[0] == "encode":
            groups.append((op[1], [op]))
        elif groups and (op[0] in ("within", "delete", "delete_elec") and op[1] == groups[-1][0]
                         or op[0] == "cross_subject" and op[2] == groups[-1][0]):
            groups[-1][1].append(op)
        elif op[0] != "delete":
            raise SystemExit(f"--claim cannot place operation {op} in a session group")
    return groups


def run_claimed(store: str, regimes: list[str], ops: list[tuple], device) -> None:
    """Several jobs (one per card) share one plan: each takes the next unclaimed session whose
    kernels are still pending. Anchors come first and are waited for, so their region
    features exist before any cross-subject kernel; the rest go in reverse plan order."""
    groups = claim_groups(ops)
    anchors = {op[1] for op in ops if op[0] == "cross_subject"}
    ordered = [g for g in groups if g[0] in anchors] + [g for g in groups if g[0] not in anchors][::-1]
    for session, group in ordered:
        handle = claim(store, session, wait=session in anchors)
        if handle is None:
            print(f"[claim] {session}: claimed by another job; skipped", flush=True)
            continue
        try:
            work = [op for op in group if op[0] in ("within", "cross_subject")]
            todo = any(pending(store, op[0], op[1], op[1]) if op[0] == "within" else pending(store, *op)
                       for op in work) or (session in anchors and region_features(session, store) is None)
            if not todo:
                print(f"[claim] {session}: kernels exist; skipped", flush=True)
                continue
            print(f"[claim] {session}: {len(group)} operations", flush=True)
            for op in group:
                execute(store, regimes, op, device)
        finally:
            handle.close()


# --------------------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("P", "E"), required=True)
    parser.add_argument("--regimes", default="within,cross_session,cross_subject")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parity", action="store_true", help="parity already passed for this code")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--parity-tasks", nargs="*", default=None)
    parser.add_argument("--max-held", type=int, choices=(1, 2, 3, 4, 5), default=2,
                        help="sessions whose per-contact features may be on disk at once (counted over every "
                             "job); 1 shares a single machine-wide slot between kernel jobs and excludes cross_session")
    parser.add_argument("--disk-floor", type=int, default=paths.DISK_FLOOR_GIB,
                        help="GiB that must stay free on the scratch filesystem (at least 250 when that "
                             "is the system disk); kernels live on the pool and are not counted")
    parser.add_argument("--available-only", action="store_true",
                        help="plan only session pairs whose recordings are on disk; rerun for the rest")
    parser.add_argument("--claim", action="store_true",
                        help="shard the plan with other --claim jobs through per-session claims on /scratch "
                             "(Tier P, within/cross_subject only)")
    args = parser.parse_args()
    global MAX_HELD, AVAILABLE_ONLY
    MAX_HELD, AVAILABLE_ONLY = args.max_held, args.available_only
    paths.DISK_FLOOR_GIB = args.disk_floor
    regimes = [r for r in args.regimes.split(",") if r]
    splits = _splits()
    unknown = set(regimes) - set(splits.REGIMES)
    if unknown:
        parser.error(f"unknown regimes {sorted(unknown)}")
    if args.claim and (args.store != "P" or "cross_session" in regimes):
        parser.error("--claim shards Tier P within/cross_subject plans only")
    try:
        if args.parity_only and not args.dry_run:
            import torch
            raise SystemExit(0 if run_parity(args.parity_tasks, torch.device("cuda" if torch.cuda.is_available() else "cpu")) else 6)
        ops = build_plan(args.store, regimes)
        report = describe(args.store, regimes, ops)
    except splits.MissingInput as error:
        raise SystemExit(f"MISSING INPUT: {error}") from None
    except mapa_features.KernelStoreUnavailable as error:
        raise SystemExit(f"KERNEL STORE: {error}") from None
    print(f"MAPA kernel job: store {args.store}, regimes {regimes}, {len(ops)} operations")
    if not ops:
        print("nothing to do: every planned kernel exists" + (" for the recordings on disk" if AVAILABLE_ONLY else ""))
        return
    for line in report["lines"]:
        print("  " + line)
    needs_gpu = any(op[0] == "encode" for op in ops)
    print(f"estimate: {report['minutes'] / 60:.1f} h wall (encode {ENCODE_MIN[True]:.0f} min/session measured by the "
          f"reproduction; kernel time assumes {ASSUMED_FLOPS:.0e} flop/s, unmeasured), "
          f"kernels {report['kernel_gib']:.0f} GiB on {report['pool']} ({report['pool_free_gib']:.0f} GiB free there), "
          f"scratch {paths.scratch()}: peak {report['peak_gib']:.0f} GiB of features and prepared caches, "
          f"free now {report['free_gib']:.0f} GiB -> {report['free_after_peak_gib']:.0f} GiB at peak "
          f"(floor {paths.floor_gib()}, at most {MAX_HELD} session(s) of per-contact features), "
          f"VRAM {VRAM_NEED_MIB} MiB, GPU {'required' if needs_gpu else 'optional'}")
    if report["kernel_gib"] > report["pool_free_gib"] - 100:
        print(f"KERNEL STORE: {report['kernel_gib']:.0f} GiB of kernels do not fit on {report['pool']} "
              f"with 100 GiB to spare. Nothing was started.")
        raise SystemExit(5)
    missing = [line for line in report["lines"] if "MISSING" in line]
    inputs = [paths.weights("mapa_vits384.pt"), paths.source("MAPA")] if needs_gpu else []
    inputs += [repro_region_cache(s) for op in ops if op[0] == "cross_subject" for s in op[1:]
               if args.store == "E" and not (own_features(s) / "meta.json").is_file() and not any(
                   o[0] == "encode" and o[1] == s for o in ops)]
    absent = sorted({str(p) for p in inputs if not Path(p).exists()})
    if missing or absent:
        for line in missing:
            print("MISSING INPUT: " + line.split("MISSING ", 1)[1])
        for item in absent:
            print("MISSING INPUT: " + item)
        raise SystemExit(2)
    if report["free_after_peak_gib"] < paths.floor_gib():
        print(f"DISK: the plan would leave {report['free_after_peak_gib']:.0f} GiB free on the scratch "
              f"filesystem, under the {paths.floor_gib()} GiB floor. Nothing was started.")
        raise SystemExit(5)
    if args.dry_run:
        print("dry run: nothing was started")
        return
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if needs_gpu and device.type != "cuda":
        raise SystemExit("this plan encodes sessions and needs one visible CUDA device")
    if not args.skip_parity and not run_parity(args.parity_tasks, device):
        raise SystemExit(6)
    started = time.monotonic()
    if args.claim:
        run_claimed(args.store, regimes, ops, device)
    else:
        for op in ops:
            execute(args.store, regimes, op, device)
    ledger.append({"tier": TIER_OF_STORE[args.store] if args.store == "P" else "V+T", "regime": ",".join(regimes),
                   "phase": "EK", "kind": "info", "arm": "mapa_kernels", "decision": "done",
                   "extra": {"store": args.store, "operations": len(ops),
                             "hours": (time.monotonic() - started) / 3600}}, paths.results())
    print(f"done in {(time.monotonic() - started) / 3600:.2f} h")


if __name__ == "__main__":
    paths.exit_on_cuda_oom(main)

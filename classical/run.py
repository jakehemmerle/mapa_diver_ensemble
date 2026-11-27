"""Phases C0-C5 of the classical track (plan: docs/06-classical-and-ensemble-plan.md).

    python -m classical.run <c0|c1|c2|c3|c4|c5> [--regimes ...] [--dry-run] [--workers N]

Every phase works per regime in stages: extract features -> fit arms and write per-trial
scores through ``common.predictions`` -> score, compare with ``common.stats`` and append
to the ledger. Fitting never reads evaluation labels. Scoring Tier T goes through the
lock (C0 as a logged reproduction, C5 as the regime's single scoring).

Rules this file implements, all fixed here before any run:

* Tier P decides everything. A block is kept if it adds >= +0.002 macro over the base.
* The weight of added kernels relative to the base kernels is one value per arm from
  {0.1, 0.25, 0.5, 1.0}, chosen on Tier P by the arm's macro AUROC on the "select" part
  (ties -> smaller weight), so the Tier P "eval" gain that drives Go/Hold/Stop is not
  used for the choice. Cross-session has no usable Tier P signal and copies the
  within-session blocks and weight. The chosen weight is written into the candidate
  config; Tiers V and T only ever read it.
* Controls run in the same invocation as their arm, at the arm's weight.
  ``controls_pass`` = shifted-sham gain < +0.002 AND the arm beats its dimension-matched
  random-projection control by >= +0.002. The phase-randomized surrogate does not gate;
  it records whether the gain needs phase.
* Session-filtered ("@session") arms are Tier P measurements, never candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import common
from classical import controls, features
from common import confirm_fallback, ridge, splits
from common.splits import Cell, MissingInput

WEIGHT_GRID = (0.1, 0.25, 0.5, 1.0)
KEEP_GAIN = 0.002
SHAM_MAX_GAIN = 0.002
RP_MIN_MARGIN = 0.002
PARITY_TOLERANCE = 0.005
RP_DEVIATION = ("C1 extra-kernel control projects the Ridge 6-band base features (not MAPA frontend features) "
                "at the block's dimension, because MAPA Tier P kernels do not exist when C1 runs; approved 2026-11-25")
PUBLISHED_RIDGE6 = {"within": 0.683, "cross_session": 0.673, "cross_subject": 0.585}   # PR #20 leaderboard macros
MIN_FREE_GIB = 250.0
ADDED_BLOCKS = ("B1", "B2", "B3", "B4", "B5")
MAPA_BASE = (("mapa", "12"), ("mapa", "0"))
BASELINES = ("mapa12", "ridge6")
REGIME_ORDER = ("within", "cross_subject", "cross_session")   # cross-session copies within's decisions
GIB = 1024 ** 3
# Cost model for --dry-run (see docs/staging/classical.md for what is measured and what is assumed).
EXTRACT_CPU_S_PER_CHANNEL_KWINDOW = 1.1     # measured (server smoke, niced): window mode, one core, per channel
#                                             per 1000 windows, recording read and normalization tiles included
EXTRACT_CPU_S_PER_CHANNEL_SESSION = 20.0    # assumed, not measured: session mode, one core, per channel
FIT_CPU_S = {"within": 1.5, "cross_session": 9.0, "cross_subject": 6.0}   # assumed: per cell and arm, 4 threads
ASSUMED_WINDOWS, ASSUMED_CHANNELS = 14000, 110                           # assumed until a cache dry run --exact


class Missing(RuntimeError):
    """An input another job must produce first."""


@dataclass(frozen=True)
class Arm:
    member: str
    kernels: tuple[tuple[str, ...], ...]   # ("c"|"sham"|"rp", block, mode, reference) or ("mapa", tap)
    weights: tuple[float, ...]
    kind: str = "member"                   # ledger kind: "member" or "control"
    oof: bool = False
    versus: str = ""                       # member this arm's gain is measured against
    role: str = ""                         # base | screen | candidate | sham | phaserand | rp | measurement | reference

    def config(self) -> dict:
        return {"member": self.member, "kernels": [list(k) for k in self.kernels], "weights": list(self.weights),
                "defaults_hash": features.defaults_hash(), "penalty_grid": [float(m) for m in ridge.LAM_MULTS]}

    def hash(self) -> str:
        return common.config_hash(self.config())


def variant(mode: str, reference: str) -> str:
    parts = [p for p in ("" if mode == "window" else mode, "" if reference == "bipolar" else reference) if p]
    return "@" + "-".join(parts) if parts else ""


def base_arm(mode: str = "window", reference: str = "bipolar", oof: bool = True) -> Arm:
    return Arm("ridge6" + variant(mode, reference), (("c", "base6", mode, reference),), (1.0,), oof=oof,
               role="base" if (mode, reference) == ("window", "bipolar") else "measurement")


def stacked(member: str, base: tuple[tuple[str, ...], ...], added: tuple[str, ...], weight: float,
            mode: str = "window", reference: str = "bipolar", **fields) -> Arm:
    kernels = base + tuple(("c", block, mode, reference) for block in added)
    return Arm(member, kernels, (1.0,) * len(base) + (float(weight),) * len(added), **fields)


def control_arms(arm: Arm, n_base: int) -> list[Arm]:
    """Sham, phase-surrogate and random-projection versions of an arm's added kernels."""
    base, added = arm.kernels[:n_base], arm.kernels[n_base:]
    out = [Arm(arm.member + "~sham", base + tuple(("sham",) + k[1:] for k in added), arm.weights,
               kind="control", versus=arm.versus, role="sham"),
           Arm(arm.member + "~rp", base + tuple(("rp",) + k[1:] for k in added), arm.weights,
               kind="control", versus=arm.versus, role="rp")]
    if any(k[1] in controls.PHASE_BLOCKS and k[2] == "window" for k in added):
        swapped = tuple(("c", k[1] + "~phaserand") + k[2:] if k[1] in controls.PHASE_BLOCKS else k for k in added)
        out.append(Arm(arm.member + "~phaserand", base + swapped, arm.weights, kind="control",
                       versus=arm.versus, role="phaserand"))
    return out


def arm_from(config: dict, **fields) -> Arm:
    return Arm(config["member"], tuple(tuple(k) for k in config["kernels"]), tuple(config["weights"]), **fields)


# --------------------------------------------------------------------------- one cell

class _CellKernels:
    def __init__(self, cell: Cell, rp_source: str) -> None:
        self.cell, self.rp_source = cell, rp_source
        self.cache: dict[tuple, ridge.KernelSet] = {}
        self.dims: dict[tuple, int] = {}
        self.projector: controls.RandomProjector | None = None

    def get(self, spec: tuple[str, ...]) -> ridge.KernelSet:
        if spec in self.cache:
            return self.cache[spec]
        kind = spec[0]
        if kind == "c":
            kernel, self.dims[spec[1:]] = features.kernels_and_dim(self.cell, *spec[1:])
        elif kind == "sham":
            kernel, _ = controls.shifted_sham(self.cell, *spec[1:])
        elif kind == "mapa":
            from common import mapa_features
            kernel = mapa_features.kernels(self.cell, int(spec[1]))
        elif kind == "rp":
            if spec[1:] not in self.dims:
                self.get(("c",) + spec[1:])
            if self.projector is None:
                if self.rp_source == "mapa0":
                    from common import mapa_features
                    source = mapa_features.kernels(self.cell, 0)
                else:
                    source, _ = features.kernels_and_dim(self.cell, "base6", "window", spec[3], cross_gram=True)
                self.projector = controls.RandomProjector(source)
            kernel = self.projector.kernel(self.dims[spec[1:]], zlib.crc32(f"{self.cell.key()}/{spec}".encode()))
        else:
            raise ValueError(f"unknown kernel spec {spec}")
        self.cache[spec] = kernel
        return kernel


def run_cell(payload: tuple) -> list[dict]:
    """Fit every pending arm of one cell and write its scores. Reads train and select labels only."""
    cell, arms, root, rp_source, overwrite = payload
    from common import ledger, predictions
    root = Path(root)
    records = [{"member": arm.member, "cell": cell.key(), "status": "existing"}
               for arm in arms if not overwrite and predictions.exists(cell, arm.member, root)]
    pending = [arm for arm in arms if overwrite or not predictions.exists(cell, arm.member, root)]
    if not pending:
        return records
    try:
        split = splits.get_cell(cell)
        y_train, y_select = split.labels("train"), split.labels("select")
    except MissingInput:
        raise
    except Exception as error:   # the benchmark cannot build this (session, task); recorded, not hidden
        return records + [{"member": arm.member, "cell": cell.key(), "status": "no-dataset",
                           "why": f"{type(error).__name__}: {error}"} for arm in pending]
    store = _CellKernels(cell, rp_source)
    uses = Counter(spec for arm in pending for spec in arm.kernels)
    for arm in pending:
        record = {"member": arm.member, "cell": cell.key()}
        strict = confirm_fallback.in_scope(cell, arm.member)
        fallback = confirm_fallback.for_select(cell, arm.member, y_select)
        try:
            kernels = [store.get(spec) for spec in arm.kernels]
        except features.BlockUnavailable as error:
            records.append(dict(record, status="unavailable", why=str(error)))
            continue
        rows = {part: split.rows[part] for part in splits.PARTS}
        meta = {"commit": ledger.commit(), "config_hash": arm.hash(), "kernels": [list(k) for k in arm.kernels],
                "weights": list(arm.weights), "role": arm.role, "defaults_hash": features.defaults_hash()}
        try:
            fit = ridge.fit_predict(kernels, y_train, y_select, weights=arm.weights,
                                    oof_blocks=split.oof_blocks() if arm.oof else None,
                                    fixed_lam=fallback["lam"] if fallback is not None else None,
                                    strict=strict)
            scores = dict(fit["scores"])
            if not strict and "train" in scores:
                finite = np.isfinite(scores["train"])
                rows["train"], scores["train"] = rows["train"][finite], scores["train"][finite]
            # The grid's matrix product selects the penalty; E0 checks the final score
            # vector at export precision. Nearly tied pairs can rank differently.
            stored_auc = None if fallback is not None else ridge.auroc(
                np.asarray(scores["select"], dtype=np.float32), y_select)
            meta.update(status="ok", lam=fit["lam"], lam_pin=fit["lam_pin"],
                        fit_select_auroc=fit["select_auroc"], select_auroc=stored_auc)
            record.update(status="ok", lam=fit["lam"], lam_pin=fit["lam_pin"], select_auroc=fit["select_auroc"])
            if fallback is not None:
                meta["confirm_fallback"] = record["confirm_fallback"] = fallback
        except ridge.Degenerate as error:
            if strict:
                raise
            # A member that cannot be fit scores at chance, so every member covers every cell.
            scores = {part: np.zeros(len(rows[part])) for part in ("select", "eval")}
            meta.update(status="degenerate", why=str(error))
            record.update(status="degenerate", why=str(error))
        if any(spec[0] == "rp" for spec in arm.kernels):
            meta["rp_source"], meta["rp_exact"] = rp_source, bool(store.projector.exact)
            record["rp_exact"] = meta["rp_exact"]
        predictions.write(cell, arm.member, rows, scores, meta, root)
        records.append(record)
        for spec in arm.kernels:
            uses[spec] -= 1
            if uses[spec] == 0:
                store.cache.pop(spec, None)
    return records


# --------------------------------------------------------------------------- a phase run

class Universe:
    """``common.splits`` restricted to the cells whose datasets the benchmark can build."""

    def __init__(self, skipped: set[tuple[str, str]]) -> None:
        self.skipped = skipped

    def list_cells(self, tier: str, regime: str) -> list[Cell]:
        return [c for c in splits.list_cells(tier, regime) if (c.target_session, c.task) not in self.skipped
                and (c.train_session, c.task) not in self.skipped]

    get_cell = staticmethod(splits.get_cell)


class Run:
    def __init__(self, phase: str, tier: str, opts: argparse.Namespace) -> None:
        self.phase, self.tier, self.opts = phase, tier, opts
        self.root = common.results_root()
        self.out = self.root / "classical"
        self.skipped: set[tuple[str, str]] = set()
        self.universe = Universe(self.skipped)   # one object: common.stats caches labels per splits object
        self.announced: set[tuple[str, str]] = set()

    # ---- bookkeeping
    def log(self, regime: str, kind: str, arm: str, **fields) -> None:
        from common import ledger
        ledger.append(dict(fields, tier=self.tier, regime=regime, phase=self.phase, kind=kind, arm=arm), self.root)

    def save(self, name: str, body: dict) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")
        os.replace(temporary, path)
        return path

    def load(self, name: str) -> dict:
        path = self.out / name
        if not path.exists():
            raise Missing(f"{path} (written by an earlier classical phase)")
        return json.loads(path.read_text())

    # ---- stages
    def extract(self, regime: str, plan: dict[tuple[str, str], tuple[str, ...]]) -> None:
        if "extract" not in self.opts.stages:
            return
        for mode, _ in plan:
            features.check_mode(self.tier, mode)
        for session in sessions_of(self.tier, regime):
            free = features.free_gib(features.features_root())
            need = estimate_feature_bytes(plan) / GIB
            if free - need < self.opts.min_free_gib:
                raise SystemExit(f"refusing to extract {session}: {free:.0f} GiB free under {features.features_root()}, "
                                 f"{need:.1f} GiB needed, floor {self.opts.min_free_gib:.0f} GiB")
            started = time.time()
            features.extract(session, plan, workers=self.opts.workers)
            print(f"[{self.phase}] features {session} {sorted(plan)} {time.time() - started:.0f} s", flush=True)

    def fit(self, regime: str, arms: list[Arm]) -> None:
        arms = self.available(regime, arms)
        if "fit" not in self.opts.stages or not arms:
            return
        cells = splits.list_cells(self.tier, regime)
        payloads = [(cell, arms, str(self.root), self.opts.rp_source, self.opts.overwrite) for cell in cells]
        started = time.time()
        if self.opts.fit_workers <= 1:
            results = [run_cell(p) for p in payloads]
        else:
            import multiprocessing
            # One chunk per (train, target) pair: a worker builds each benchmark dataset once.
            per_pair = len(splits.TASKS) * (2 if regime == "within" else 1)
            with ProcessPoolExecutor(self.opts.fit_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(run_cell, payloads, chunksize=per_pair))
        by_member: dict[str, Counter] = {}
        for cell, records in zip(cells, results):
            for record in records:
                by_member.setdefault(record["member"], Counter())[record["status"]] += 1
                if record["status"] in ("no-dataset", "unavailable"):
                    # The cell leaves every comparison of this run (no channel, region or dataset for it).
                    self.skipped.add((cell.target_session, cell.task))
        print(f"[{self.phase}] fit {regime}: {len(cells)} cells x {len(arms)} arms in {time.time() - started:.0f} s",
              flush=True)
        for member, counts in sorted(by_member.items()):
            print(f"[{self.phase}]   {member}: {dict(counts)}", flush=True)
        if self.skipped:
            self.log(regime, "info", "skipped-datasets", extra={"skipped": sorted(self.skipped)})

    def available(self, regime: str, arms: list[Arm]) -> list[Arm]:
        """Arms defined in this regime. B4 is per shaft and has no cross-subject form."""
        def defined(arm: Arm) -> bool:
            return not (regime == "cross_subject" and any(k[1].split("~")[0] == "B4" for k in arm.kernels if k[0] != "mapa"))
        for arm in arms:
            if not defined(arm) and (regime, arm.member) not in self.announced:
                self.announced.add((regime, arm.member))
                self.log(regime, "info", arm.member, decision="not-applicable",
                         extra={"why": "B4 is defined per shaft and cannot be aligned across subjects"})
        return [arm for arm in arms if defined(arm)]

    def macro(self, member: str, regime: str, part: str = "eval") -> float:
        from common import stats
        return stats.macro(member, self.tier, regime, self.root, part=part, splits=self.universe)

    def report(self, regime: str, arms: list[Arm]) -> dict[str, float]:
        """Eval macro of each arm and its gain over ``versus``; one ledger row per arm."""
        scores: dict[str, float] = {}
        for arm in arms:
            scores[arm.member] = self.macro(arm.member, regime)
            if arm.versus and arm.versus not in scores:
                scores[arm.versus] = self.macro(arm.versus, regime)
        for arm in arms:
            reference = scores.get(arm.versus)
            self.log(regime, arm.kind, arm.member, config_hash=arm.hash(), score=scores[arm.member],
                     baseline=arm.versus or None, baseline_score=reference,
                     delta=None if reference is None else scores[arm.member] - reference,
                     extra={"role": arm.role, "weights": list(arm.weights), "kernels": [list(k) for k in arm.kernels],
                            "select_macro": self.macro(arm.member, regime, part="select"),
                            **({"rp_source": self.opts.rp_source,
                                "declared_deviation": RP_DEVIATION if self.opts.rp_source == "base6" else None}
                               if arm.role == "rp" else {})})
        return scores

    def choose_weight(self, regime: str, grid: dict[float, Arm]) -> float:
        """Weight with the best Tier P select-part macro; ties go to the smaller weight."""
        if self.tier != "P":
            raise RuntimeError("kernel weights are chosen on Tier P only")
        select = {weight: self.macro(arm.member, regime, part="select") for weight, arm in grid.items()}
        best = max(select.values())
        chosen = min(weight for weight, value in select.items() if value == best)
        for weight, arm in grid.items():
            self.log(regime, "info", arm.member, config_hash=arm.hash(), score=self.macro(arm.member, regime),
                     decision="weight-chosen" if weight == chosen else "weight-not-chosen",
                     extra={"role": "weight-screen", "weight": weight, "select_macro": select[weight]})
        return chosen

    def verdict(self, regime: str, candidate: Arm, shadows: list[Arm], scores: dict[str, float]) -> dict:
        reference = scores[candidate.versus]
        gains = {arm.role: scores[arm.member] - reference for arm in shadows if arm.member in scores}
        gain = scores[candidate.member] - reference
        out = {"member": candidate.member, "versus": candidate.versus, "gain": gain, "control_gains": gains,
               "sham_pass": "sham" in gains and gains["sham"] < SHAM_MAX_GAIN,
               "rp_pass": "rp" in gains and gain - gains["rp"] >= RP_MIN_MARGIN,
               "needs_phase": None if "phaserand" not in gains else bool(gains["phaserand"] < KEEP_GAIN)}
        out["controls_pass"] = bool(out["sham_pass"] and out["rp_pass"])
        self.log(regime, "control", candidate.member + "~verdict", delta=gain,
                 decision="controls-pass" if out["controls_pass"] else "controls-fail", extra=out)
        return out


def sessions_of(tier: str, regime: str) -> list[str]:
    return sorted({session for pair in splits.pairs(tier, regime) for session in pair})


def estimate_feature_bytes(plan: dict[tuple[str, str], tuple[str, ...]], windows: int = ASSUMED_WINDOWS,
                           channels: int = ASSUMED_CHANNELS) -> float:
    total = 0.0
    for blocks in plan.values():
        for block in blocks:
            total += windows * (3000 * 4 if block == "B4" else channels * features.BLOCK_WIDTH[block] * 2)
    return total


def weight_tag(weight: float) -> str:
    return f"#w{weight:g}"


# --------------------------------------------------------------------------- phases

def phase_c0(run: Run, regime: str) -> None:
    """Ridge 6-band parity on the official protocol, once, declared defaults, in-window features."""
    from common import lock
    arm = base_arm()
    run.extract(regime, {("window", "bipolar"): ("base6",)})
    run.fit(regime, [arm])
    if "score" not in run.opts.stages:
        return
    os.environ["NPEXP_TEST_PURPOSE"], os.environ["NPEXP_TEST_MEMBER"] = "reproduction", arm.member
    lock.require_test_unlocked("T", regime, run.root, purpose="reproduction", member=arm.member,
                               config_hash=arm.hash())
    score = run.macro(arm.member, regime)
    gap = score - PUBLISHED_RIDGE6[regime]
    tasks = task_table(run, arm.member, regime)
    decision = "parity" if abs(gap) <= PARITY_TOLERANCE else "gap"
    run.log(regime, "member", arm.member, config_hash=arm.hash(), score=score, baseline="published:PR20",
            baseline_score=PUBLISHED_RIDGE6[regime], delta=gap, decision=decision,
            extra={"purpose": "reproduction", "tolerance": PARITY_TOLERANCE, "defaults": features.DEFAULTS,
                   "select_macro": run.macro(arm.member, regime, part="select"), "per_task": tasks})
    run.save(f"c0_{regime}.json", {"regime": regime, "member": arm.member, "config_hash": arm.hash(), "macro": score,
                                   "published": PUBLISHED_RIDGE6[regime], "gap": gap, "decision": decision,
                                   "per_task": tasks, "defaults": features.DEFAULTS})
    print(f"[c0] {regime}: ridge6 {score:.4f}, published {PUBLISHED_RIDGE6[regime]:.3f}, gap {gap:+.4f} ({decision})")
    for task, value in tasks.items():
        print(f"[c0] {regime}   {task:18s} {value:.4f}", flush=True)


def task_table(run: Run, member: str, regime: str) -> dict[str, float]:
    """Per-task AUROC: folds averaged within a session, then sessions averaged."""
    from common import stats
    cells = stats.member_scores(member, run.tier, regime, run.root, splits=run.universe)
    by_session: dict[tuple[str, str], list[float]] = {}
    for (session, task, _fold), value in cells.items():
        by_session.setdefault((task, session), []).append(value)
    return {task: float(np.nanmean([np.nanmean(v) for (t, _), v in by_session.items() if t == task]))
            for task in splits.TASKS if any(t == task for t, _ in by_session)}


def phase_p0(run: Run, regime: str) -> None:
    """ridge6 on every Tier P cell: the C1 baseline and the ensemble's Ridge 6-band member there."""
    arm = base_arm()
    run.extract(regime, {("window", "bipolar"): ("base6",)})
    run.fit(regime, [arm])
    if "score" in run.opts.stages:
        scores = run.report(regime, [arm])
        print(f"[p0] {regime}: ridge6 Tier P macro {scores[arm.member]:.4f}", flush=True)


def c1_plan() -> dict[tuple[str, str], tuple[str, ...]]:
    return {("window", "bipolar"): features.BLOCKS + features.SURROGATE_BLOCKS,
            ("session", "bipolar"): features.BLOCKS,
            ("window", "laplacian"): ("base6",), ("window", "shaft_mean"): ("base6",)}


def _block_arm(blocks: tuple[str, ...], weight: float, mode: str = "window", tag: str = "", **fields) -> Arm:
    member = "classical_" + "+".join(("base6",) + blocks) + tag + variant(mode, "bipolar")
    return stacked(member, (("c", "base6", mode, "bipolar"),), blocks, weight, mode=mode,
                   versus="ridge6" + variant(mode, "bipolar"), **fields)


def phase_c1(run: Run, regime: str) -> None:
    """Block screen on Tier P without MAPA: one block at a time on the Ridge 6-band base."""
    from common import stats
    run.extract(regime, c1_plan())
    base = [base_arm(), base_arm(reference="laplacian", oof=False), base_arm(reference="shaft_mean", oof=False),
            base_arm(mode="session", oof=False)]
    copied = run.load("c1_within.json") if regime == "cross_session" else None
    weights: dict[str, float] = {}
    if copied is None:
        grids = {block: {w: _block_arm((block,), w, tag=weight_tag(w), role="screen") for w in WEIGHT_GRID}
                 for block in ADDED_BLOCKS}
        grids = {block: grid for block, grid in grids.items() if run.available(regime, list(grid.values()))}
        run.fit(regime, base + [arm for grid in grids.values() for arm in grid.values()])
        weights = {block: run.choose_weight(regime, grid) for block, grid in grids.items()}
    else:
        run.fit(regime, base)
        weights = {block: float(w) for block, w in copied["block_weights"].items()
                   if run.available(regime, [_block_arm((block,), float(w))])}

    finals = {block: _block_arm((block,), weight, role="candidate") for block, weight in weights.items()}
    arms = list(finals.values())
    for block, arm in finals.items():
        arms += control_arms(arm, 1) + [_block_arm((block,), weights[block], mode="session", role="measurement")]
    run.fit(regime, arms)
    arms = run.available(regime, arms)
    scores = run.report(regime, base + arms)
    gains = {block: scores[arm.member] - scores["ridge6"] for block, arm in finals.items() if arm.member in scores}
    verdicts = {block: run.verdict(regime, finals[block], [a for a in arms if a.member.startswith(finals[block].member + "~")], scores)
                for block in gains}
    kept = tuple(copied["kept_blocks"]) if copied is not None else tuple(b for b in ADDED_BLOCKS if gains.get(b, 0.0) >= KEEP_GAIN)
    kept = tuple(b for b in kept if b in finals and finals[b].member in scores)

    body = {"regime": regime, "block_weights": weights, "block_gains": gains, "block_controls": verdicts,
            "kept_blocks": list(kept), "keep_rule": f"gain >= {KEEP_GAIN}",
            "copied_from_within": copied is not None, "reference_factor": {
                name: scores[name] for name in ("ridge6", "ridge6@laplacian", "ridge6@shaft_mean", "ridge6@session")}}
    if not kept:
        run.log(regime, "comparison", "classical_base6", baseline="ridge6", decision="Stop",
                extra={"why": "no block adds +0.002 over the base; classical-only claim invalidated", "gains": gains})
        body.update(kept_weight=None, classical_member=None, decision="Stop", controls=None)
    else:
        if len(kept) == 1:
            candidate, weight = finals[kept[0]], weights[kept[0]]
            shadows = [a for a in arms if a.member.startswith(candidate.member + "~")]
        else:
            if copied is None:
                grid = {w: _block_arm(kept, w, tag=weight_tag(w), role="screen") for w in WEIGHT_GRID}
                run.fit(regime, list(grid.values()))
                weight = run.choose_weight(regime, grid)
            else:
                weight = float(copied["kept_weight"])
            candidate = _block_arm(kept, weight, role="candidate", oof=True)
            shadows = control_arms(candidate, 1)
            extra = shadows + [candidate, _block_arm(kept, weight, mode="session", role="measurement")]
            run.fit(regime, extra)
            scores.update(run.report(regime, extra))
        comparison = stats.compare(candidate.member, "ridge6", "P", regime, run.root, splits=run.universe)
        run.log(**_comparison_row(comparison, regime, config_hash=candidate.hash()))
        body.update(kept_weight=weight, classical_member=candidate.member, decision=comparison.decision,
                    delta=comparison.delta, controls=run.verdict(regime, candidate, shadows, scores),
                    arm=candidate.config())
    stack = ("base6",) + kept
    body.update(blockset="+".join(stack), member="mapa_plus_classical_" + "+".join(stack))
    run.save(f"c1_{regime}.json", body)
    run.save(f"kept_{regime}.json", {k: body[k] for k in ("blockset", "member", "kept_blocks", "kept_weight", "decision")})
    print(f"[c1] {regime}: kept {list(kept)} decision {body['decision']}", flush=True)


def _comparison_row(comparison, regime: str, **fields) -> dict:
    row = comparison.row("x", **fields)
    for key in ("tier", "phase", "regime"):
        row.pop(key, None)
    return dict(row, regime=regime)


def c2_design(run: Run, regime: str) -> tuple[tuple[str, ...], str]:
    source = "within" if regime == "cross_session" else regime
    result = run.load(f"c1_{source}.json")
    decision, blocks = result.get("decision"), result.get("kept_blocks")
    passed = (result.get("controls") or {}).get("controls_pass")
    if decision not in ("Stop", "Hold", "Go") or not isinstance(blocks, list) or (
            (blocks or decision != "Stop") and not isinstance(passed, bool)):
        raise Missing(f"completed C1 decision and controls for {source}")
    keep = decision == "Go" or (decision == "Hold" and passed)
    added = ("base6",) + (tuple(blocks) if keep else ())
    return added, "mapa_plus_classical_" + "+".join(added)


def phase_c2(run: Run, regime: str) -> None:
    """Kept blocks (with the base) as extra kernels beside MAPA's last layer and frontend, Tier P."""
    from common import lock, stats
    added, member = c2_design(run, regime)
    surrogates = tuple(f"{b}~phaserand" for b in added if b in controls.PHASE_BLOCKS)
    run.extract(regime, {("window", "bipolar"): added + surrogates, ("session", "bipolar"): added})
    reference = Arm("mapa_plus_classical_none", MAPA_BASE, (1.0, 1.0), role="reference")

    def stack(weight: float, mode: str = "window", tag: str = "", **fields) -> Arm:
        return stacked(member + tag + variant(mode, "bipolar"), MAPA_BASE, added, weight, mode=mode,
                       versus=reference.member, **fields)

    if regime == "cross_session":
        weight = float(run.load("c2_within.json")["weight"])
        run.fit(regime, [reference])
    else:
        grid = {w: stack(w, tag=weight_tag(w), role="screen") for w in WEIGHT_GRID}
        run.fit(regime, [reference] + list(grid.values()))
        weight = run.choose_weight(regime, grid)
    candidate = stack(weight, role="candidate", oof=True)
    shadows = control_arms(candidate, len(MAPA_BASE))
    arms = [candidate] + shadows + [stack(weight, mode="session", role="measurement")]
    run.fit(regime, arms)
    scores = run.report(regime, [reference] + arms)
    controls_verdict = run.verdict(regime, candidate, shadows, scores)

    baseline = stats.best_baseline("P", regime, run.root, members=BASELINES, splits=run.universe)
    comparison = stats.compare(candidate.member, baseline, "P", regime, run.root, splits=run.universe)
    run.log(**_comparison_row(comparison, regime, config_hash=candidate.hash()))
    side = {"vs_mapa_stack": stats.compare(candidate.member, reference.member, "P", regime, run.root,
                                           splits=run.universe, descriptive=True)}
    if "btbank5_0" in {c.target_session for c in run.universe.list_cells("P", regime)}:
        # 5:0 is the only Tier P session MAPA was not pretrained on.
        side["session_5_0"] = stats.compare(candidate.member, baseline, "P", regime, run.root, splits=run.universe,
                                            sessions=["btbank5_0"], descriptive=True)
    for name, extra in side.items():
        run.log(**_comparison_row(extra, regime, config_hash=candidate.hash(), kind="info", extra={"analysis": name}))
    spec = dict(candidate.config(), added_blocks=list(added), weight=weight, weight_grid=list(WEIGHT_GRID),
                window_mode="window", reference="bipolar", defaults=features.DEFAULTS)
    proposal = lock.propose(regime, "classical", candidate.member, spec, comparison.delta, comparison.decision,
                            controls_verdict["controls_pass"], run.root)
    run.save(f"c2_{regime}.json", {"regime": regime, "member": candidate.member, "weight": weight, "baseline": baseline,
                                   "delta": comparison.delta, "decision": comparison.decision,
                                   "controls": controls_verdict, "arm": candidate.config(),
                                   "side": {k: {"delta": v.delta, "n_subjects": v.n_subjects} for k, v in side.items()},
                                   "proposal": str(proposal)})
    print(f"[c2] {regime}: {candidate.member} weight {weight:g} vs {baseline}: {comparison.delta:+.4f} "
          f"{comparison.decision}; controls_pass={controls_verdict['controls_pass']}", flush=True)


def phase_c3(run: Run, regime: str) -> None:
    """Controls report: refit any missing control arm of the C1 and C2 candidates and restate the verdicts."""
    from common import lock
    report = {}
    for source, n_base in (("c1", 1), ("c2", len(MAPA_BASE))):
        try:
            saved = run.load(f"{source}_{regime}.json")
        except Missing:
            continue
        if not saved.get("arm"):
            continue
        run.opts.rp_source = "base6" if source == "c1" else "mapa0"
        versus = "ridge6" if source == "c1" else "mapa_plus_classical_none"
        candidate = arm_from(saved["arm"], versus=versus, role="candidate")
        shadows = control_arms(candidate, n_base)
        run.fit(regime, shadows)
        shadows = run.available(regime, shadows)
        scores = run.report(regime, shadows + [candidate])
        report[source] = run.verdict(regime, candidate, shadows, scores)
        measured = candidate.member + "@session"
        try:
            report[source]["window_discipline"] = {
                "in_window_gain": scores[candidate.member] - scores[versus],
                "session_filtered_gain": run.macro(measured, regime) - run.macro(versus + "@session" if source == "c1" else versus, regime)}
        except FileNotFoundError:
            report[source]["window_discipline"] = None
        if source == "c2":
            try:
                spec = json.loads(Path(saved["proposal"]).read_text())["spec"]
                lock.propose(regime, "classical", candidate.member, spec, saved["delta"], saved["decision"],
                             report[source]["controls_pass"], run.root)
            except lock.FreezeError as error:
                print(f"[c3] {regime}: proposal not rewritten ({error})", flush=True)
    if not report:
        raise Missing(f"{run.out}/c1_{regime}.json or c2_{regime}.json with a candidate arm")
    run.save(f"controls_{regime}.json", report)
    print(f"[c3] {regime}: " + ", ".join(f"{k} controls_pass={v['controls_pass']}" for k, v in report.items()), flush=True)


def frozen_design(run: Run, regime: str) -> tuple[dict, Arm | None]:
    """The frozen candidate and only the classical stack actually used by it."""
    from common import lock
    try:
        design = lock.frozen(regime, run.root)
    except lock.FreezeError as error:
        raise Missing(str(error)) from None
    if design["track"] == "classical":
        spec = design["spec"]
    else:
        members = design["spec"]["members"]
        used = [p for p in lock.proposals(regime, run.root)
                if p["track"] == "classical" and p["member"] in members]
        if not used:
            return design, None
        spec = used[0]["spec"]
    return design, arm_from(spec, oof=True, role="candidate")


def _confirm(run: Run, regime: str, tier: str) -> None:
    from common import ledger, lock, stats
    design, stack = frozen_design(run, regime)
    is_candidate = design["track"] == "classical"
    candidate_id = lock.candidate_id(design)
    if is_candidate:
        ledger.guard_candidate(run.root, tier, regime, candidate_id)
    arms = [base_arm()] + ([stack] if stack is not None else [])
    blocks = tuple(dict.fromkeys(k[1] for arm in arms for k in arm.kernels if k[0] == "c"))
    run.extract(regime, {("window", "bipolar"): blocks})
    run.fit(regime, arms)
    if "score" not in run.opts.stages:
        return
    if not is_candidate:
        print(f"[{run.phase}] {regime}: frozen candidate is {candidate_id}; classical members exported, "
              "no labels read, no comparison", flush=True)
        for arm in arms:
            run.log(regime, "member", arm.member, config_hash=arm.hash(), extra={"exported_for": candidate_id})
        return
    if tier == "T":
        lock.require_test_unlocked("T", regime, run.root, purpose="score", config_hash=design["config_hash"])
    baseline = stats.best_baseline(tier, regime, run.root, members=BASELINES, splits=run.universe)
    comparison = stats.compare(stack.member, baseline, tier, regime, run.root, splits=run.universe)
    run.log(**_comparison_row(comparison, regime, config_hash=design["config_hash"], candidate=candidate_id))
    run.save(f"{run.phase}_{regime}.json", {"regime": regime, "candidate": candidate_id, "member": stack.member,
                                            "baseline": baseline, "delta": comparison.delta,
                                            "decision": comparison.decision, "score": comparison.score})
    print(f"[{run.phase}] {regime}: {stack.member} vs {baseline}: {comparison.delta:+.4f} {comparison.decision}", flush=True)


def phase_c4(run: Run, regime: str) -> None:
    """Tier V veto check of the frozen candidate (one pass); otherwise export members for the ensemble."""
    _confirm(run, regime, "V")


def phase_c5(run: Run, regime: str) -> None:
    """Tier T, once, through the lock."""
    _confirm(run, regime, "T")


PHASES = {"c0": ("T", phase_c0), "c1": ("P", phase_c1), "c2": ("P", phase_c2), "c3": ("P", phase_c3),
          "c4": ("V", phase_c4), "c5": ("T", phase_c5), "p0": ("P", phase_p0)}


def recording_path(session: str) -> Path:
    subject, trial = splits.parse_session(session)
    return common.data_root() / f"sub_{subject}_trial{trial:03d}.h5"


def cache_sessions(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        out += {"eval": list(splits.EVAL_SESSIONS), "dev": list(splits.DEV_SESSIONS)}.get(name, [name])
    unknown = [s for s in out if s not in splits.EVAL_SESSIONS + splits.DEV_SESSIONS]
    if unknown:
        raise SystemExit(f"not a benchmark or Tier P session: {unknown}")
    return out


def cache_plan(session: str, which: str = "full") -> dict[tuple[str, str], tuple[str, ...]]:
    """Evaluation sessions: every block, in-window only. Tier P sessions: everything C1-C3 read,
    or with ``which="base"`` the in-window bipolar blocks alone (enough for ridge6 and C2's kernels)."""
    if session in splits.DEV_SESSIONS and which == "full":
        return c1_plan()
    return {("window", "bipolar"): features.BLOCKS}


def session_cost(session: str, plan: dict, exact: bool) -> tuple[float, float, int, int]:
    """Estimated (core-hours, bytes, windows, channels) to extract the uncached part of ``plan``."""
    windows, channels = ASSUMED_WINDOWS, ASSUMED_CHANNELS
    if exact:
        windows = len(features.session_onsets(session)[0])
        recording = features.recording(session)
        channels = len(features.channels(recording.labels, recording.regions, "bipolar"))
    cpu_s = size = 0.0
    for (mode, reference), blocks in plan.items():
        meta = features.store_dir(session, mode, reference) / "meta.json"
        have = set(json.loads(meta.read_text())["blocks"]) if meta.exists() else set()
        lacking = tuple(b for b in blocks if b not in have)
        if not lacking:
            continue
        size += estimate_feature_bytes({(mode, reference): lacking}, windows, channels)
        if mode == "window":
            passes = 2 if any("~" in b for b in lacking) else 1
            tiles = 1750
            cpu_s += channels * EXTRACT_CPU_S_PER_CHANNEL_KWINDOW * (windows * passes + tiles) / 1000
        else:
            cpu_s += channels * EXTRACT_CPU_S_PER_CHANNEL_SESSION
    return cpu_s / 3600, size, windows, channels


def phase_cache(opts: argparse.Namespace) -> int:
    """Extract and cache classical features per session (CPU only; no labels are read)."""
    from common import ledger
    sessions = cache_sessions(opts.sessions)
    missing, total_h, total_bytes = [], 0.0, 0.0
    print(f"== cache {'dry run' if opts.dry_run else 'run'}: {len(sessions)} sessions, commit {ledger.commit()}, "
          f"defaults {features.defaults_hash()}, root {common.root()}")
    for session in sessions:
        subject, trial = splits.parse_session(session)
        recording = common.data_root() / f"sub_{subject}_trial{trial:03d}.h5"
        plan = cache_plan(session, opts.plan)
        if not recording.exists():
            missing.append(f"recording {recording}")
            print(f"   {session}: recording absent")
            continue
        hours, size, windows, channels = session_cost(session, plan, opts.exact)
        total_h, total_bytes = total_h + hours, total_bytes + size
        print(f"   {session}: {'~' if not opts.exact else ''}{windows} windows x {'~' if not opts.exact else ''}{channels} "
              f"bipolar channels; to extract {hours:.2f} core-h, {size / GIB:.2f} GiB; "
              f"stores {[f'{m}/{r}' for m, r in plan]}")
    free = features.free_gib(features.features_root())
    workers = max(opts.workers, 1)
    print(f"== estimates: {total_h:.1f} core-hours, about {total_h / workers * 1.25 + 0.03 * len(sessions):.1f} h wall with "
          f"{workers} workers; cache +{total_bytes / GIB:.1f} GiB under {features.features_root()} ({free:.0f} GiB free, floor "
          f"{opts.min_free_gib:.0f}); RAM about {2.5 * workers:.0f} GiB in-window, {4.5 * workers:.0f} GiB while a "
          f"session-filtered store is extracted; VRAM 0")
    if free - total_bytes / GIB < opts.min_free_gib:
        missing.append(f"disk: {free:.0f} GiB free minus {total_bytes / GIB:.1f} GiB is below the floor")
    if opts.dry_run:
        if missing:
            print("== MISSING INPUTS (these sessions would be skipped or the launcher would stop):")
            for item in missing:
                print(f"   - {item}")
            return 2
        print("== dry run ok: every input is present")
        return 0
    failed = 0
    pending, deadline = list(sessions), time.time() + 60 * opts.wait_minutes
    while pending:
        ready = [s for s in pending if recording_path(s).exists()]
        if not ready:
            if time.time() >= deadline:
                break
            time.sleep(60)   # recordings still downloading; the downloader moves each into place atomically
            continue
        session = ready[0]
        pending.remove(session)
        tier = "P" if session in splits.DEV_SESSIONS else "VT"
        free = features.free_gib(features.features_root())
        need = estimate_feature_bytes(cache_plan(session, opts.plan)) / GIB
        if free - need < opts.min_free_gib:
            raise SystemExit(f"refusing to extract {session}: {free:.0f} GiB free under {features.features_root()}, "
                             f"floor {opts.min_free_gib:.0f} GiB")
        started = time.time()
        try:
            folders = features.extract(session, cache_plan(session, opts.plan), workers=opts.workers)
        except MissingInput as error:
            failed += 1
            ledger.append({"tier": tier, "regime": "all", "phase": "cache", "kind": "info", "arm": session,
                           "decision": "missing-input", "extra": {"why": str(error)}}, common.results_root())
            print(f"[cache] {session}: skipped, {error}", flush=True)
            continue
        seconds = time.time() - started
        sizes = {f"{m}/{r}": sum(f.stat().st_size for f in folder.glob("*.npy")) for (m, r), folder in folders.items()}
        ledger.append({"tier": tier, "regime": "all", "phase": "cache", "kind": "info", "arm": session,
                       "config_hash": features.defaults_hash(), "decision": "cached",
                       "extra": {"seconds": round(seconds, 1), "workers": opts.workers, "bytes": sizes}},
                      common.results_root())
        print(f"[cache] {session}: {seconds:.0f} s, {sum(sizes.values()) / GIB:.2f} GiB", flush=True)
    for session in pending:   # recordings that never arrived within --wait-minutes
        failed += 1
        ledger.append({"tier": "P" if session in splits.DEV_SESSIONS else "VT", "regime": "all", "phase": "cache",
                       "kind": "info", "arm": session, "decision": "missing-input",
                       "extra": {"why": f"recording absent: {recording_path(session)}"}}, common.results_root())
        print(f"[cache] {session}: skipped, recording absent", flush=True)
    return 2 if failed else 0


# --------------------------------------------------------------------------- dry run

def first_pass(run: Run, regime: str) -> tuple[dict, list[Arm], list[str]]:
    """Extraction plan, first-pass arms and other required inputs of a phase, for --dry-run."""
    phase, needs = run.phase, []
    if phase == "p0":
        return {("window", "bipolar"): ("base6",)}, [base_arm()], []
    if phase == "c0":
        return {("window", "bipolar"): ("base6",)}, [base_arm()], []
    if phase == "c1":
        arms = [base_arm(), base_arm(reference="laplacian", oof=False), base_arm(reference="shaft_mean", oof=False),
                base_arm(mode="session", oof=False)]
        arms += [_block_arm((b,), w, tag=weight_tag(w), role="screen") for b in ADDED_BLOCKS for w in WEIGHT_GRID]
        if regime == "cross_session" and "within" not in run.opts.regimes:   # otherwise this run writes it first
            needs.append(str(run.out / "c1_within.json"))
        return c1_plan(), arms, needs
    if phase in ("c2", "c3"):
        try:
            added, member = c2_design(run, regime)
        except Missing as error:
            return {}, [], [str(error)]
        arms = [Arm("mapa_plus_classical_none", MAPA_BASE, (1.0, 1.0), role="reference")]
        arms += [stacked(member + weight_tag(w), MAPA_BASE, added, w, role="screen") for w in WEIGHT_GRID]
        needs += [f"preds:{m}" for m in BASELINES]
        return {("window", "bipolar"): added, ("session", "bipolar"): added}, arms, needs
    try:
        design, stack = frozen_design(run, regime)
    except Missing as error:
        return {}, [base_arm()], [str(error)]
    arms = [base_arm()] + ([stack] if stack is not None else [])
    blocks = tuple(dict.fromkeys(k[1] for arm in arms for k in arm.kernels if k[0] == "c"))
    if design["track"] == "classical":
        needs.append("preds:mapa12")
    return {("window", "bipolar"): blocks}, arms, needs


def dry_run(run: Run, regimes: list[str]) -> int:
    from common import ledger, lock, mapa_features, predictions
    missing: list[str] = []
    tier = run.tier
    print(f"== {run.phase} dry run: tier {tier}, regimes {regimes}, commit {ledger.commit()}, "
          f"defaults {features.defaults_hash()}, root {common.root()}")
    total_cpu_h = total_disk = fit_cpu_h = 0.0
    seen_sessions: set[str] = set()
    for regime in regimes:
        cells = splits.list_cells(tier, regime)
        plan, arms, needs = first_pass(run, regime)
        sessions = sessions_of(tier, regime)
        print(f"-- {regime}: {len(cells)} cells ({len({c.target_session for c in cells})} target sessions x {len(splits.TASKS)} tasks"
              f"{' x 2 folds' if regime == 'within' else ''}); first {[c.key() for c in cells[:2]]}")
        print(f"   arms, first pass ({len(arms)}): {[a.member for a in arms]}")
        if run.phase in ("c1", "c2"):
            print(f"   then per kept arm: chosen-weight arm (out-of-fold), ~sham, ~rp, ~phaserand (B1/B2), @session; "
                  f"weights {WEIGHT_GRID} chosen on select macro")
        for session in sessions:
            subject, trial = splits.parse_session(session)
            recording = common.data_root() / f"sub_{subject}_trial{trial:03d}.h5"
            if not recording.exists():
                missing.append(f"recording {recording}")
        to_extract = 0
        for session in sessions:
            for (mode, reference), blocks in plan.items():
                meta = features.store_dir(session, mode, reference) / "meta.json"
                have = set(json.loads(meta.read_text())["blocks"]) if meta.exists() else set()
                lacking = [b for b in blocks if b not in have]
                if lacking and session not in seen_sessions:
                    to_extract += 1
                    total_disk += estimate_feature_bytes({(mode, reference): tuple(lacking)})
                    total_cpu_h += ASSUMED_CHANNELS * (
                        EXTRACT_CPU_S_PER_CHANNEL_KWINDOW * ASSUMED_WINDOWS / 1000 * (2 if any("~" in b for b in lacking) else 1)
                        if mode == "window" else EXTRACT_CPU_S_PER_CHANNEL_SESSION) / 3600
        seen_sessions.update(sessions)
        print(f"   features: {to_extract} (session, mode, reference) stores to extract")
        taps = sorted({int(k[1]) for a in arms for k in a.kernels if k[0] == "mapa"}
                      | ({0} if run.opts.rp_source == "mapa0" and run.phase in ("c1", "c2", "c3") else set()))
        for tap in taps:
            try:
                files = [mapa_features.kernel_file(mapa_features.STORE_OF_TIER[tier], regime, c.key(), tap) for c in cells]
            except mapa_features.KernelStoreUnavailable as error:
                missing.append(f"MAPA kernel store, tap {tap}, {regime}: {error}")
                continue
            absent = [f for f in files if not f.is_file()]
            if absent:
                missing.append(f"MAPA kernels tap {tap}, {regime}: {len(absent)} of {len(cells)} cells absent, e.g. {absent[0]}")
        for need in needs:
            if need.startswith("preds:"):
                member = need.split(":", 1)[1]
                absent = [c for c in cells if not predictions.exists(c, member, run.root)]
                if absent:
                    producer = "classical C1" if member == "ridge6" else "ensemble export"
                    missing.append(f"predictions of member {member}, tier {tier} {regime}: {len(absent)} of "
                                   f"{len(cells)} cells absent under {run.root}/preds ({producer})")
            elif not Path(need).exists():
                missing.append(need)
        n_arms = len(arms) + (len(ADDED_BLOCKS) * 5 + 8 if run.phase == "c1" else 5 if run.phase == "c2" else 0)
        fit_cpu_h += len(cells) * n_arms * FIT_CPU_S[regime] * 4 / 3600
    free = features.free_gib(features.features_root())
    workers = max(run.opts.workers, 1)
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    wall_h = total_cpu_h / workers * 1.25 + fit_cpu_h / (run.opts.fit_workers * threads)
    print(f"== estimates: extraction {total_cpu_h:.1f} core-h, fitting {fit_cpu_h:.1f} core-h, about {wall_h:.1f} h wall "
          f"with {workers} extraction workers and {run.opts.fit_workers} fit workers x {threads} threads; "
          f"feature cache +{total_disk / GIB:.1f} GiB under {features.features_root()} ({free:.0f} GiB free, floor {run.opts.min_free_gib:.0f}); "
          f"RAM about {max(2.6 * workers, 2.5 * run.opts.fit_workers):.0f} GiB peak; "
          f"VRAM {0 if os.environ.get('NPEXP_GRAM_DEVICE', 'cpu') == 'cpu' else 2 * run.opts.fit_workers} GiB")
    if free - total_disk / GIB < run.opts.min_free_gib:
        missing.append(f"disk: {free:.0f} GiB free minus {total_disk / GIB:.1f} GiB falls below the "
                       f"{run.opts.min_free_gib:.0f} GiB floor")
    if missing:
        print("== MISSING INPUTS (the launcher would stop):")
        for item in dict.fromkeys(missing):
            print(f"   - {item}")
        return 2
    print("== dry run ok: every input is present")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(PHASES) + ["cache"])
    parser.add_argument("--sessions", nargs="+", default=["eval"],
                        help="cache phase: 'eval', 'dev' and/or session names such as btbank3_0")
    parser.add_argument("--exact", action="store_true",
                        help="cache dry run: count windows and channels from the benchmark instead of assuming them")
    parser.add_argument("--plan", choices=["full", "base"], default="full",
                        help="cache phase, Tier P sessions: 'base' caches only the in-window bipolar blocks")
    parser.add_argument("--wait-minutes", type=float, default=0.0,
                        help="cache phase: keep waiting this long for recordings that are still downloading")
    parser.add_argument("--regimes", nargs="+", default=list(REGIME_ORDER), choices=splits.REGIMES)
    parser.add_argument("--stages", nargs="+", default=["extract", "fit", "score"], choices=["extract", "fit", "score"])
    parser.add_argument("--workers", type=int, default=16, help="feature-extraction processes (one thread each)")
    parser.add_argument("--fit-workers", type=int, default=8, help="processes fitting cells")
    parser.add_argument("--rp-source", choices=["mapa0", "base6"], default=None,
                        help="features behind the random-projection control; default base6 for c1 (a declared "
                             "deviation: MAPA frontend kernels do not exist yet when C1 runs), mapa0 otherwise")
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args(argv)
    if opts.phase == "cache":
        return phase_cache(opts)
    tier, function = PHASES[opts.phase]
    if opts.rp_source is None:
        opts.rp_source = "base6" if opts.phase == "c1" else "mapa0"
    run = Run(opts.phase, tier, opts)
    regimes = [r for r in REGIME_ORDER if r in opts.regimes]
    if opts.dry_run:
        return dry_run(run, regimes)
    from common import ledger, lock, mapa_features
    guards = (PermissionError, ledger.LedgerError, lock.FreezeError)
    for regime in regimes:
        run.log(regime, "info", f"{opts.phase}-start", extra={"argv": sys.argv[1:], "stages": opts.stages})
        try:
            function(run, regime)
        except (Missing, MissingInput, FileNotFoundError, mapa_features.KernelStoreUnavailable) as error:
            run.log(regime, "info", f"{opts.phase}-stopped", decision="missing-input", extra={"why": str(error)})
            print(f"[{opts.phase}] {regime}: stopped, missing input: {error}", file=sys.stderr, flush=True)
            return 2
        except guards as error:
            run.log(regime, "info", f"{opts.phase}-stopped", decision="refused", extra={"why": str(error)})
            print(f"[{opts.phase}] {regime}: refused: {error}", file=sys.stderr, flush=True)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

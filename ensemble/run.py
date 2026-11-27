"""Ensemble track phases E0-E5 (see ``06-classical-and-ensemble-plan.md``).

    python -m ensemble.run export  --tier P --regime within --members mapa12 mapa0 [--gpu-shard i/n]
    python -m ensemble.run e0      --tier P --regime within --wave 1
    python -m ensemble.run e1|e2|e3 --regime within --wave 1
    python -m ensemble.run freeze  --regime within
    python -m ensemble.run e4|e5   --regime within
    python -m ensemble.run parity     --regime within          (no label read)
    python -m ensemble.run reproduce  --regime within --member mapa12   (test labels, logged)

Every subcommand takes ``--dry-run``: it checks that its inputs exist, lists what it
would do, prints time, disk and memory estimates and starts nothing.
Design decisions (members, rung) use Tier P only. Tier V vetoes one frozen candidate.
Tier T is scored once behind the lock. Every run appends to the ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from common import confirm_fallback, ledger, lock, predictions, stats
from ensemble import controls, fuse, paths
from ensemble.members import trained

WAVE_MEMBERS = {1: ("mapa12", "mapa0", "ridge6"), 2: ("mapa12", "mapa0", "ridge6", "diver", "cnn")}
RIDGE_MEMBERS = ("mapa12", "mapa0")
TRAINED_MEMBERS = ("diver", "cnn")
BASELINES = ("mapa12", "ridge6")
E0_TOLERANCE = 1e-6
E1_MINIMUM = 0.005
E2_MINIMUM = 0.003
MAPA_HELD_OUT_SESSION = "btbank5_0"     # the only Tier P session MAPA was not pretrained on
PUBLISHED = {   # leaderboard macro AUROC: within, cross-session, cross-subject
    "mapa12": {"within": 0.6953, "cross_session": 0.6909, "cross_subject": 0.6083},
    "mapa0": {"within": 0.6744, "cross_session": 0.6660, "cross_subject": 0.5872},
    "diver": {"within": 0.678},
    "cnn": {"within": 0.669, "cross_session": 0.670, "cross_subject": 0.578},
}


class Refused(SystemExit):
    """A phase declined to run; the message says exactly why."""

    def __init__(self, message: str, code: int = 3):
        print(f"REFUSED: {message}", file=sys.stderr)
        super().__init__(code)


def _splits():
    from common import splits
    return splits


def state_dir(root: Path) -> Path:
    return Path(root) / "ensemble"


def load_state(root: Path, name: str) -> dict:
    file = state_dir(root) / f"{name}.json"
    return json.loads(file.read_text()) if file.is_file() else {}


def save_state(root: Path, name: str, body: dict) -> Path:
    file = state_dir(root) / f"{name}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(temp, file)
    return file


def backend(member: str):
    if member == "diver":
        from ensemble.members.diver import Backend
    elif member == "cnn":
        from ensemble.members.cnn import Backend
    else:
        raise KeyError(member)
    return Backend()


def classical_member(regime: str, root: Path) -> str | None:
    """M6 is the surviving C2 stack, not the block set nominated by C1."""
    source = "within" if regime == "cross_session" else regime
    for proposal in lock.proposals(source, root):
        if proposal["track"] == "classical" and lock.eligible(proposal):
            return proposal["member"]
    return None


def missing_predictions(member: str, tier: str, regime: str, root: Path) -> list:
    return [c for c in _splits().list_cells(tier, regime) if not predictions.exists(c, member, root)]


# --------------------------------------------------------------------------- export

def export(args, root: Path) -> None:
    splits = _splits()
    cells = splits.list_cells(args.tier, args.regime)
    print(f"export: tier {args.tier}, regime {args.regime}, {len(cells)} cells, members {args.members}")
    problems = []
    for member in args.members:
        if member in RIDGE_MEMBERS:
            from common import mapa_features
            from ensemble.members import mapa
            todo = [c for c in cells if not predictions.exists(c, member, root)]
            absent = [c for c in todo if not mapa_features.kernel_file(
                mapa_features.STORE_OF_TIER[c.tier], c.regime, c.key(), mapa.TAP[member]).is_file()]
            rows = [len(splits.get_cell(c).rows["train"]) for c in todo]
            # one symmetric eigendecomposition plus five block solves per cell (unmeasured rate)
            hours = sum(30.0 * (n / 3500) ** 3 for n in rows) / 3600
            print(f"  {member}: {len(todo)} cells to fit on CPU, ~{hours:.1f} h at 4 threads (estimate), VRAM 0")
            if absent:
                first = mapa_features.kernel_file(mapa_features.STORE_OF_TIER[absent[0].tier], absent[0].regime,
                                                  absent[0].key(), mapa.TAP[member])
                problems.append(f"{len(absent)} kernels for {member}, first {first} (run launch/mapa_kernels.sh)")
            elif not args.dry_run:
                print(f"  {member}: wrote {mapa.export(member, args.tier, args.regime, root)} cells")
        elif member in TRAINED_MEMBERS:
            engine = backend(member)
            seeds, oof = ((0,), False) if args.quick else (trained.SEEDS, True)
            if args.seeds_only:
                seeds, oof = trained.SEEDS[1:], False
            elif args.oof_only:
                seeds, oof = trained.SEEDS[:1], True
            plan = trained.plan(engine, args.tier, args.regime, root, available_only=args.available_only,
                                seeds=seeds, oof=oof, reuse_full=args.oof_only)
            if plan.get("note"):
                print(f"  {member}: {plan['note']}; nothing to export")
                continue
            print(f"  {member}: {plan['cells']} cells, {plan['fits']} fits ("
                  + ("seed 0 select/eval only" if args.quick else "seeds 1-2 select/eval" if args.seeds_only
                     else "seed 0 out-of-fold blocks, reusing its full model" if args.oof_only
                     else "3 seeds + 5 out-of-fold blocks for seed 0")
                  + f"), ~{plan['hours']:.1f} h on one GPU, VRAM {engine.vram_mib} MiB, "
                  f"session pairs: {len(plan['pairs'])}")
            if plan["missing"]:
                problems += plan["missing"]
            elif not args.dry_run:
                shard = tuple(int(x) for x in args.gpu_shard.split("/"))
                done = trained.export(engine, args.tier, args.regime, root, shard=shard,
                                      available_only=args.available_only, seeds=seeds, oof=oof,
                                      reuse_full=args.oof_only)
                print(f"  {member}: wrote {done} cells")
        else:
            raise Refused(f"{member} is exported by the classical track, not here")
    if problems:
        for item in problems:
            print(f"MISSING INPUT: {item}")
        raise SystemExit(2)
    if args.dry_run:
        print("dry run: nothing was started")


# --------------------------------------------------------------------------- E0

def check_member(member: str, tier: str, regime: str, root: Path, *, need_train: bool = True) -> dict:
    """E0: the AUROC recomputed from the exported file equals the member's own reported
    AUROC (on the select part, which every tier may read) to 1e-6, for every cell."""
    splits = _splits()
    worst, bad, cells = 0.0, [], splits.list_cells(tier, regime)
    undefined_select = []
    for cell in cells:
        if not predictions.exists(cell, member, root):
            bad.append(f"{cell.key()}: no prediction file")
            continue
        pred = predictions.read(cell, member, root)
        split = splits.get_cell(cell)
        try:
            if not {"select", "eval"} <= pred.scores.keys():
                raise KeyError("missing select or eval scores")
            for part in pred.scores:
                if not np.isfinite(pred.scores[part]).all():
                    raise ValueError(f"nonfinite {part} scores")
                if part != "train" or "@boot" not in member:
                    predictions.align(pred.scores[part], pred.rows[part], split.rows[part], f"{member} {cell.key()} {part}")
            fallback = confirm_fallback.for_select(cell, member, split.labels("select"))
            if confirm_fallback.in_scope(cell, member) and member in ("mapa12", "diver"):
                if len(np.unique(pred.scores["select"])) < 2:
                    raise ValueError("select-CDF normalization needs at least two distinct scores")
            reported = pred.meta.get("select_auroc")
            recomputed = stats.cell_auroc(member, cell, root, part="select")
        except (predictions.RowSetMismatch, KeyError, ValueError) as error:
            bad.append(f"{cell.key()}: {error}")
            continue
        if need_train and "train" not in pred.scores:
            bad.append(f"{cell.key()}: no out-of-fold training scores")
        if pred.meta.get("confirm_fallback") != fallback:
            bad.append(f"{cell.key()}: missing or unexpected fallback provenance")
            continue
        if fallback is not None:
            trained_value = pred.meta.get("lam") if "lam" in fallback else pred.meta.get("epochs_run")
            if (trained_value != fallback.get("lam", fallback.get("epochs")) or
                    "select_auroc" not in pred.meta or reported is not None or not np.isnan(recomputed)):
                bad.append(f"{cell.key()}: wrong fitted parameter or undefined select AUROC metadata")
            else:
                undefined_select.append(cell.key())
            continue
        if reported is None or not np.isfinite(reported):
            bad.append(f"{cell.key()}: meta has no select_auroc")
            continue
        gap = abs(recomputed - reported)
        worst = max(worst, gap)
        if not np.isfinite(gap) or gap > E0_TOLERANCE:
            bad.append(f"{cell.key()}: recomputed {recomputed:.8f} vs reported {reported:.8f}")
    return {"member": member, "passed": not bad, "cells": len(cells), "max_abs_gap": worst, "problems": bad[:20],
            "n_problems": len(bad), "undefined_select_cells": undefined_select}


def e0(args, root: Path) -> None:
    members = list(args.members or WAVE_MEMBERS[args.wave])
    extra = classical_member(args.regime, root)
    if extra and not args.members:
        members.append(extra)
    state = load_state(root, f"e0_{args.tier}_{args.regime}")
    print(f"E0: tier {args.tier}, regime {args.regime}, members {members}; CPU only, minutes, VRAM 0")
    absent = {m: len(missing_predictions(m, args.tier, args.regime, root)) for m in members}
    for member, count in absent.items():
        if count:
            print(f"MISSING INPUT: {count} prediction files of {member} under "
                  f"{Path(root) / 'preds' / args.tier / args.regime / member}")
    if args.dry_run:
        print("dry run: nothing was started")
        raise SystemExit(2 if any(absent.values()) else 0)
    for member in members:
        report = check_member(member, args.tier, args.regime, root)
        extra_info = {"max_abs_gap": report["max_abs_gap"], "n_problems": report["n_problems"],
                      "problems": report["problems"][:5], "wave": args.wave,
                      "select_macro": None, "seed_spread": None}
        score = None
        if report["passed"]:
            extra_info["select_macro"] = stats.macro(member, args.tier, args.regime, root, part="select")
            if args.tier == "P":
                score = stats.macro(member, "P", args.regime, root)
            if member in TRAINED_MEMBERS:
                replicas = [trained.seed_member(member, s) for s in trained.SEEDS]
                part = "eval" if args.tier == "P" else "select"
                values = [stats.macro(m, args.tier, args.regime, root, part=part) for m in replicas
                          if not missing_predictions(m, args.tier, args.regime, root)]
                extra_info["seed_spread"] = {"part": part, "values": values, "min": min(values),
                                             "max": max(values), "std": float(np.std(values))}
                report["seeds_complete"] = len(values) == len(replicas)
        state[member] = report
        ledger.append({"tier": args.tier, "regime": args.regime, "phase": "E0", "kind": "member", "arm": member,
                       "score": score, "decision": "pass" if report["passed"] else "excluded",
                       "extra": extra_info}, root)
        print(f"  {member}: {'pass' if report['passed'] else 'EXCLUDED'} max gap {report['max_abs_gap']:.1e} "
              f"problems {report['n_problems']}" + (f" Tier P macro {score:.4f}" if score is not None else ""))
    save_state(root, f"e0_{args.tier}_{args.regime}", state)


def pool(regime: str, wave: int, root: Path, tier: str = "P") -> list[str]:
    """Members that passed E0 on a tier for this regime and belong to the wave."""
    state = load_state(root, f"e0_{tier}_{regime}")
    allowed = list(WAVE_MEMBERS[wave])
    extra = classical_member(regime, root)
    if extra:
        allowed.append(extra)
    return [m for m in allowed if state.get(m, {}).get("passed")]


# --------------------------------------------------------------------------- E1-E3 (Tier P)

def _baseline(members, regime: str, root: Path, tier: str = "P") -> str:
    have = [m for m in BASELINES if m in members]
    if not have:
        raise Refused(f"neither baseline of {BASELINES} passed E0 for {regime}")
    return stats.best_baseline(tier, regime, root, have)


def e1(args, root: Path) -> None:
    regime, wave = args.regime, args.wave
    members = pool(regime, wave, root)
    if args.members:     # a restriction of the E0 pool, recorded in the summary and the ledger
        members = [m for m in members if m in args.members]
    combos = fuse.subsets(members) if len(members) >= 2 else []
    print(f"E1 uniform screen: Tier P, regime {regime}, wave {wave}, pool {members}, {len(combos)} subsets; "
          f"CPU only, about {max(1, len(combos) // 4)} min, VRAM 0, disk < 1 GiB of fused files")
    if len(members) < 2:
        print(f"MISSING INPUT: fewer than two members passed E0 on Tier P for {regime} "
              f"({state_dir(root) / f'e0_P_{regime}.json'})")
        raise SystemExit(2)
    if args.dry_run:
        for combo in combos:
            print("  " + fuse.fused_name(combo, "a"))
        print("dry run: nothing was started")
        return
    baseline = _baseline(members, regime, root)
    single = {m: stats.macro(m, "P", regime, root) for m in members}
    results = []
    for combo in combos:
        name = fuse.fuse(combo, "a", "P", regime, root)
        best_member = max(combo, key=single.get)
        over_member = stats.compare(name, best_member, "P", regime, root, descriptive=True)
        over_baseline = stats.compare(name, baseline, "P", regime, root)
        ledger.append(over_member.row("E1", wave=wave, extra={"question": "gain over the subset's best member"}), root)
        ledger.append(over_baseline.row("E1", wave=wave, extra={"question": "threshold table against the regime baseline"}), root)
        results.append({"members": list(combo), "name": name, "score": over_baseline.score,
                        "best_member": best_member, "gain_over_best_member": over_member.delta,
                        "delta_vs_baseline": over_baseline.delta, "decision": over_baseline.decision})
        print(f"  {name}: {over_baseline.score:.4f} best member {best_member} {over_member.delta:+.4f} "
              f"baseline {baseline} {over_baseline.delta:+.4f} {over_baseline.decision}")
    best = max(results, key=lambda r: r["score"])
    passed = bool(best["gain_over_best_member"] >= E1_MINIMUM - 1e-12)
    summary = {"regime": regime, "wave": wave, "pool": members, "baseline": baseline, "single": single,
               "subsets": results, "best": best, "passed": passed, "threshold": E1_MINIMUM,
               "pool_restricted_to": args.members}
    # Controls that run alongside the screen.
    summary["leave_one_member_out"] = controls.leave_one_member_out(best["members"], "a", "P", regime, root)
    for member, info in summary["leave_one_member_out"].items():
        ledger.append({"tier": "P", "regime": regime, "phase": "E1", "kind": "control",
                       "arm": f"leave_out:{member}", "baseline": best["name"], "delta": info["contribution"],
                       "decision": "keep" if info["keep"] else "drop", "wave": wave}, root)
    summary["self_ensemble"] = _self_ensemble("P", regime, root, wave)
    summary["mapa_pretraining_split"] = _held_out_session_report(best["name"], baseline, regime, root, wave)
    ledger.append({"tier": "P", "regime": regime, "phase": "E1", "kind": "info", "arm": best["name"],
                   "baseline": best["best_member"], "delta": best["gain_over_best_member"],
                   "decision": "pass" if passed else "fail", "wave": wave,
                   "extra": {"rule": f"best subset gains >= {E1_MINIMUM} over its best member",
                             "subsets_screened": len(results)}}, root)
    save_state(root, f"e1_{regime}_w{wave}", summary)
    print(f"E1 {'PASS' if passed else 'FAIL'}: best subset {best['name']} gains "
          f"{best['gain_over_best_member']:+.4f} over {best['best_member']}")
    _wave2_trigger(summary, regime, wave)


def _self_ensemble(tier: str, regime: str, root: Path, wave: int) -> dict:
    """Self-ensemble control. Needs the MAPA kernels; reports why if it cannot run."""
    from common import mapa_features
    try:
        controls.export_bootstraps(tier, regime, root)
        comparison = controls.self_ensemble(tier, regime, root)
    except (mapa_features.KernelMissing, FileNotFoundError) as error:
        ledger.append({"tier": tier, "regime": regime, "phase": "E3", "kind": "control", "arm": "self_ensemble",
                       "decision": "not run", "wave": wave, "extra": {"reason": str(error)}}, root)
        return {"ran": False, "reason": str(error)}
    ledger.append(comparison.row("E3", kind="control", wave=wave,
                                 extra={"control": "self-ensemble of mapa12, five bootstrap resamples"}), root)
    return {"ran": True, "gain": comparison.delta, "score": comparison.score}


def _held_out_session_report(name: str, baseline: str, regime: str, root: Path, wave: int) -> dict:
    """MAPA was pretrained on every Tier P session except 5:0: report it separately."""
    splits = _splits()
    targets = sorted({c.target_session for c in splits.list_cells("P", regime)})
    out = {}
    for label, sessions in (("held_out_5_0", [MAPA_HELD_OUT_SESSION]),
                            ("pretraining_sessions", [s for s in targets if s != MAPA_HELD_OUT_SESSION])):
        sessions = [s for s in sessions if s in targets]
        if not sessions:
            continue
        comparison = stats.compare(name, baseline, "P", regime, root, sessions=sessions, descriptive=True)
        ledger.append(comparison.row("E1", kind="info", wave=wave, extra={"sessions": label}), root)
        out[label] = {"delta": comparison.delta, "score": comparison.score, "n_subjects": comparison.n_subjects}
    return out


def _wave2_trigger(summary: dict, regime: str, wave: int) -> None:
    if wave != 1:
        return
    decision = summary["best"]["decision"]
    ridge6 = summary["leave_one_member_out"].get("ridge6", {})
    runs = decision in ("Go", "Hold") or (decision == "Stop" and ridge6.get("keep", False))
    print(f"wave 2 trigger for {regime}: wave 1 decision {decision}; ridge6 leave-one-out contribution "
          f"{ridge6.get('contribution', float('nan')):+.4f} -> wave 2 {'enters' if runs else 'does not enter'} the decision")


def e2(args, root: Path) -> None:
    regime, wave = args.regime, args.wave
    screen = load_state(root, f"e1_{regime}_w{wave}")
    print(f"E2 fusion ladder: Tier P, regime {regime}, wave {wave}; CPU only, minutes, VRAM 0")
    if not screen:
        print(f"MISSING INPUT: {state_dir(root) / f'e1_{regime}_w{wave}.json'} (run E1 first)")
        raise SystemExit(2)
    members = screen["best"]["members"]
    print(f"  members {members}: (a) uniform mean vs (b) stacker, penalty {fuse.STACKER_PENALTY}; "
          f"control: stacker on shuffled training labels")
    if args.dry_run:
        print("dry run: nothing was started")
        return
    if not screen["passed"] and regime != "cross_session":
        ledger.append({"tier": "P", "regime": regime, "phase": "E2", "kind": "info", "arm": screen["best"]["name"],
                       "decision": "not run", "wave": wave, "extra": {"reason": "E1 failed; ensemble claim invalidated"}}, root)
        raise Refused("E1 failed for this regime and wave; the ladder does not run")
    uniform = fuse.fuse(members, "a", "P", regime, root)
    stacked = fuse.fuse(members, "b", "P", regime, root)
    step = stats.compare(stacked, uniform, "P", regime, root, descriptive=True)
    shuffle = controls.shuffled_stacker(members, "P", regime, root)
    adopt = bool(step.delta >= E2_MINIMUM - 1e-12 and shuffle["passed"])
    ledger.append(step.row("E2", wave=wave, decision="adopt (b)" if adopt else "keep (a)",
                           extra={"rule": f"(b) replaces (a) only if it adds >= {E2_MINIMUM}",
                                  "penalty": fuse.STACKER_PENALTY}), root)
    ledger.append(shuffle["comparison"].row("E2", kind="control", wave=wave,
                                            decision="pass" if shuffle["passed"] else "fail",
                                            extra={"control": "shuffled stacker vs uniform mean",
                                                   "tolerance": shuffle["tolerance"]}), root)
    summary = {"regime": regime, "wave": wave, "members": members, "rung": "b" if adopt else "a",
               "stacker_gain": step.delta, "shuffled_delta": shuffle["delta"], "shuffled_passed": shuffle["passed"],
               "threshold": E2_MINIMUM, "penalty": fuse.STACKER_PENALTY}
    save_state(root, f"e2_{regime}_w{wave}", summary)
    print(f"E2: stacker adds {step.delta:+.4f}; shuffled-label stacker differs from uniform by "
          f"{shuffle['delta']:+.4f} ({'pass' if shuffle['passed'] else 'FAIL'}); rung ({summary['rung']})")


def e3(args, root: Path) -> None:
    regime, wave = args.regime, args.wave
    screen, ladder = load_state(root, f"e1_{regime}_w{wave}"), load_state(root, f"e2_{regime}_w{wave}")
    print(f"E3 controls and proposal: Tier P, regime {regime}, wave {wave}; CPU only; the self-ensemble "
          f"refits mapa12 five times per cell (same cost as five mapa12 exports), VRAM 0")
    for name, state in ((f"e1_{regime}_w{wave}", screen), (f"e2_{regime}_w{wave}", ladder)):
        if not state:
            print(f"MISSING INPUT: {state_dir(root) / (name + '.json')}")
            raise SystemExit(2)
    if args.dry_run:
        print(f"  design under test: {ladder['members']} rung ({ladder['rung']})")
        print("dry run: nothing was started")
        return
    members, rung = list(ladder["members"]), ladder["rung"]
    lomo = controls.leave_one_member_out(members, rung, "P", regime, root)
    dropped = [m for m, info in lomo.items() if not info["keep"]]
    for member, info in lomo.items():
        ledger.append({"tier": "P", "regime": regime, "phase": "E3", "kind": "control",
                       "arm": f"leave_out:{member}", "delta": info["contribution"], "wave": wave,
                       "decision": "keep" if info["keep"] else "drop",
                       "extra": {"rule": f"a member contributing under {controls.LOMO_MINIMUM} is dropped", "rung": rung}}, root)
    kept = [m for m in members if m not in dropped]
    single = screen["single"]
    if len(kept) < 2:
        candidate = None
    else:
        candidate = fuse.fuse(kept, rung, "P", regime, root)
    baseline = screen["baseline"]
    summary = {"regime": regime, "wave": wave, "members": kept, "dropped": dropped, "rung": rung,
               "leave_one_member_out": lomo, "baseline": baseline, "e1_passed": screen["passed"]}
    if candidate is None:
        summary.update(decision="Stop", controls_pass=False, reason="leave-one-member-out left fewer than two members")
        ledger.append({"tier": "P", "regime": regime, "phase": "E3", "kind": "comparison", "arm": "ensemble",
                       "baseline": baseline, "decision": "Stop", "wave": wave, "extra": {"reason": summary["reason"]}}, root)
    else:
        final = stats.compare(candidate, baseline, "P", regime, root)
        best_member = max(kept, key=single.get)
        over_member = stats.compare(candidate, best_member, "P", regime, root, descriptive=True)
        self_control = _self_ensemble("P", regime, root, wave)
        margin = over_member.delta - self_control["gain"] if self_control["ran"] else None
        self_pass = bool(margin is not None and margin >= controls.SELF_ENSEMBLE_MARGIN - 1e-12)
        concentration = controls.subject_concentration(final)
        shuffle_pass = rung == "a" or ladder["shuffled_passed"]
        controls_pass = bool(screen["passed"] and self_pass and shuffle_pass)
        decision = final.decision
        if decision in ("Go", "Hold") and not self_pass:
            decision = "Stop"       # the gain is variance reduction (or the control could not run)
        ledger.append(final.row("E3", wave=wave, decision=decision,
                                extra={"table_decision": final.decision, "members": kept, "rung": rung,
                                       "gain_over_best_member": over_member.delta, "best_member": best_member,
                                       "self_ensemble": self_control, "self_ensemble_margin": margin,
                                       "self_ensemble_pass": self_pass, "shuffled_stacker_pass": shuffle_pass,
                                       "e1_pass": screen["passed"], "controls_pass": controls_pass,
                                       "subject_concentration": concentration}), root)
        summary.update(candidate=candidate, decision=decision, table_decision=final.decision,
                       tier_p_gain=final.delta, gain_over_best_member=over_member.delta,
                       self_ensemble=self_control, self_ensemble_margin=margin, self_ensemble_pass=self_pass,
                       shuffled_stacker_pass=shuffle_pass, controls_pass=controls_pass,
                       subject_concentration=concentration,
                       mapa_pretraining_split=_held_out_session_report(candidate, baseline, regime, root, wave))
        spec = {"members": sorted(kept), "rung": rung, "normalization": fuse.NORMALIZATION,
                "penalty": fuse.STACKER_PENALTY if rung == "b" else None, "baselines": list(BASELINES), "wave": wave}
        if regime != "cross_session" and not args.no_propose:
            lock.propose(regime, "ensemble", candidate, spec, final.delta, decision, controls_pass, root)
            if regime == "within":   # cross-session takes the within-session choice
                lock.propose("cross_session", "ensemble", candidate, spec, None, "Inherited", controls_pass, root)
            print(f"proposal written under {Path(root) / 'frozen' / 'proposals'}")
        print(f"E3: {candidate} vs {baseline}: {final.delta:+.4f} ({final.subjects_up}/{final.n_subjects} up, "
              f"p={final.p_signflip:.4f}) table {final.decision}; self-ensemble margin "
              f"{margin if margin is None else round(margin, 4)}; decision {decision}; controls "
              f"{'pass' if controls_pass else 'FAIL'}; carried by {concentration['carried_by'] or 'no single subject'}")
    save_state(root, f"e3_{regime}_w{wave}", summary)


def freeze(args, root: Path) -> None:
    found = lock.proposals(args.regime, root)
    print(f"freeze {args.regime}: {len(found)} proposal(s): "
          + ", ".join(f"{p['track']}:{p['member']} gain={p['tier_p_gain']} {p['decision']} controls={p['controls_pass']}" for p in found))
    if args.dry_run:
        print("dry run: nothing was written")
        return
    try:
        design = lock.freeze(args.regime, root)
    except lock.FreezeError as error:
        raise Refused(str(error)) from None
    ledger.append({"tier": "P", "regime": args.regime, "phase": "FREEZE", "kind": "info", "arm": design["member"],
                   "candidate": lock.candidate_id(design), "config_hash": design["config_hash"],
                   "decision": "frozen", "delta": design["tier_p_gain"],
                   "extra": {"track": design["track"], "considered": design["considered"]}}, root)
    print(f"frozen: {lock.candidate_id(design)} = {design['member']}")


# --------------------------------------------------------------------------- E4, E5

def _frozen_ensemble(regime: str, root: Path) -> dict:
    try:
        design = lock.frozen(regime, root)
    except lock.FreezeError as error:
        print(f"MISSING INPUT: {error}")
        raise SystemExit(2) from None
    if design["track"] != "ensemble":
        raise Refused(f"the frozen candidate of {regime} is {lock.candidate_id(design)} "
                      f"({design['member']}); the ensemble track has nothing to run here")
    return design


def _need_predictions(members, tier: str, regime: str, root: Path) -> None:
    absent = {m: len(missing_predictions(m, tier, regime, root)) for m in members}
    if any(absent.values()):
        for member, count in absent.items():
            if count:
                print(f"MISSING INPUT: {count} prediction files of {member} for tier {tier} "
                      f"(run launch/e0_export.sh --tier {tier} --regime {regime})")
        raise SystemExit(2)


def quick_confirmation(design: dict, root: Path) -> bool:
    """Only the independently verified fixed DIVER/MAPA12 mean can omit the DIVER OOF fits."""
    spec, regime = design["spec"], design["regime"]
    if (design["track"] != "ensemble" or regime not in ("within", "cross_session") or
            spec.get("rung") != "a" or set(spec.get("members", ())) != {"diver", "mapa12"} or
            spec.get("normalization") != fuse.NORMALIZATION or spec.get("penalty") is not None):
        return False
    path = Path(root) / "parity" / "diver_uniform_confirmation.json"
    if not path.is_file():
        return False
    parity = json.loads(path.read_text())
    return (parity.get("passed") is True and parity.get("tier") == "P" and
            parity.get("independent_full_heads") is True and parity.get("tolerance") == E0_TOLERANCE and
            parity.get("regimes", {}).get(regime, {}).get("passed") is True)


def e4(args, root: Path) -> None:
    regime = args.regime
    design = _frozen_ensemble(regime, root)
    candidate, spec = lock.candidate_id(design), design["spec"]
    print(f"E4 veto check: Tier V, regime {regime}, candidate {candidate} = {design['member']}; CPU only, minutes, VRAM 0")
    try:
        ledger.guard_candidate(root, "V", regime, candidate)
    except ledger.CandidateConflict as error:
        raise Refused(str(error)) from None
    needed = sorted(set(spec["members"]) | set(spec["baselines"]))
    _need_predictions(needed, "V", regime, root)
    if args.dry_run:
        print("dry run: nothing was started")
        return
    quick = quick_confirmation(design, root)
    undefined_select = {}
    for member in needed:
        report = check_member(member, "V", regime, root, need_train=not (quick and member == "diver"))
        if not report["passed"]:
            raise Refused(f"{member} fails the export check on Tier V: {report['problems'][:3]}")
        if report["undefined_select_cells"]:
            undefined_select[member] = report["undefined_select_cells"]
    name = fuse.fuse(spec["members"], spec["rung"], "V", regime, root)
    if name != design["member"]:
        raise Refused(f"frozen member {design['member']} does not match its specification ({name})")
    baseline = stats.best_baseline("V", regime, root, spec["baselines"])
    result = stats.compare(name, baseline, "V", regime, root)
    ledger.append(result.row("E4", candidate=candidate, config_hash=design["config_hash"],
                             extra={"subject_concentration": controls.subject_concentration(result),
                                    **({"protocol": confirm_fallback.PROTOCOL, "undefined_select": undefined_select}
                                       if regime == "within" and design["config_hash"] == "cc49ff955fa0ae88" else {}),
                                    "note": "single veto pass; Tier V does not compare candidates"}), root)
    save_state(root, f"e4_{regime}", {"candidate": candidate, "decision": result.decision, "delta": result.delta,
                                      "baseline": baseline, "subjects_up": result.subjects_up,
                                      "n_subjects": result.n_subjects, "p_signflip": result.p_signflip})
    print(f"E4 {result.decision}: {name} vs {baseline} {result.delta:+.4f} "
          f"({result.subjects_up}/{result.n_subjects} up, p={result.p_signflip:.4f})")


def e5(args, root: Path) -> None:
    regime = args.regime
    design = _frozen_ensemble(regime, root)
    candidate, spec = lock.candidate_id(design), design["spec"]
    print(f"E5 score: Tier T, regime {regime}, candidate {candidate} = {design['member']}; CPU only, minutes, VRAM 0")
    needed = sorted(set(spec["members"]) | set(spec["baselines"]))
    _need_predictions(needed, "T", regime, root)
    if args.dry_run:
        try:
            ledger.guard_candidate(root, "T", regime, candidate)
        except ledger.CandidateConflict as error:
            raise Refused(str(error)) from None
        print("dry run: nothing was started; no test label was read")
        return
    amended = regime == "within" and design["config_hash"] == "cc49ff955fa0ae88"
    undefined_select = {}
    for member in needed if amended else (["diver"] if quick_confirmation(design, root) else []):
        report = check_member(member, "T", regime, root, need_train=False)  # existing T schema; select labels only
        if not report["passed"]:
            raise Refused(f"{member} fails the export check on Tier T: {report['problems'][:3]}")
        if report["undefined_select_cells"]:
            undefined_select[member] = report["undefined_select_cells"]
    # The fused scores need no eval label, so they are written before the lock is opened.
    name = fuse.fuse(spec["members"], spec["rung"], "T", regime, root)
    if name != design["member"]:
        raise Refused(f"frozen member {design['member']} does not match its specification ({name})")
    try:
        lock.require_test_unlocked("T", regime, root, purpose="score", config_hash=design["config_hash"])
    except lock.TestLocked as error:
        raise Refused(str(error)) from None
    baseline = stats.best_baseline("T", regime, root, spec["baselines"])
    result = stats.compare(name, baseline, "T", regime, root)
    for member in needed:
        ledger.append({"tier": "T", "regime": regime, "phase": "E5", "kind": "member", "arm": member,
                       "score": stats.macro(member, "T", regime, root), "decision": "Reported",
                       "candidate": candidate}, root)
    ledger.append(result.row("E5", candidate=candidate, config_hash=design["config_hash"],
                             extra={"subject_concentration": controls.subject_concentration(result),
                                    **({"protocol": confirm_fallback.PROTOCOL, "undefined_select": undefined_select}
                                       if amended else {})}), root)
    save_state(root, f"e5_{regime}", {"candidate": candidate, "score": result.score, "baseline": baseline,
                                      "baseline_score": result.baseline_score, "delta": result.delta,
                                      "ci": [result.ci_lo, result.ci_hi], "p_signflip": result.p_signflip})
    print(f"E5: {name} {result.score:.4f} vs {baseline} {result.baseline_score:.4f} ({result.delta:+.4f}, "
          f"95% CI {result.ci_lo:+.4f} to {result.ci_hi:+.4f}, p={result.p_signflip:.4f})")


# --------------------------------------------------------------------------- parity and reproduction

def _reference_cells(member: str, regime: str) -> dict:
    """The reproduction's own numbers per (target session, task): ``val`` and penalty where
    it stored them, test AUROC per fold. Empty where it has not finished."""
    out, results = {}, paths.repro() / "results"
    splits = _splits()
    if member in RIDGE_MEMBERS:
        tap = 12 if member == "mapa12" else 0
        prefix = {"within": "ws", "cross_session": "csession", "cross_subject": "cs"}[regime]
        key = f"enc{tap}|std" if regime == "cross_subject" else f"enc{tap}_elec|std"
        for target in sorted({c.target_session for c in splits.list_cells("T", regime)}):
            subject, trial = paths.parse_session(target)
            file = results / f"{prefix}_S{subject}T{trial}.json"
            if not file.is_file():
                continue
            for name, cell in json.loads(file.read_text())["cells"].items():
                entry = (cell.get("cells") or {}).get(key)
                if entry is None:
                    continue
                task = name.rsplit("|", 1)[1]
                if regime == "within":
                    lams = entry.get("lam_mult", [])
                    for i, fold in enumerate(entry.get("folds", [])):
                        out[(target, task, fold["fold_idx"])] = {"test": fold["test_roc_auc"],
                                                                 "lam": lams[i] if i < len(lams) else None}
                else:
                    out[(target, task, 0)] = {"test": entry["test"], "val": entry.get("val"), "lam": entry.get("lam_mult")}
    elif regime in PUBLISHED.get(member, {}):
        track = {"within": "Within-Session", "cross_session": "Cross-Session", "cross_subject": "Cross-Subject"}[regime]
        for cell in splits.list_cells("T", regime):
            subject, trial = paths.parse_session(cell.target_session)
            file = (results / f"diver_s{subject}_t{trial}_{cell.task}.json" if member == "diver"
                    else results / "cnn-official" / track / f"s{subject}_t{trial}_{cell.task}.json")
            if not file.is_file():
                continue
            folds = json.loads(file.read_text())
            if cell.fold < len(folds):
                value = folds[cell.fold]
                out[(cell.target_session, cell.task, cell.fold)] = {
                    "test": value["test_roc_auc"] if isinstance(value, dict) else float(value)}
    return out


def parity(args, root: Path) -> None:
    """Check of the MAPA kernels against the reproduction's finished within- and
    cross-session results without reading a test label: per cell, the readout is refit
    from the stored kernel (training and validation labels) and the penalty it chooses
    (and the validation AUROC, where the reproduction stored one) is compared with the
    reproduction's. No prediction file is written."""
    from common import mapa_features, ridge
    from ensemble.members import mapa
    splits = _splits()
    report = {}
    for member in RIDGE_MEMBERS:
        tap = mapa.TAP[member]
        reference = _reference_cells(member, args.regime)
        cells = [c for c in splits.list_cells("T", args.regime)
                 if (c.target_session, c.task, c.fold) in reference
                 and mapa_features.kernel_file("E", c.regime, c.key(), tap).is_file()]
        print(f"parity {member} {args.regime}: {len(reference)} reference cells in {paths.repro() / 'results'}, "
              f"{len(cells)} of them with a stored kernel; CPU only, about {max(1, len(cells) // 6)} min, VRAM 0")
        if args.dry_run or not cells:
            continue
        equal, gaps, differing = 0, [], []
        for cell in cells:
            split = splits.get_cell(cell)
            fit = ridge.fit_predict([mapa_features.kernels(cell, tap)], split.labels("train"), split.labels("select"))
            theirs = reference[(cell.target_session, cell.task, cell.fold)]
            same = bool(theirs["lam"] is not None and np.isclose(fit["lam"], theirs["lam"], rtol=1e-9))
            equal += same
            if not same:
                differing.append({"cell": cell.key(), "ours": fit["lam"], "reproduction": theirs["lam"],
                                  "n_tied": fit["n_tied"]})
            if theirs.get("val") is not None:
                gaps.append(abs(fit["select_auroc"] - theirs["val"]))
        report[member] = {"cells": len(cells), "penalty_equal": equal, "penalty_differs": differing[:20],
                          "max_abs_val_gap": max(gaps) if gaps else None, "val_cells": len(gaps),
                          "sessions": sorted({c.target_session for c in cells})}
        ledger.append({"tier": "T", "regime": args.regime, "phase": "EK", "kind": "info", "arm": f"parity:{member}",
                       "baseline": "reproduction results (penalty, validation AUROC)",
                       "delta": report[member]["max_abs_val_gap"], "decision": "reported",
                       "extra": dict(report[member], note="no test label read")}, root)
        print(f"  penalty equal in {equal}/{len(cells)} cells; max |validation AUROC gap| "
              f"{report[member]['max_abs_val_gap']} over {len(gaps)} cells; sessions {report[member]['sessions']}")
    if args.dry_run:
        print("dry run: nothing was started")
    elif report:
        save_state(root, f"parity_{args.regime}", report)


def reproduce(args, root: Path) -> None:
    """Official-protocol scores of one member on Tier T, as a logged reproduction run.
    Reads evaluation-session test labels through the lock with purpose ``reproduction``."""
    member, regime = args.member, args.regime
    splits = _splits()
    print(f"reproduction: {member}, {regime}, official protocol (train, official val selects, official test scores); "
          f"CPU only, minutes, VRAM 0")
    _need_predictions([member], "T", regime, root)
    if args.dry_run:
        print("dry run: nothing was started; no test label was read")
        return
    os.environ["NPEXP_TEST_PURPOSE"], os.environ["NPEXP_TEST_MEMBER"] = "reproduction", member
    try:
        lock.require_test_unlocked("T", regime, root, purpose="reproduction", member=member)
    except lock.TestLocked as error:
        raise Refused(str(error)) from None
    cell_scores = stats.member_scores(member, "T", regime, root)
    by_task: dict = {}
    for (session, task, fold), value in cell_scores.items():
        by_task.setdefault(task, {}).setdefault(session, []).append(value)
    # leaderboard macro: folds, then sessions, then the 15 tasks
    per_task = {t: float(np.nanmean([np.nanmean(v) for v in sessions.values()])) for t, sessions in by_task.items()}
    macro = float(np.mean(list(per_task.values())))
    published = PUBLISHED.get(member, {}).get(regime)
    reference = _reference_cells(member, regime)
    gaps = [cell_scores[k] - v["test"] for k, v in reference.items() if k in cell_scores]
    body = {"member": member, "regime": regime, "macro": macro, "published": published,
            "gap_to_published": None if published is None else macro - published, "per_task": per_task,
            "cells": len(cell_scores),
            "vs_reproduction": {"cells": len(gaps), "max_abs_gap": float(np.max(np.abs(gaps))) if gaps else None,
                                "mean_gap": float(np.mean(gaps)) if gaps else None},
            "per_cell": {f"{s}|{t}|{f}": v for (s, t, f), v in sorted(cell_scores.items())}}
    out = Path(root) / "reproduction" / f"{member}_{regime}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1) + "\n")
    ledger.append({"tier": "T", "regime": regime, "phase": "REPRO", "kind": "info", "arm": member,
                   "baseline": "published leaderboard macro", "score": macro, "baseline_score": published,
                   "delta": body["gap_to_published"], "decision": "Reported",
                   "extra": {"purpose": "reproduction", "vs_reproduction": body["vs_reproduction"]}}, root)
    print(f"reproduction {member} {regime}: macro {macro:.4f}"
          + (f", published {published:.4f}, gap {macro - published:+.4f}" if published is not None else "")
          + f"; {len(gaps)} cells also in the reproduction's results -> {out}")


# --------------------------------------------------------------------------- entry point

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function, *, tier=False, wave=False):
        sub = commands.add_parser(name)
        sub.add_argument("--regime", required=True, choices=("within", "cross_session", "cross_subject"))
        sub.add_argument("--dry-run", action="store_true")
        if tier:
            sub.add_argument("--tier", required=True, choices=("P", "V", "T"))
        if wave:
            sub.add_argument("--wave", type=int, choices=(1, 2), default=1)
        sub.set_defaults(function=function)
        return sub

    sub = add("export", export, tier=True)
    sub.add_argument("--members", nargs="+", required=True)
    sub.add_argument("--gpu-shard", default="0/1", help="i/n: take every n-th session pair (trained members)")
    sub.add_argument("--available-only", action="store_true",
                     help="trained members: only session pairs whose recordings are on disk; rerun for the rest")
    sub.add_argument("--quick", action="store_true",
                     help="trained members: seed 0 select/eval scores only (no out-of-fold, no seeds 1-2)")
    sub.add_argument("--seeds-only", action="store_true", help="trained members: seeds 1-2 select/eval only")
    sub.add_argument("--oof-only", action="store_true",
                     help="trained members: add seed 0's out-of-fold training scores, keeping its existing full model")
    sub = add("e0", e0, tier=True, wave=True)
    sub.add_argument("--members", nargs="+")
    add("e1", e1, wave=True).add_argument("--members", nargs="+",
                                          help="restrict the E0 pool to these members (recorded)")
    add("e2", e2, wave=True)
    add("e3", e3, wave=True).add_argument("--no-propose", action="store_true")
    add("freeze", freeze)
    add("e4", e4)
    add("e5", e5)
    add("parity", parity)
    add("reproduce", reproduce).add_argument("--member", required=True)
    args = parser.parse_args(argv)
    root = paths.results()
    try:
        paths.exit_on_cuda_oom(lambda: args.function(args, root))
    except _splits().MissingInput as error:
        print(f"MISSING INPUT: {error}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

"""Report-only export of per-cell Tier T within-session test AUROCs for the leaderboard files.

The scored T predictions are unchanged; this rereads the eval labels once (logged to
test_access.log, purpose report-export) and refuses to write anything unless the recomputed
aggregates reproduce the recorded T comparison exactly.
"""
import csv, json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import ledger, predictions, splits, stats

BASE = Path(os.environ["NPEXP_BASE"])
RESULTS = BASE / "results"
ROOT = RESULTS   # predictions.path adds amendments/<protocol> for within V/T
OUT = Path(sys.argv[1])
CAND, BASELINE = "ens_a__diver+mapa12", "mapa12"
META = json.loads(sys.argv[2])

row = [r for r in csv.DictReader(open(RESULTS / "ledger.csv"))
       if r["tier"] == "T" and r["regime"] == "within" and r["kind"] == "comparison"]
assert len(row) == 1, len(row)
row = row[0]
extra = json.loads(row["extra"])
# e5 writes the fused files, opens the lock, scores, then saves e5_within.json: nothing may be newer.
scored_at = (RESULTS / "ensemble/e5_within.json").stat().st_mtime

cells = splits.list_cells("T", "within")
assert len(cells) == 360
for c in cells:
    for m in (CAND, BASELINE):
        p = predictions.path(c, m, ROOT)
        assert p.stat().st_mtime <= scored_at, f"{p} modified after T scoring"

with open(RESULTS / "test_access.log", "a") as f:
    f.write(json.dumps({"argv": sys.argv[:1], "candidate": "ensemble:cc49ff955fa0ae88", "commit": ledger.commit(),
                        "member": CAND, "pid": os.getpid(), "purpose": "report-export", "regime": "within",
                        "tier": "T", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "note": "per-cell AUROCs of already-scored, unchanged predictions for leaderboard files"}) + "\n")

auc = {CAND: {}, BASELINE: {}}
for c in cells:
    split = splits.get_cell(c)
    labels = split._labels["eval"]()          # report-only reread; the lock refuses re-scoring by design
    rows = split.rows["eval"]
    for m in (CAND, BASELINE):
        pred = predictions.read(c, m, ROOT)
        s = predictions.align(pred.scores["eval"], pred.rows["eval"], rows, f"{m} {c.key()}")
        auc[m][c] = stats.auroc(s, labels)


def subject_mean(a):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c, v in a.items():
        by[splits.parse_session(c.target_session)[0]][c.task][c.target_session].append(v)
    return float(np.mean([np.mean([np.mean([np.mean(f) for f in t.values()]) for t in s.values()]) for s in by.values()]))


def leaderboard(a):
    by = defaultdict(list)
    for c, v in a.items():
        by[c.task].append(v)
    return float(np.mean([np.mean(v) for v in by.values()]))


assert all(np.isfinite(v) for m in auc for v in auc[m].values())
got = {m: subject_mean(auc[m]) for m in auc}
assert abs(got[CAND] - float(row["score"])) < 1e-12, (got[CAND], row["score"])
assert abs(got[BASELINE] - float(row["baseline_score"])) < 1e-12, (got[BASELINE], row["baseline_score"])
for sess, d in extra["per_session"].items():
    mine = np.mean([auc[CAND][c] - auc[BASELINE][c] for c in cells if c.target_session == sess])
    assert abs(mine - d) < 1e-12, (sess, mine, d)

OUT.mkdir(parents=True, exist_ok=False)
for task in splits.TASKS:
    results = {}
    for c in sorted((c for c in cells if c.task == task), key=lambda c: (c.target_session, c.fold)):
        folds = results.setdefault(c.target_session, {"population": {"one_second_after_onset": {
            "time_bin_start": 0.0, "time_bin_end": 1.0, "folds": []}}})["population"]["one_second_after_onset"]["folds"]
        folds.append({"test_roc_auc": float(auc[CAND][c]), "fold_idx": int(c.fold)})
    (OUT / f"population_{task}.json").write_text(json.dumps({**META, "evaluation_results": results}, indent=1) + "\n")
print(json.dumps({"subject_weighted": got, "leaderboard_style": {m: leaderboard(auc[m]) for m in auc},
                  "files": len(list(OUT.glob("population_*.json")))}))

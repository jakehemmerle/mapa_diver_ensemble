# Classical track: staged and running

November 25, 2026. Plan: `../06-classical-and-ensemble-plan.md` (phases C0–C5). Rules: `../05-rules-verdicts.md`.
Everything runs on the GPU server from `~/runs/neuroprobe-experiments/code`. Sync with `bash server/sync.sh` from this folder; it uses `rsync -a` without delete and writes `code/COMMIT.json`.

## November 25 decision record (21:15 CST)

Ridge6 C0 macro AUROC was **0.676019 within**, **0.666477 cross-session**, **0.586089 cross-subject**. Only cross-subject met the ±0.005 parity criterion; this is our reimplementation, not an exact reproduction. No retuning against the published test scores. All 13 permitted sessions are present; `data_dev`, the classical cache and kernel store resolve to `/scratch`. Tier P Ridge6 macro: 0.6740 within, 0.6517 cross-session, 0.5740 cross-subject.

`spark-npexp-cchain` runs C1. `npexp-c2-after-c1` starts C2 only after successful completion; `npexp-c3-after-c2-<regime>.path` starts C3 when that regime's C2 report lands. C2 uses the **kept set's** C1 decision: Go or Hold with passing controls adds the kept blocks, otherwise MAPA receives base6 only. Cross-session copies the within-session design. An incomplete C1 is not a negative result and cannot start C2. C3 uses the original projection source (Ridge6 in C1, MAPA frontend in C2). No frozen candidate, Tier V pass or candidate test score at this checkpoint. The staging snapshot below predates these results.

Cross-session E0 initially excluded Ridge6: five of 330 cells exceeded the 1e-6 reported/select-score AUROC tolerance (max 3.92e-6). Independent refits reproduced the selected penalty and stored float32 select scores bit-for-bit in all five. In four cells the grid's matrix product and final score-vector product differed at a near tie; in the fifth, casting the final scores to float32 changed a tie. As for MAPA, metadata now retains the grid objective as `fit_select_auroc` and reports AUROC on the actual export-precision score vector as `select_auroc`. No change to the ridge grid, chosen penalty, scores or tolerance.

## Status at 16:25 CST (commit d2510f3, lock written by Main)

| Unit | What | State |
|---|---|---|
| `npexp-data-download` | 13 Tier P recordings into `data_dev/` | Running; 5 of 13 done; finish about 17:00 |
| `spark-npexp-cfeat-eval` | Features, 12 evaluation sessions (in-window, bipolar, all blocks) | Running since 15:57 |
| `spark-npexp-cfeat-btbank2_3` | Tier P base cache, the other 10 recordings as each lands (waits up to 120 min) | Running |
| `spark-npexp-cchain` | Unattended chain, `work/chain/classical_chain.sh`, log `logs/cchain.log` | Waiting for the evaluation cache |

The chain runs these steps in order, each only if the previous one exits 0:
1. C0 extract and fit, under the clean commit. No earlier C0 fit exists, so no refit is needed.
2. C0 score through the lock: `work/chain/c0_score.py`. It is the committed `c0 --stages score`, and the same process also writes the per-cell AUROCs to `results/classical/c0_cells.json`.
3. `work/chain/c0_vs_pr20.py`: per-subject and per-task deltas against PR #20's published per-session numbers (copied to `work/pr20/`). Reads no labels.
4. Waits until all 13 Tier P base stores exist, then runs `p0`.
5. C1. Its own extract stage builds the rest of the Tier P plan first.

The chain's scripts live in `work/chain/` on the server, not in this repository.

**Status, one command:**
`ssh gpu 'cd ~/runs/neuroprobe-experiments; for u in npexp-data-download spark-npexp-cfeat-eval spark-npexp-cfeat-btbank2_3 spark-npexp-cchain; do printf "%-28s %s\n" $u "$(systemctl --user is-active $u)"; done; tail -n1 logs/download-dev.log; for f in logs/cfeat-*.log logs/cchain.log; do echo "== $f"; tail -n1 $f; grep -E "FAILED|Traceback|Error|refused|stopped" $f | tail -n3; done'`

**C0 parity and the C1 decision from the ledger:**
`ssh gpu 'cd ~/runs/neuroprobe-experiments; .venv/bin/python -c "import csv; [print(r[\"phase\"], r[\"regime\"], r[\"kind\"], r[\"arm\"], r[\"score\"], r[\"baseline_score\"], r[\"delta\"], r[\"subjects_up\"], r[\"p_signflip\"], r[\"decision\"]) for r in csv.DictReader(open(\"results/ledger.csv\")) if (r[\"phase\"]==\"c0\" and r[\"kind\"]==\"member\") or (r[\"phase\"]==\"c1\" and r[\"kind\"]==\"comparison\")]"; grep -h "^\[c0\]\|^== \|per subject\|per task\|^\[c1\].*kept" logs/cchain.log'`

**C2 to C5.** Run from `~/runs/neuroprobe-experiments/code`.
- C2 and C3 need the Tier P MAPA kernels and the `mapa12` Tier P predictions to exist:
  - `NPEXP_WORKERS=5 NPEXP_FIT_WORKERS=6 NPEXP_MEMORY_MAX=30G bash classical/launch/c2.sh cpu`
  - then `bash classical/launch/c3.sh cpu`
- C4 and C5 are Main's, after freezing:
  - `bash classical/launch/c4.sh cpu`
  - `bash classical/launch/c5.sh cpu`
- Add `--dry-run` to any of these to see the missing inputs.

## Launch commands, in order

Run these on `gpu` from `~/runs/neuroprobe-experiments/code`.

**Launcher behaviour**
- The first argument is `cpu` (the design) or a GPU UUID.
- With a GPU UUID, the launcher declares its VRAM need, reads free memory with `nvidia-smi`, and exits 3 if the card is short. Classical launchers declare 2048 MiB per fit worker.
- `--dry-run` runs in the foreground, CPU only, and starts nothing. It exits 0, or 2 with the missing inputs listed.
- Units are named `spark-npexp-<tag>`.
- Sizing is set with `NPEXP_WORKERS`, `NPEXP_FIT_WORKERS`, `NPEXP_THREADS`, `NPEXP_MEMORY_MAX` and `NPEXP_CLASSICAL_CACHE`.
- The values below stay within Main's budget while `spark-neuroprobe-mapa-board` runs: at most half the cores and 40 GiB in total.

**Commands**
1. `NPEXP_WORKERS=8 NPEXP_MEMORY_MAX=24G bash classical/launch/features.sh cpu --sessions eval`
2. `NPEXP_WORKERS=5 NPEXP_MEMORY_MAX=14G bash classical/launch/features.sh cpu --sessions dev --plan base --wait-minutes 120`
3. C0, in three steps:
   - `NPEXP_TAG=fit NPEXP_FIT_WORKERS=6 NPEXP_MEMORY_MAX=16G bash classical/launch/c0.sh cpu --stages extract fit`
   - after the commit and lock: `bash classical/launch/c0.sh cpu --stages fit --overwrite`
   - then: `bash classical/launch/c0.sh cpu --stages score`
4. `bash classical/launch/p0.sh cpu`: `ridge6` on every Tier P cell. This is the C1 baseline and the ensemble's Ridge 6-band member on Tier P.
5. `NPEXP_WORKERS=5 NPEXP_MEMORY_MAX=24G bash classical/launch/features.sh cpu --sessions dev`: the rest of the Tier P plan (surrogates, session-filtered stores, the other two references).
6. `bash classical/launch/c1.sh cpu`: block screen, kernel-weight selection and controls. Needs the commit first.
7. `bash classical/launch/c2.sh cpu`: needs the Tier P MAPA kernels (taps 0 and 12) and the `mapa12` Tier P predictions, both from the ensemble track.
8. `bash classical/launch/c3.sh cpu`: controls report. It rewrites the classical proposal with the final `controls_pass`.
9. `c4.sh` (Tier V) and `c5.sh` (Tier T) are Main's to launch.
   - Both need `results/frozen/<regime>.json`.
   - `c5` also needs a `score` grant.
   - When the frozen candidate belongs to the ensemble track, they only export the classical members' scores and read no labels.

## Resources per job

| Job | Wall | RAM | Disk | VRAM | How derived |
|---|---|---|---|---|---|
| Evaluation cache, per session | 195–360 s with 5–8 workers | Unit peak 22.9 GiB at 8 workers, 15.0 GiB at 5 | 0.42–0.62 GiB | 0 | Measured: `logs/cfeat-*.log`, `systemctl show -p MemoryPeak` |
| Evaluation cache, all 12 | about 1 h | as above | about 6.5 GiB | 0 | Measured per session × 12 |
| Tier P base cache, 13 sessions | about 1.2 h of extraction, but paced by the download | ≤ 15 GiB at 5 workers | about 7 GiB | 0 | Measured per session |
| Tier P full plan (adds surrogates, session-filtered, references) | about 3–5 h at 5 workers | about 4.5 GiB per worker while a session-filtered store is built | about 25 GiB | 0 | Estimate. The in-window part scales from measurements; session-filtered cost (20 core-s per channel) is assumed, not measured |
| C0 fit | about 1 h | ≤ 16 GiB | under 0.1 GiB of predictions | 0 | Estimate. Fit cost per cell is assumed (1.5–9 core-s); it will be replaced by the measured C0 time |
| p0 (Tier P ridge6) | about 0.5–1 h | ≤ 16 GiB | under 0.1 GiB | 0 | Estimate, same basis as C0 |
| C1 | dry run says about 270 fit core-hours, i.e. about 11 h at 6 × 4 threads | ≤ 16 GiB | about 1 GiB of predictions | 0 | Estimate. 840 cells × 24–55 arms × assumed per-fit cost. Recalibrate from the C0 and p0 timings before launch |
| C2 | about 1 h | ≤ 16 GiB | under 0.5 GiB | 0 | Estimate |
| C3, C4, C5 | under 1 h each | ≤ 16 GiB | small | 0 | Estimate |

The memory figures above come from before a later change: re-referenced signals are now built one contact at a time, which roughly halves the per-worker raw buffers. Bipolar output was checked on btbank3_0 and is bit-identical to the old path.

Disk:
- The classical cache root is `$NPEXP_CLASSICAL_CACHE` (default `work/classical`).
- Every free-space check measures the filesystem that holds the path being written. The floor is 250 GiB for classical caches on whichever filesystem holds them (`classical/run.py` `MIN_FREE_GIB`) and 350 GiB for the download.
- Jake's directive: everything bulk of ours goes to `/scratch/neuroprobe-experiments`. The caches and `data_dev` built so far sit on `/` while the C0/p0/C1 chain reads them; they are copied, verified and replaced by symlinks (`work/classical`, `data_dev`) at the first point no unit has them open, before C2.
- `features.cache_root()` refuses (exit 6) when `NPEXP_CLASSICAL_CACHE` is set or `work/classical` is a symlink and the path resolves onto `/`, so an unmounted `/scratch` never sends a cache to the system disk.
- MAPA kernels are read only through `common.mapa_features`; they live on `/scratch` (ext4, checked).

## Declared defaults (Ridge 6-band reimplementation)

The full list, with what is from PR #20 and what is ours, is at the top of `features.py` (`DEFAULTS`, hash `be9f84c18b619eee`).

**Stated in PR #20**
- Lite electrodes as adjacent bipolar pairs on the same shaft.
- Line noise removed.
- Six streams at 200 Hz: 1–30 Hz waveform; theta, alpha, beta and 30–70 Hz log envelopes; 70–170 Hz log amplitude.
- Median/MAD normalization per channel over the session.
- Ten 100 ms bins over the second after onset.
- Features z-scored on the training rows; ridge penalty chosen on validation AUROC.
- Cross-subject: features averaged within DKT regions.

**Our choices**
- Decimation to a 512 Hz working rate with polyphase FIR.
- Notches at 60, 120 and 180 Hz, Q = 30.
- Bands: theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz.
- Butterworth order 4, zero phase. In-window padding is odd reflection of the window itself.
- Envelope from the Hilbert transform.
- Log floor 1e-9; normalized values clipped to ±50.
- Normalization statistics: in-window pipeline on one-second tiles every 4 s across the session.
- MAPA's 25-point penalty grid, ties to the smallest penalty.
- Regions: the DKT column restricted to MAPA's 74-label vocabulary. A bipolar pair takes the region of its lower-numbered contact.
- Cross-session uses the channels shared by name.
- Subject 9 (1024 Hz): first second only. The benchmark slices 2048 samples, which is 2 s at that rate.

**Approved by Main**
- 512 Hz decimation.
- Kernel weight per arm chosen from {0.1, 0.25, 0.5, 1.0} on the Tier P select-part macro, ties to the smaller weight. Cross-session copies the within-session choice.
- `controls_pass` = sham gain < +0.002 AND the arm beats its random-projection control by ≥ +0.002. The phase surrogate is recorded but does not gate.
- B4 is not defined across subjects; it is skipped there and recorded in the ledger.
- Bipolar is the design; the other references are reported only.

**Declared deviation**
- C1's extra-kernel control projects the Ridge 6-band base features instead of the MAPA frontend, because the Tier P MAPA kernels do not exist when C1 runs. This is `--rp-source base6`, the default for c1, and it is written to prediction meta and the ledger.
- C2 uses MAPA tap 0 as planned.

Window discipline:
- Tiers V and T accept only `window_mode="window"`.
- Session-filtered stores are Tier P measurements (`@session` arms).

## Tier P data

- 13 sessions: 1_0, 2_1, 2_2, 2_3, 2_5, 2_6, 3_2, 4_2, 5_0, 6_1, 6_4, 8_0, 9_0. Session 6_0 and the partial sessions are not downloaded.
- Downloaded by `server/download_dev.py`: sequential, idle I/O, zips deleted after extraction, sha256 receipts in `data_dev/.receipts/`.
- Metadata for all sessions already existed in the reproduction's `data/` and is symlinked into `data_dev/`.
- Total is 62.8 GiB compressed. 4 of 13 recordings were done at 16:11.

## Verified

- Local tests (`python -m pytest classical/tests -q`): `31 passed in 55.67s`. The same suite passed in the server venv: `30 passed in 40.44s`, before `test_phases.py` was added.
- What the tests cover:
  - ridge matches a direct closed-form solve and sklearn `Ridge`;
  - ties go to the smallest penalty;
  - out-of-fold scores are independent of the held block's labels;
  - Tier V halves are time-ordered, purged by at least 2 s on each side, and never reach official test rows;
  - out-of-fold blocks partition the training rows and keep fit windows more than 2 s from the held block;
  - Tier T `labels("eval")` raises `TestLocked` without the lock, while rows and onsets stay reachable;
  - on a synthetic recording with a planted phase-locked response whose polarity is the label, B1, the base and B2 detect it (AUROC > 0.9 / > 0.8), magnitude-only B3 and the five envelopes do not (0.3–0.7), and the phase surrogate and shifted sham remove it (0.3–0.7);
  - in-window features are bit-identical when every sample outside the windows is replaced with 50x noise, and session-filtered features change under the same replacement;
  - session mode is refused on Tiers V and T;
  - standardization uses training rows only;
  - cross-subject pooling uses only shared DKT regions, and B4 raises there;
  - the random-projection control converges to its source kernel;
  - an end-to-end synthetic run of p0, C1, C2 and C3 writes label-free predictions for every arm, chooses and records weights, copies the within-session choice to cross-session, writes proposals, and keeps the ledger chain valid.
- Server smoke test (`python -m classical.smoke --max-shafts 3`; Tier V cell btbank3_0 speech; 300 train and 150 val rows; no test rows), actual output:
  `features: 450 windows x 16 bipolar channels (3 shafts) + 1711 normalization tiles in 38.4 s wall with 1 worker(s)`;
  `fit: dim 960, lam 4.64 (interior), select AUROC 0.9684, out-of-fold train AUROC 0.9656`;
  `prediction file .../results/smoke/preds/V/within/ridge6/btbank3_0__btbank3_0__speech__f0.npz: parts ['select', 'train'], rows [300, 150], select AUROC recomputed from file 0.9684`.
- Server venv: `setup ok: numpy 2.4.6 scipy 1.17.1 sklearn 1.9.1 torch 2.15.0+cu132 cuda visible: False`.
- Server dry runs:
  - `features.sh --sessions eval` exit 0;
  - `c0.sh` exit 2, naming only the three lock grants and the dirty commit;
  - `p0`, `c1`, `c2` and `c3` exit 2, naming the recordings still downloading, the Tier P MAPA kernels, `kept_*.json` and `mapa12` predictions;
  - `c4` and `c5` exit 2, naming `frozen/<regime>.json` (and, for c5, the clean commit);
  - a GPU UUID that does not exist, or a card with too little free memory, exits 3 without starting.
- First real cache sessions match the dry-run assumptions: btbank1_1 has 13,592 windows × 103 bipolar channels, and its cache meta lists no task dataset that failed to build. btbank3_0 has 12,933 × 94 (from `--exact`).

## Not verified yet

- No phase result exists yet. C0 parity (0.683 / 0.673 / 0.585), Tier P numbers and every decision are still to come.
- Fit cost per cell is assumed. The C1 wall estimate (about 11 h) is only as good as that assumption; it will be recalibrated from the C0 timing.
- Session-filtered extraction has run only on synthetic data; its time and memory on real recordings are unmeasured.
- B4, B5 and the cross-session and cross-subject kernels have run only on synthetic data.
- The GPU Gram path (`NPEXP_GRAM_DEVICE=cuda`) has never run; every job so far is CPU.
- Subject 9's onsets are assumed to be sample indices of its 1024 Hz file, as the benchmark uses them. To check when 9_0 is cached.
- The CPU-only units still count against `restore_inference.sh`, which waits on `spark-npexp-*`.
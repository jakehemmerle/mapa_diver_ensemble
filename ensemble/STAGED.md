# Ensemble track: what is staged and what is running

Status on 2026-11-25. The plan is `06-classical-and-ensemble-plan.md` (phases E0–E5). Server root `~/runs/neuroprobe-experiments` (`$NPEXP_ROOT`).

## November 25 decision record (21:15 CST)

- Official Tier P within wave 1: MAPA12 + Ridge6 **0.695847**, Δ −0.000089 vs MAPA12, 5/8 up, exact sign-flip p=0.51171875; Stop. Session 5:0 Δ +0.012789. The MAPA self-ensemble lost 0.006178. The corrected select-CDF analysis did not reverse the earlier Stop.
- DIVER within seed-0 macro was 0.6774. The descriptive +0.0055 for MAPA12 + DIVER was the best of 17 subsets and justified a disclosed post-hoc exception, not an official pass. Only MAPA12, Ridge6 and DIVER are admitted to the restricted decision. E0 must pass for all three with complete OOF parts; the MAPA12 and Ridge6 training parts must be refreshed with the fit-row centering fix before any E2/E3 ladder. CNN stays outside this pool.
- DIVER cross-session quick export hit the 24 GiB memory cap twice. Its retry maps the immutable `/scratch` tensors rather than copying whole sessions. Completion, not launch, establishes recovery. **DIVER cross-subject: not applicable**, because the flattened contact head has no cross-subject protocol; this is neither a score nor a Stop.
- The historical commands and estimates below are staging notes. Freeze only an eligible candidate after both tracks' Tier P decisions. Tier V is a single veto; only a Go unlocks candidate Tier T scoring.

**Completed cross-subject wave 1 (21:40 CST).** MAPA0 + MAPA12: 0.600364 vs MAPA12 0.606481 (Δ −0.006117; 1/7 up; exact p=0.9375). Subject gains for 1/3/4/5/6/8/9: −0.010696, −0.007582, +0.009132, −0.001971, −0.004050, −0.024452, −0.003202. Self-ensemble +0.002054. Leave-one-out contributions: MAPA0 −0.006117 (drop), MAPA12 +0.022248 (keep). Stop; Ridge6 did not trigger wave 2. Session 5:0 −0.001971. No cross-subject ensemble candidate.

**Overnight restart points.** On `gpu`, `results/overnight/selection_<regime>.json` is written only when both tracks are terminal. `server.overnight_freeze` checks the completed classical controls, the authorized ensemble pool and the cross-session inheritance rule. `server.overnight_confirm <regime>` waits for the exact frozen bytes to appear in a clean Git commit and the sync stamp. Confirmation runs are serialized for resources, but each regime waits independently. See `logs/overnight-freeze.log`, `logs/confirm-<regime>.log`, and `results/overnight/{attempt,confirmation}_<tier>_<regime>.json`. Do not remove an attempt receipt or retry scoring on a partial result. A veto never starts T.

**Freeze receipt and deployment:** copy only the terminal eligible regime's `results/frozen/<regime>.json` byte-for-byte into `neuroprobe_experiments/frozen-designs/<regime>.json` and commit it. Never sync over a running experiment's code or `COMMIT.json`. From a clean pinned tree, set `NPEXP_REMOTE_ROOT=runs/neuroprobe-experiments/releases/<full-commit>` for `server/sync.sh` and use that fresh versioned root for the confirmation process. Its `results`, `logs`, `work`, `data_dev`, `.venv` and `src` link to the original server root; the release owns its own `code/COMMIT.json`. Start exactly one confirmation supervisor for the regime after the byte/hash attestation. A missing receipt, mismatched digest, or dirty sync keeps the gate closed. No-candidate branches need no receipt.

**Pre-V quick-head parity:** independent DIVER seed-zero quick and full exports (separate empty prediction roots, `reuse_full=False`) agreed on every member/fused select and eval score and AUROC, maximum difference **0**, on four permitted cells: onset and face count, fold 0, within 1:0 and cross-session 2:1→2:2. Row identities and trainer settings matched; the stacker rejected the quick files in all four. For two cells per regime the measured quick/full times including loading were 30.0/152.6 s within and 131.6/511.4 s cross-session, not a fixed sixfold speedup. The four-cell report is `results/parity/diver_uniform_confirmation.json`. Current confirmation uses the exception only for the frozen uniform DIVER + MAPA12 design. Both baselines still require complete exports. Float32 CNN and DIVER descriptive reports are in `results/ensemble/quick_*.json`; no quick report admits a new E1 member.

For the morning report from `~/runs/neuroprobe-experiments/code`, run `../.venv/bin/python -m analysis.decision_report`. It reads only the ledger, phase reports, freeze receipts and test-access log, and distinguishes pending phases from completed negative results.

## Status in one command

```
ssh gpu 'bash ~/runs/neuroprobe-experiments/code/ensemble/launch/queue.sh status'
```

For each chain it prints the watcher state, the last state line and any `FAILED` lines. For each `spark-npexp-e*` unit it prints the state and the last log line. It also prints the parity verdict, free disk on `/` and on the pool, `MemAvailable`, and GPU memory.

- Chain logs: `logs/queue/<chain>.state`, written as the chain runs.
- Job logs: `logs/<unit without spark-npexp->.log`.

## Queued unattended jobs (started by `queue.sh start`)

Watcher units are named `npexp-queue-<chain>`, outside the `spark-npexp-*` glob. Each job a chain starts is a `spark-npexp-*` unit with its own checks:

- the GPU's free memory against the job's declared VRAM;
- `MemAvailable` of at least the unit's MemoryMax plus 20 GiB;
- the disk floor on the filesystem that holds the scratch path.

A job refused for resources exits 4 (GPU, RAM, CUDA out of memory) or 5 (disk); the chain retries it every 5 minutes. Any other failure stops the chain. Every job writes per cell and atomically, so rerunning a job resumes it.

| Chain | Card | Steps |
|---|---|---|
| card0 | 0 `GPU-013cce7e…` | 1. Parity, then the evaluation-store `cross_subject` kernels. 2. Evaluation-store `within` kernels. |
| card1 | 1 `GPU-ad6718d6…` | Waits for parity to pass. Then Tier P `within`+`cross_subject` kernels on `/scratch`. While `npexp-data-download` is active it runs with `--available-only` every 10 min; after the download ends it runs once more over everything. |
| card3 | 3 `GPU-5b9864f6…` | CNN Tier P export for `within`, `cross_session` and `cross_subject`, three seeds, scheduled against downloads the same way as card1; window caches on `/scratch`. |
| card2 | 2 `GPU-b8d6fafd…` | DIVER Tier P export for `within` and `cross_session`; window caches on `/scratch`. Each regime starts only when MemAvailable ≥ 60 GiB. |
| wave1 | CPU | Waits for chain card1 to finish. Then, per regime (`within`, `cross_subject`): `mapa12`+`mapa0` Tier P export; wait for `ridge6` (classical track) predictions on every cell; E0 check; E1 with its controls. E1 starts only if `code/COMMIT.json` says `dirty: false`. |
| xsess | 1 then 0 | Waits for chains card0 and card1. Then Tier P and evaluation `cross_session` kernels on `/scratch`, `mapa12`+`mapa0` Tier P `cross_session` export, E0 check, descriptive E1. |

Kernel limits:

- **Every chain is on `/scratch`.** `_common.sh` exports `NPEXP_SCRATCH=/scratch/neuroprobe-experiments` and `NPEXP_DISK_FLOOR_GIB=100`; all kernel jobs run `--max-held 2 --disk-floor 100`.
- **Held features:** card0 and card1 each hold one session's per-contact features at a time (within and cross_subject plans visit then delete), so together they stay within two. xsess holds two at once and runs only after both.
- **Mount check:** `/scratch` is not mounted again after a reboot. `ensemble/paths.py` refuses (exit 6, the chain stops) when `NPEXP_SCRATCH` resolves onto `/`; `common/mapa_features.py` refuses a kernel store that is not ext4 on its own mount. Neither falls back to the system disk.

## Commands for later phases (Main runs these)

All `--dry-run` first; each refuses with a message naming what is missing.

```
cd ~/runs/neuroprobe-experiments/code/ensemble/launch
# Cross-session kernel stores: blocked until the scratch disk is mounted and spark-neuroprobe-mapa-board has ended
NPEXP_SCRATCH=<mount> NPEXP_DISK_FLOOR_GIB=100 bash mapa_kernels.sh GPU-013cce7e-9628-49c2-f849-251f3304d038 --store E --regimes cross_session --skip-parity --max-held 2
NPEXP_SCRATCH=<mount> NPEXP_DISK_FLOOR_GIB=100 bash mapa_kernels.sh GPU-ad6718d6-1cf2-f3e7-2f2b-515f9c00859f --store P --regimes cross_session --skip-parity --max-held 2
bash e0.sh cpu --wave 1 --tier P --regime cross_session            # after the P cross_session kernels
bash e1.sh --regime cross_session --wave 1                          # descriptive (two subjects)
# Wave 2 enters the decision only per the plan's trigger (printed by E1)
bash e0.sh cpu --check-only --wave 2 --tier P --regime within      # after diver/cnn exports; checks every wave-2 member
bash e1.sh --regime within --wave 2
bash e2.sh --regime within --wave 1      # fusion ladder + shuffled-stacker control
bash e3.sh --regime within --wave 1      # LOMO + self-ensemble + subject concentration; writes the ensemble proposal
bash freeze.sh --regime within           # one candidate per regime, from both tracks' proposals
bash e0.sh cpu --wave 1 --tier V --regime within --export-only   # members of the frozen design on Tier V (diver/cnn: wave 2 + GPU)
bash e4.sh --regime within               # Tier V veto, one pass
bash e0.sh cpu --wave 1 --tier T --regime within --export-only
bash e5.sh --regime within               # Tier T behind results/LOCK.json, once
bash reproduce.sh parity --regime within                 # no test label: penalties/val AUROC vs the reproduction's ws_/csession_ results
bash reproduce.sh reproduce --regime within --member mapa12   # official-protocol test scores, LOCK grant purpose "reproduction"
```

Replace `within` with `cross_subject` and `cross_session` where they apply. The cross-session regime takes the within-session choice: E3 on `within` writes both proposals.

## Storage

- **Kernel store:** `/scratch/neuroprobe-experiments/kernels/` (ext4 on the provisioned NVMe, Jake's directive; it was on the ZFS pool until 16:50 and nothing had been written there). `work/kernels` is a symlink to it.
  - Every read and write checks that the path is on an ext4 mount other than `/` and fails with `KernelStoreUnavailable` otherwise; there is no fallback.
  - Files are written whole (temporary file, then rename) and read whole; nothing is memory-mapped there.
  - Estimated size from the dry runs: evaluation store 24 GiB `cross_subject` + 15 GiB `within`; Tier P about 20 + 32 GiB; cross-session about 35 GiB (evaluation) and 64 GiB (Tier P).
- **Scratch:** `/scratch/neuroprobe-experiments` holds per-contact features (~50–72 GiB per session), our prepared sessions and the DIVER/CNN window caches.
  - Floors are measured on the filesystem that holds the path: 100 GiB on `/scratch`, at least 250 GiB on the system disk.
- **Small outputs:** predictions, ledger and logs go under `results/` and `logs/`.

## Reused reproduction caches (read-only)

- `work/features/mapa_cs/enc_s*_t*_mapa_cs.pt`: region features, 11 sessions; read memory-mapped, label-free fields only.
- `work/prepared/s*_t*`: band caches of 11 evaluation sessions. `s2_t0` is prepared by us.
- `work/cnn/s*_t*`: CNN windows of the 12 evaluation sessions.
- `results/cs_*.json` for parity.

## Out-of-fold design for trained members

- Seed 0 trains the full model plus five models on the purged out-of-fold blocks. Its file `<member>` carries train (out-of-fold), select and eval.
- Seeds 1 and 2 train the full model only (`<member>@s1`, `<member>@s2`; select and eval) and are used for the reported seed spread.
- That is 8 fits per cell instead of 18; the stacker uses seed 0.
- The only thing chosen on select is the stopping epoch by AUROC:
  - DIVER: patience 10, at most 40 epochs.
  - CNN: patience 10, at most 100 epochs.
  - Both are recorded in each prediction file's meta.
- DIVER has no cross-subject protocol (a flattened per-contact head). Its cross-session uses contacts shared by label, which is our extension; the published entry is within-session only.

## Parity criterion (MAPA kernels vs the reproduction, cross-subject 3:0, validation quantities only)

Pass requires (b) for every fit and (a) or (c) for every fit:

- (a) with the penalty forced to the reproduction's choice, every validation AUROC agrees within 1e-4;
- (b) where our chosen penalty differs, the readout's validation AUROCs at the two penalties differ by less than 2e-3. Given (a), these equal the reproduction's own values within 1e-4.
- (c) a fit that fails (a) is accepted only if the same fit's value at the reproduction's penalty moves by at least its gap when nothing but dtype or device changes (this device float32, CPU float32, CPU float64 Gram products), the gap is under 2e-3, and at most 2 of the 30 fits need (c). The job measures the three values itself and writes them to the JSON (`rows[].sensitivity`) and the ledger. Added by Main's ruling as a pre-decision amendment.

If the check fails, the job stops and writes no kernels. The verdict, with every fit, is written to `results/parity/mapa_kernels_cs_S3T0.json` and the ledger.

First attempt, under the old rule (gap ≤ 1e-3 at our own chosen penalty):

- Tap 0: max gap 3.3e-5; penalty equal in 14 of 15 fits.
- Tap 12: face_num gap 1.06e-3, where the chosen penalty differed (a near-tie); the other tap-12 tasks matched exactly or to about 1e-6.
- 1 of 30 penalties differed in total, and the run stopped.

Second attempt, criterion (a)+(b), 16:35: 28 of 30 fits within 3e-6 at the reproduction's penalty; face_num failed (a) on both taps and the run stopped.

| face_num, val AUROC at penalty 1e-4 | tap 0 | tap 12 |
|---|---|---|
| Reproduction (`cs_S3T0.json`) | 0.503975 | 0.473793 |
| GPU float32 (the job) | 0.503656 (gap 3.19e-4) | 0.474853 (gap 1.06e-3) |
| CPU float32 (`work/diag/parity_dtype.py`) | 0.504145 (gap 1.70e-4) | 0.472057 (gap 1.74e-3) |
| CPU float64 | 0.503976 (gap 1.3e-6) | 0.473007 (gap 7.86e-4) |
| Spread from dtype/device alone | 4.89e-4 | 2.80e-3 |

The cell is ill-conditioned at the grid's smallest penalty: 3 regions, val AUROC near chance. The reproduction (MAPA `readout._lam_grid`) solves the primal when feature width is below the training rows, with float32 in-place standardization in 1024-column blocks; this job solves the dual from stored float32-product Gram kernels. This is a declared numerical deviation.

## Measured figures

| Figure | Value | Source |
|---|---|---|
| Local tests | `76 passed in 57.94s` | `pytest ensemble/tests`, CPU, synthetic data |
| Server tests | `test_mapa_kernels.py`: 10 passed in 16.95s | an earlier version of the file |
| Server dry runs (most of the time is Neuroprobe building split tables; every job start pays it) | evaluation `cross_subject` 4m56s, exit 0; evaluation `within` 3m21s, exit 0; CNN Tier P `within`/`cross_session`/`cross_subject` exit 0 | server runs |
| MAPA encode, measured | ≤ 3.1 GiB VRAM, ~12 min per session with per-contact taps | the reproduction's logs |
| MAPA encode, estimated | 51–72 GiB written per session (window counts from our dry run) | estimate |
| Kernel accumulation | ~4.8 GiB VRAM, ~11–14 min per within session | estimate; assumes 1 TFLOP/s, not measured; first real numbers will be in `logs/ek-mapa-E.within.log` |
| Declared VRAM (launcher checks) | kernels 6144 MiB (from 3.1 GiB measured + accumulator estimate); CNN 3072 MiB (2.6 GiB measured by the reproduction); DIVER 6144 MiB (not logged by the reproduction) | as stated |
| MemoryMax | kernels 24G (prepare+encode peaked at 23–27 GiB in the reproduction's units); DIVER 24G; CNN 16G | caps, not measurements |
| Wall time, evaluation `within` | 5.2 h | dry-run estimate |
| Wall time, CNN Tier P | 2.8 h `within` and 8.3 h `cross_session` for the recordings present at dry-run time | per-fit times from the reproduction's file timestamps: 7.3 s `within`, 22 s cross-session, 4 s cross-subject |
| Wall time, DIVER | ~8.8 s per fit | the reproduction: 30 fits in 4.4 min |

## Disclosures a submission must carry

- MAPA's features carry its session-level preprocessing: filtering and STFT on the continuous session, which reaches 250 ms before onset. They also carry median/MAD normalization over the whole session and MAPA's per-session contact exclusions (Guard 1). Session 5:0 has no exclusion list; its feature files carry `no_mapa_exclusion_list`.
- MAPA and DIVER-1 are used as released.
- Readout standardization uses training rows only.
- The ridge penalty is chosen per cell on validation AUROC over MAPA's 25-point grid.
- Trained members early-stop on validation AUROC (settings above).
- Fusion weights are never chosen on validation labels.
  - The stacker penalty is fixed at 0.01.
  - Held-out scores use each member's select-part CDF; OOF stacker inputs use that member's OOF-part CDF. Neither reference uses evaluation labels.

## Precision (Jake's ruling, November 25)

- MAPA and DIVER use bf16 autocast with float32 master weights and optimizer state:
  - MAPA encode: bf16 autocast forward, as the reproduction did.
  - DIVER: encoder and the head's training forward under bf16 autocast (DIVER's `use_amp`); the predict path uses the same autocast since commit after `d1ffec4`.
- CNN alone uses float32 after the matched Tier P precision diagnostic (1:0 onset fold 0: select AUROC 0.742369 bf16 vs 0.922972 float32). Old bf16 outputs are isolated under `/scratch/neuroprobe-experiments/stale/cnn-bf16`; they are never pooled with the regeneration. Prediction metadata records the precision.
- Kept in float32/float64 by the ruling: Gram products (float32 products, float64 accumulation, TF32 off) and the CPU float64 ridge solve.

## Historical pre-launch verification limits (superseded by the decision record above)

- No GPU path has been run by this code: session encode, GPU kernel accumulation, the DIVER and CNN adapters. The kernel time per session is an estimate.
- No real-data E0/E1 has been run; fusion, stats, controls and E1–E5 are verified on synthetic predictions only.
- DIVER adapter: it imports DIVER-1 from source and was not run here.
- Subject 9 (1024 Hz): the resampling to 2048 samples in the DIVER and CNN adapters is untested.
- The `--wait` exit-status propagation of `systemd-run` and the queue chains were not exercised before launch.
- Restore-inference interaction: `restore_inference.sh` watches `spark-npexp-*` units. While a chain waits between jobs (for recordings, memory, ridge6), no such unit may be active, so inference could restart and take about 56 GiB per card. Our jobs would then be refused (exit 4) and retried every 5 minutes until the card is free.
- Not yet checked: the within-session comparison against the reproduction's first finished within-session pair. Its results did not exist yet; run `reproduce.sh parity --regime within` when `ws_S*.json` appear.
- Cross-session kernel stores are queued in chain xsess (after card0 and card1), on `/scratch`.

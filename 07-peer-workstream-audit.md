# Audit of the other Neuroprobe workstream (reproduction + cross-subject improvements)

November 25, 2026, read between 15:27 and 15:35 CST. This audit was read-only. It made no writes on the server, used no GPU and changed no units. `R` means `gpu:/home/jake/runs/neuroprobe-reproduction`. Script line numbers refer to `~/ws/code/personal/spark/experiments/neuroprobe/`. Every shared script there has the same md5 as its server copy, so those line numbers match the code that is running. Server-only launchers (`cnn-board.sh`, `diver-sweep.sh`, `pilot.sh`, `score-pilot.sh`) were read on the server. **[INF]** marks inference. Everything else comes from a cited log, result file or command output.

## 1. Status of everything running or finished

GPU map (`nvidia-smi`, 15:27:59): index 0 = `GPU-013cce7e…`, 1 = `GPU-ad6718d6…`, 2 = `GPU-b8d6fafd…`, 3 = `GPU-5b9864f6…`. Each card has 64 GiB. `gpu-inference.service` is in the **failed** state and has been off since 10:59:13, so every card now has about 57 GiB free.

| Unit / experiment | What it is | GPU (index, memory) | Started | Progress | Measured throughput | Projected finish |
|---|---|---|---|---|---|---|
| `spark-neuroprobe-popt-board0` | PopT full board: public PopT + BrainBERT weights, 1,000-step fine-tune per fold (`popt_board.py`) | 0, 7,584 MiB | 13:39:44 | Shared queue, all four workers: **75 of 690 folds** (60 of 510 job files) at 15:29 (`R/results/popt-official/*/*.json`) | Four workers: 45 folds from 14:12 to 15:00 = **0.94 folds/min**, about 4.3 min per fold per worker (file mtimes). Preparing each new recording's STFT cache takes 11–15 min (`R/work/popt-stft` lock vs directory mtimes). All workers sat idle from 15:00:00 to 15:15:51 while recording 1_1 was prepared. | 615 folds ÷ 0.94 = 656 min ≈ 10.9 h. Seven recordings are still unprepared (2_0, 4_0, 4_1, 7_0, 7_1, 10_0, 10_1); about 4 idle gaps × 15 min adds ≈ 1 h. Finish ≈ **03:30 Nov 26** [INF; fold time scales with window count] |
| `…popt-board1` | same queue | 1, 7,724 MiB | 13:39:44 | (shared) | (shared) | (shared) |
| `…popt-board2` | same queue | 2, 6,978 MiB | 14:12:04 | (shared) | (shared) | (shared) |
| `…popt-board3` | same queue | 3, 7,038 MiB | 14:10:55 | (shared) | (shared) | (shared) |
| `spark-neuroprobe-mapa-board` | MAPA reproduction sweep (`mapa_sweep.py`). Cross-subject is done; it is now on per-contact within-session and cross-session | Assigned card 2. It holds GPU memory only while encoding; it held none at 15:28 (CPU readout) | 12:47:15 | Cross-subject **10 of 10 targets, 150 of 150 cells** (`R/results/cs_S*T*.json`, last at 14:02:15). Within-session **0 of 12** shards and cross-session **0 of 12**. Shard `ws 0` (session 1_1) has run on CPU since 14:15:38 and prints no progress. Per-contact encodes: 1 of 12. | Per-contact encode: 12 min and 74.6 GB per session (`logs/mapa-board.log`; `work/features/mapa_contact/enc_s1_t1_mapa_contact.pt`). The within-session readout had written 1.109 TB to disk after 78 min (`/proc/2243347/io`, **11.8 GB/min** sampled over 60 s), with 58.9 GB RSS (peak 64.5 GB) and about 190% CPU. It uses file-backed fp32 copies of 16.6 GB (train) and 8.3 GB (val, test) per task-fold (`mapa_cpu.py:24-35`, `/proc/…/fd`). | Each task-fold writes about 33–66 GB; 30 task-folds per session gives 1.0–2.0 TB, so shard `ws 0` is 55–100% done and one within-session shard takes about 80–145 min [INF]. Cross-session trains on the sibling's full task set, so assume it costs at least as much [INF]. 24 shards × 80–145 min + 11 encodes × 12 min ≈ 34–60 h. Finish between **≈ 01:30 Nov 27 and ≈ 03:30 Nov 28**, with roughly 24–48 TB written to the root NVMe [INF]. |
| `spark-neuroprobe-temporal-dev` | Temporal filter-bank test (`mapa_temporal.py`), CPU only (no `CUDA_VISIBLE_DEVICES`, no process in `nvidia-smi`) | none | 14:55:20 | **Finished 15:32:45**: 6 of 6 sessions, 90 of 90 cells (`R/results/improvements/mapa-temporal/S*.csv`). Unit inactive at 15:34:56. | 90 cells in 37.4 min ≈ 6.2 min per session (385% CPU, 15.5 GB RAM) | done |
| multiscale validation | `mapa_multiscale.py`, CPU | none | ≈14:21 | Finished 14:58: 150 of 150 cells (`R/results/improvements/mapa-multiscale-validation/`) | ≈ 3.7 min per session | done |
| parcel-transfer pilot | `mapa_parcel_transfer.py`, CPU, session 3_0 only | none | ≈14:30 | Finished 14:33: 15 tasks × 4 arms (`…/mapa-parcel-transfer/S3T0.csv`) | — | done |
| CNN board | `cnn_benchmark.py`, run by `cnn-board.sh` (GPU 0) and two `cnn_extra.sh` workers | none now | 11:08 | Finished 13:16:53: 510 of 510 files (`R/results/cnn-official/`) | 2.1 h for the board | done |
| DIVER board | `diver_pilot.py` run by `diver-sweep.sh`, two sequential passes | none now | ≈11:08 | Finished 13:53:06: 180 of 180 files (12 sessions × 15 tasks) | 13–14 min per session (result mtimes 11:21 → 11:35 → 11:48 …) | done |
| `spark-restore-inference-after-neuroprobe` | `restore_inference.sh` watcher | none | 15:18:04 | polling | every 30 s | Fires when the last unit exits; see below |
| Stimulus-supervised adapter | proposed (`neuroprobe-architecture-research.md:54`) | — | not started; no script found locally or on the server | — | — | — |

**What happens when the last unit exits.**
- `restore_inference.sh:4-11` loops while any unit matching `spark-neuroprobe-*.service` **or `spark-npexp-*.service`** is active, activating or deactivating.
- When none is left, it runs `sudo -n systemctl start gpu-inference.service` (line 12) and polls `127.0.0.1:8000/health` for up to 20 min (lines 13-21).
- Our units are named `spark-npexp-*`, so any unit of ours that is still running also holds inference off.
- With today's units, `mapa-board` is the long pole: about 1.5–2.5 days [INF]. Stopping PopT alone does not bring inference back.
- Their notes measured inference at about 56.4 GiB per card, leaving 7.5 GiB free on card 2 (`research/neuroprobe-reproduction.md:7`). After restoration each card has about 7–7.5 GiB free. I did not measure this, because inference is off now.

## 2. Reproduction scorecard

Source: `R/results/paired-results.csv` (15:17, 1,268 rows), written by `paired_results.py`. Each row pairs one observed fold with the published fold from the leaderboard snapshot at `b282acd`. Macros below are means over the paired cells.

| Entry, regime | Cells paired | Published macro (same cells) | Observed macro | Mean Δ | Median \|Δ\| | 90th pct \|Δ\| | Max \|Δ\| | Cells \|Δ\|>0.01 | Per-subject mean Δ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| MAPA, cross-subject | 150 of 150 | 0.6083 | 0.6082 | −0.0000 | 0.0000 | 0.0001 | 0.0058 | 0 | all within ±0.0002 | **Exact.** 146 of 150 cells are within 0.001. The pilot's pitch differs by −0.0027 (`R/results/pilot-comparison.csv`; `mapa_sweep.py:38-39`). Baseline-grade. |
| MAPA, within-session | 0 of 360 | 0.6953 | — | — | — | — | — | — | — | In progress |
| MAPA, cross-session | 0 of 180 | 0.6909 | — | — | — | — | — | — | — | In progress |
| DIVER-1 tiny frozen, within-session | 360 of 360 | 0.6777 | 0.6792 | +0.0015 | 0.0052 | 0.0168 | 0.048 | 104 | +0.005 (1), +0.001 (2), −0.000 (3), +0.000 (4), +0.001 (7), +0.002 (10) | **Good** at macro and subject level; cells scatter (SD 0.011). Usable as member M4. |
| CNN Laplacian, within-session | 360 of 360 | 0.6686 | 0.6694 | +0.0008 | 0.0123 | 0.0499 | 0.245 | 206 | −0.005 (1), +0.002, −0.003, **+0.011 (4)**, −0.001, +0.001 | Macro good, cells noisy (SD 0.036). The largest cell is 4_0 speech fold 0: published 0.592, observed 0.837. |
| CNN, cross-session | 180 of 180 | 0.6704 | 0.6666 | −0.0038 | 0.0086 | 0.0410 | 0.142 | 82 | **−0.013 (4)**, the others −0.005 to +0.000 | Macro within 0.004 |
| CNN, cross-subject | 150 of 150 | 0.5777 | 0.5748 | −0.0028 | 0.0108 | 0.0284 | 0.055 | 84 | +0.002 to −0.007 | Macro within 0.003 |
| PopT, within-session | 32 (2 sessions) | 0.7221 | 0.6611 | **−0.061** | 0.048 | 0.139 | 0.242 | 28 | −0.041 (1), −0.062 (3) | **Does not reproduce** |
| PopT, cross-session | 16 | 0.7245 | 0.6687 | **−0.056** | 0.050 | 0.118 | 0.185 | 15 | −0.053, −0.056 | Does not reproduce |
| PopT, cross-subject | 16 | 0.6430 | 0.5493 | **−0.094** | 0.071 | 0.200 | 0.298 | 15 | −0.156 (1), −0.090 (3) | Does not reproduce |

- **The 0.065 CNN speech fold.** `research/neuroprobe-results/cnn-public-runner-speech.csv` shows −0.0649 on 3_0 speech fold 1 with the upstream example runner. The workstream's own `cnn_benchmark.py` places the official validation split at the classifier's internal validation boundary (`cnn_benchmark.py:99-103`). With that change the same fold gives −0.0147 (seed 42), −0.0041 (seed 43) and −0.0151 (seed 44) (`paired-results.csv`). The gap is mostly closed. The remaining ±0.01 is seed-level noise.
- **Trustworthy as baselines or members:**
  - MAPA cross-subject: exact.
  - DIVER within-session: macro +0.0015, every subject within +0.005.
  - CNN on all three regimes at macro level only: macro within 0.004, per-subject up to ±0.013, cells unreliable.
- **Not trustworthy:** PopT. On the 64 paired folds it is off by −0.04 to −0.16 per subject. It is also excluded as a member under Q5.
- **No per-trial predictions.** No reproduction stores per-trial scores. DIVER stores only `test_auroc` (`diver_pilot.py:124-129`), CNN only the AUROC (`cnn_benchmark.py:104-110`), PopT only `test_roc_auc` (`popt_board.py:169-170`), and MAPA only its readout JSON. Our E0 export therefore needs reruns of every member.

## 3. Their improvement experiments

All three share one scorer, `heldout_auc` (`mapa_multiscale.py:23-30`):
- Closed-form ridge over MAPA's 25-point grid (`LAM_MULTS`, `readout.py:78`), fit on all finite labels of anchor 2_4.
- The penalty is chosen on the **earlier half** of the target session's official cross-subject validation split (line 28-29) and scored on the **later half** (line 30).
- Standardization uses the anchor's training statistics (`mapa_multiscale.py:14`, calling MAPA `readout._standardize_inplace`, `readout.py:108-130`).
- Kernels are trace-normalized (line 17-20).
- Features come from the frozen region-mean cache `R/work/features/mapa_cs` (frontend `enc0` and block-12 `enc12`).
- No test labels are read.

Their advancement bar is a mean validation gain of at least +0.005 and gains in at least 3 of 5 participants (`neuroprobe-architecture-research.md:43`). Ours for this regime: Tier V cross-subject Go needs Δ ≥ +0.010 and at least 4 of 5 subjects up; Tier P cross-subject Go needs Δ ≥ +0.015 and at least 6 of 7 (`06-classical-and-ensemble-plan.md:50-56`). The noise floor in `03-lineage-survey.md:9-13` is a cross-subject subject-level paired SE of 0.0102 between two different models. A new #1 needs about +0.025.

### 3.1 Multiscale: frontend + final-layer blend (`mapa_multiscale.py`)

- **Hypothesis.** MAPA's spectrogram frontend features add information to the final-layer readout (`neuroprobe-competition-plan.md:26`; `architecture-research.md:47`).
- **Protocol.**
  - All 10 cross-subject targets, which includes confirmation subjects 7 and 10 (`CS_TEST_CELLS`, line 41).
  - Baseline: `enc12` kernel. Candidate: `(K_enc12 + K_enc0) × 0.5`, a fixed equal weight (lines 64-68).
  - Penalty selected separately for each arm on the earlier half of validation.
  - The cut is the plain median row of the time-sorted validation set, with **no purge** (lines 58-61).
  - No control arm.
- **Results** (`R/results/improvements/mapa-multiscale-validation/*.csv`, `logs/multiscale-validation.log:151-152`):
  - Mean of subject means **−0.00522**; 48 of 150 cells positive; median cell −0.0070.
  - Per subject: 1: −0.0103, 3: −0.0107, 4: −0.0020, 7: +0.0010, 10: −0.0041.
  - Development subjects (1, 3, 4) −0.0077; confirmation subjects (7, 10) −0.0016.
  - Tasks: face_num +0.018 and local_flow +0.002 rise; onset −0.019, speech −0.016 and word_index −0.016 fall.
- **Bars.** It fails theirs (mean < 0 and 1 of 5 up; the script prints `Advance: False`) and fails ours.
  - Exact one-sided sign-flip over the 5 subjects: p = 0.97.
  - Subject SD of the paired Δ is 0.0051, SE 0.0023. The loss sits about 2.3 SE below zero and is half the size of 03's between-model floor.

### 3.2 Parcel-transfer pooling (`mapa_parcel_transfer.py`)

- **Hypothesis.** One coefficient vector applied to a parcel-mean branch lets the target's parcels that the anchor lacks contribute (`architecture-research.md:5-11`).
- **Protocol.**
  - Target 3_0 only, because it has the sparsest overlap: 27 of 100 contacts sit in matched parcels (`research/neuroprobe-results/parcel-coverage.csv`). The CLI also accepts subjects 1, 3 and 4 (line 76).
  - Second kernel: the per-window mean of 384-dimensional block-12 vectors over "matched" or "all" mapped parcels, on the anchor side and on the target side separately, giving 4 arms (lines 11-21, 58-67).
  - Branch weights fixed at 0.5 (line 66). Standardization uses anchor statistics (line 59).
  - 2 s purge on both sides of the validation cut (lines 50-53).
  - The matched/matched arm is the "shared regularization" control the design calls for.
- **Results** (`…/mapa-parcel-transfer/S3T0.csv`; baseline mean 0.6489):

  | Anchor pool → target pool | Mean Δ (15 tasks) | Tasks up | SE over tasks |
  |---|---|---|---|
  | matched → matched (control) | −0.0061 | 5 | 0.0028 |
  | matched → all | −0.0056 | 5 | 0.0039 |
  | all → matched | −0.0022 | 6 | 0.0059 |
  | all → all (hypothesis) | −0.0084 | 8 | 0.0082 |

  In the all → all arm, local_flow (−0.094) and global_flow (−0.056) fall while onset, speech and frame_brightness each rise by about +0.018.
- **Bars.** Every arm is negative, and one session is a single subject, so it fails both bars. The design's own decisive comparison (all → all against matched → matched) is −0.0023, inside noise.

### 3.3 Temporal filter bank (`mapa_temporal.py`)

- **Hypothesis.** Shift-tolerant temporal motifs in MAPA's mid and fast token sequences transfer across subjects (`architecture-research.md:15-21`).
- **Protocol.**
  - Development sessions only: 3_0, 1_1, 4_0, 3_1, 1_2, 4_1 (`DEV_CELLS`, line 14). Subjects 7 and 10 were not opened.
  - Fixed-seed random 384 → 16 projections and 64 three-tap filters with dilations 1, 2 and 4, applied to the mid (16-token) and fast (32-token) bands. The summary statistics are max and fraction-positive, 256 features per parcel (lines 17-26, 29-55).
  - Controls run in the same pass: `width_one` (centre tap only) and `shuffled` (fixed token permutation) (lines 13, 44-52).
  - Candidate: `(K_enc12 + K_variant) × 0.5` (line 100). 2 s purge (lines 85-89).
- **Results** (all 90 cells, `…/mapa-temporal/S*.csv`; baseline mean 0.6318):

  | Arm | Mean Δ (subject means) | Subject 1 | Subject 3 | Subject 4 | Cells up |
  |---|---|---|---|---|---|
  | ordered | **−0.0484** | −0.0434 | −0.0689 | −0.0328 | 15 of 90 |
  | width_one (control) | −0.0471 | −0.0437 | −0.0662 | −0.0312 | 16 of 90 |
  | shuffled (control) | −0.0437 | −0.0444 | −0.0607 | −0.0261 | 17 of 90 |

  - Ordered minus width_one: −0.0013 (42 of 90 cells up). Ordered minus shuffled: −0.0046 (43 of 90).
  - 14 of 15 tasks fall; only frame_brightness rises (+0.042).
- **Bars.** It fails both bars decisively (sign-flip p = 1.0; subject SE about 0.010). The controls lose as much as the ordered arm. The loss therefore comes from adding a weakly informative kernel at half weight, not from temporal order.

### 3.4 Stimulus-supervised adapter

Not started. The plan stages raw voltage for sessions 1_0 (*Fantastic Mr. Fox*) and 2_2 (*Guardians of the Galaxy*), trains a linear adapter on permitted recordings, and runs random-projection and shifted-teacher controls (`architecture-research.md:54`).

## 4. Protocol check against `05-rules-verdicts.md`

| Experiment | Use of evaluation-session labels or signal (code fact) | Ruling |
|---|---|---|
| All three improvements | Earlier half of target validation labels picks one ridge penalty per arm (`mapa_multiscale.py:28-29`) | Q3a: **allowed** (single penalty, fixed grid) |
| All three | Later half of target validation labels scores the arms, and the arm is kept or dropped on that score ("Set aside those configurations", `architecture-research.md:51`; `mapa_multiscale.py:84`) | Choosing between designs on evaluation-session validation labels: **not cleared**. No merged-entry precedent; Q3c logic; our rule 2 (`06…:19`) puts every design decision on Tier P. |
| All three | Standardization fit on anchor training rows and applied to the target (`mapa_multiscale.py:14`; `mapa_parcel_transfer.py:59`) | Q6a′: **compliant** (training statistics only) |
| All three | Kernel weights fixed at 0.5 (`mapa_multiscale.py:67`; `mapa_parcel_transfer.py:66`; `mapa_temporal.py:100`) | Q3c not triggered (weights not chosen on validation) |
| All three | Input features are MAPA's whole-session filtered, median/MAD-normalized frontend with Guard 1 exclusions (`prepare_session.py:1`, `SessionRobustZNormalizer`) | Q4b/Q4c/Q6a: **allowed with disclosure** |
| Multiscale | Ran on subjects 7 and 10 (`mapa_multiscale.py:41`), the subjects their own protocol reserves for confirmation (`architecture-research.md:42-43`) | Confirmation subjects opened during development. No rules issue, but it spends the held-out look. |
| Multiscale | Unpurged validation cut (`mapa_multiscale.py:58-61`) | Below our Tier V standard (2 s purge, `06…:34`) |
| Parcel transfer | "all" pooling averages target parcels the anchor lacks, per window, unlabelled (`mapa_parcel_transfer.py:11-21`) | Not a fitted alignment, so Q6b is not triggered. It matches the cleared atlas-pooling family (`05…:205`). |
| Temporal | Fixed-seed filters; nothing fit on the target (`mapa_temporal.py:17-22`) | Compliant |
| Their development plan as a whole | Development on evaluation subjects 1, 3, 4 (validation labels), then confirmation on evaluation subjects 7, 10 (`architecture-research.md:42-43`) | No permitted-session tier. Every design screen consumes evaluation validation labels. Under our rules this is Tier V used for selection, which is **not cleared**. |
| CNN reproduction | StandardScaler fit on training rows (`cnn_benchmark.py:96`); early stop on official validation (`cnn_benchmark.py:99-103`) | Q3b, Q6a′: compliant |
| DIVER reproduction | Early stop on `val_auroc`, patience 10, 40 epochs (`diver_pilot.py:86-87`); whole-session 0.5 Hz high-pass, 60/120/180 Hz notch and ±200 clip (`diver_pilot.py:67-73`) | Q3b compliant. Q4b: the high-pass has the same type and cutoff as MAPA's, so it is allowed with disclosure. |
| PopT reproduction | Public PopT and BrainBERT weights (`popt_board.py:128, 235`); checkpoint chosen on validation every 100 steps (`popt_board.py:161-165`); normalization fit on training windows (`popt_board.py:201`) | Q5: **not allowed** as a member or in an entry. Fine as a reproduction. |
| MAPA reproduction | Published readout; validation picks the penalty | Q3a: compliant |
| Stimulus adapter (plan) | Annotations of 1_0 and 2_2 | Q1a: **allowed with disclosure**. Both films are absent from evaluation. It must also exclude 6_0 and the partial sessions (Q1b). Their plan excludes subjects 7 and 10 entirely (`architecture-research.md:27`), which covers the partials. |

**What would have to change.**
- **Design screening.** Screen each design on Tier P: within-session on the 13 permitted sessions, and cross-subject with anchor 2_6 on the other permitted sessions. Freeze one candidate per regime. Use the 10 evaluation-session cross-subject validation splits once, as a veto.
- **Reusable scorer.** Their scorer with the 2 s purge is identical to our Tier V construction and can be reused.
- **Multiscale.** Also needs the purge.

**Can the existing results still be used?**
- They cannot count as Tier P evidence. They also cannot count as a valid Tier V pass, because no candidate was frozen from Tier P.
- They are now **already-seen evaluation-validation looks** and must be recorded in our ledger as such (rule 9).
- Any later Tier V check of a frontend + final-layer blend in cross-subject is a second look at the same labels and must be disclosed as one.
- Re-running under our rules would produce new evidence on Tier P. The old numbers remain descriptive only.

## 5. Overlap and complementarity

| Area | Theirs | Ours | Relation |
|---|---|---|---|
| MAPA frontend + last-layer combination | Multiscale blend, cross-subject, eval validation: −0.0052 | M1 + M2 members; C2 stacks blocks beside the last layer and frontend; base kernels at equal fixed weights (`06…:82`) | **Same mechanism** in a different regime. Their result is the first measurement of equal-weight frontend+final in cross-subject, and it is negative. Within-session is untested. |
| Extra-kernel dilution | Temporal: any 256-d random-feature kernel at weight 0.5 costs 0.044–0.048 in cross-subject, with controls equal | Extra-kernel control (random projection of frontend features) | **Direct evidence for our control's premise.** With equal fixed weights, MAPA's share falls to 1/k as kernels are added, and an uninformative kernel can cost far more than the +0.002 keep bar. Their numbers are cross-subject only; within-session dilution is unmeasured [INF]. |
| DIVER-1 frozen | Reproduced, all 360 within-session cells, +0.0015 | M4 | **Input we need:** validated code (`diver_pilot.py`) and settings. Missing: per-trial scores and permitted sessions. |
| CNN Laplacian | Reproduced, all 690 cells, macro within 0.004 | M5 | Same. The cache `R/work/cnn` (39 GB, 12 eval sessions) serves Tier V/T only. |
| MAPA reproduction | Cross-subject exact; within/cross-session in progress | Baseline is the better of MAPA and Ridge 6-band per regime | **Input we need:** parity on the per-contact path checks our `common/mapa_features.py`. Two sessions of within-session plus one cross-session pair would suffice [INF]. |
| Feature caches | `R/work/features/mapa_cs` (84 GB, region-mean enc0+enc12, 11 sessions); `R/work/prepared` (25 GB MAPA frontend band caches); `R/work/popt-stft` (32 GB) | Our Tier V/T cross-subject arms need the mapa_cs features; Tier P needs new downloads | Reusable read-only for Tier V/T. PopT STFTs are useless to us (Q5). |
| Regime focus | Cross-subject only, anchor 2_4, eval sessions | Within-session first, all three regimes, Tier P | **Genuinely different.** Cross-subject has the most headroom (0.608) but the weakest statistics (5 subjects). |
| Cross-subject alignment | Parcel pooling (atlas, unfitted) | L10 functional alignment (later lineage) | Their negative pilot is weak evidence against atlas-coarsening variants of L10. CCA-style fitted alignment is untested. |
| Stimulus supervision | Adapter plan on 1_0 and 2_2 | L4 (later lineage) | **Same lineage.** It needs the same permitted-session downloads as our Tier P (1_0 and 2_2 are both in our Tier P list). |
| Validation scorer | `heldout_auc` with purge | `common/splits.py` Tier V | Identical construction; code can be shared. |

## 6. Strength comparison of all directions

| Rank | Direction (owner) | Prior evidence it will help | Gain needed to matter | Cost to a decisive result | Rules status | Regime |
|---|---|---|---|---|---|---|
| 1 | Classical stack on MAPA (ours) | Indirect but specific: waveform entries beat MAPA on word_index (+0.017); linear on raw voltage reaches 0.742 on word_index; Ridge 6-band wins both subject-7 sessions (`03…:15-27`). The dilution in §5 is a risk to the equal-weight design. | Tier P within-session Go +0.010 with 7 of 8 subjects up | About 2 GPU-h + 20 CPU-h, plus the Tier P download (not on server) | Cleared (in-window, single penalty) | Within-session, cross-session |
| 2 | Ensemble MAPA/Ridge/DIVER/CNN (ours) | Weak: the best-per-task ceiling is only +0.004 (`03…:14`). DIVER and CNN now reproduce at macro level (§2), so the members are real. | E1 screen +0.005 over best member; Go +0.010 | Wave 1 is CPU-only after features; wave 2 needs ~1 GPU-h per member on Tier P [INF from §8 timings] | Cleared for the uniform mean and the out-of-fold stacker | Within-session (DIVER has within-session only) |
| 3 | Stimulus-supervised adapter (theirs, not started) | External only: Evanson 2025; BAT-CLIP mixed (linguistic up, onset down) (`architecture-research.md:35`). No Neuroprobe number. | Cross-subject V +0.010 (theirs +0.005) | About 10 A100-h (`03…:40`) plus staging two recordings; one GPU | Q1a allowed with disclosure | All; language tasks |
| 4 | Parcel-transfer pooling (theirs) | One session, every arm negative (−0.0022 to −0.0084); the decisive contrast is in noise | +0.010 to +0.015 cross-subject | CPU minutes per session on the existing cache; a decisive test needs Tier P cross-subject | Mechanism compliant; protocol not (§4) | Cross-subject |
| 5 | Multiscale blend (theirs) | Measured negative: −0.0052, 1 of 5 subjects | — | Done | Mechanism compliant; protocol not | Cross-subject |
| 6 | Temporal filter bank (theirs) | Measured decisively negative: −0.048, controls equal | — | Done | Compliant mechanism | Cross-subject |

The ranking is my judgement from the evidence above. Ranks 1–3 rest on prior, not on measured gains [INF].

## 7. Recommendations for their items

| Item | Recommendation | Reason | What is lost |
|---|---|---|---|
| PopT full board (`popt-board0..3`) | **SCRAP now** | It cannot be an ensemble member or entry (Q5). It does not reproduce: −0.061, −0.056 and −0.094 on 64 paired folds, with −0.04 to −0.16 per subject. 11% of folds are done; finishing costs ≈ 12 h on all four GPUs. Results are saved after every fold (`popt_board.py:215-218`), so stopping now loses at most one in-progress fold per worker (≈ 4.3 min each). | A complete PopT comparison table. The 75 folds already show that the reconstruction misses by more than 0.04. The leaderboard publishes the PopT numbers anyway. |
| MAPA board, within/cross-session (`mapa-board`) | **CONTINUE with a change:** let pair 1 (1_1 and 1_2, within-session and cross-session) finish as a per-contact parity check, then stop. Alternatively, move the solve to the blocked GPU Gram path (`mapa_ridge.py`) after its parity pilot. | The full board is projected at 34–60 h and roughly 24–48 TB of NVMe writes [INF], and it keeps the inference watcher waiting. Parity on two sessions plus exact cross-subject parity is enough to trust published MAPA per-cell values as the evaluation-session reference [INF]. | Reproduced MAPA cells for the other 10 sessions. They are needed only if a Tier T claim has to cite a reproduced rather than published MAPA. |
| Temporal filter bank | **SCRAP** (finished) | −0.048, controls equal, 14 of 15 tasks down | Nothing; the results are kept. The recurrent follow-up (`architecture-research.md:21`) is unsupported by this result. |
| Multiscale blend | **SCRAP** (finished) | −0.0052, 1 of 5 subjects | Nothing. Record it in our ledger as a seen evaluation-validation look (§4). |
| Parcel-transfer pooling | **SCRAP the uniform-pooling form**; any learned set-pooling head waits for Tier P cross-subject | Every arm is negative on its best-case session | The untested learned head |
| Stimulus-supervised adapter | **CONTINUE with a protocol change:** screen on Tier P with a held-out permitted film, use the evaluation sessions only for one frozen-candidate veto, and exclude 6_0 and the partial-session annotations. Start it on one GPU after PopT stops, and share the permitted-session download with our Tier P. | It is the only one of their directions with a plausible, untested prior, and it is rules-cleared under Q1a | — |
| CNN and DIVER reproductions | **Keep (done)**. Reuse the code for M4/M5 with per-trial export. | Validated against the published values | — |

## 8. GPU facts for scheduling

| Job | VRAM | Wall time | Fits in ~7 GiB beside inference? | Disk written |
|---|---|---|---|---|
| MAPA encode, region-mean (`mapa_cs`) | Batch 8: 1,766 MiB reserved (batch 4: 882; batch 16: 4,026). 103–115 windows/s at batch 4–16 (`R/logs/bench.log`). Whole process ≤ 3.1 GiB observed (`research/neuroprobe-results/gpu-load-2026-11-25.csv`, card 2 at 12:52). | ≈ 4 min CPU prepare plus ≈ 4.7 min encode for 13,592 windows (s1_t1: 12:52:04 → 12:56:45, `logs/mapa-board.log`) | Yes | 5.8–10.4 GB per session (84 GB for 11) plus about 2 GB of prepared bands per session (25 GB for 12) |
| MAPA encode, per-contact (`mapa_contact`, taps 0 and 12) | Not logged; same model and batch, so about 2–3 GiB [INF] | 12 min (s1_t1: 14:03:21 → 14:15:26) | Yes [INF] | **74.6 GB per session** |
| MAPA ridge on GPU (`mapa_ridge.py`) | Not logged | Cross-subject pilot S3T0, 15 tasks in about 2 min (`R/results/mapa-gpu/cs/S3T0`, 11:15–11:17) | Unknown | Small |
| MAPA ridge on CPU, for comparison | — | Cross-subject 2.3 min per target (`logs/pilot-score.log`). Within-session per-contact ≥ 78 min per session, still running, with 58.9 GB RSS (peak 64.5 GB). | n/a | Cross-subject negligible. Within-session ≈ 1.1 TB of temp writes in 78 min, deleted files. |
| CNN training | Log prefix peaks at 2.1–2.6 G (`logs/cnn-board.log`, `cnn-extra0.log`, `cnn-extra2.log`); host RAM ≤ 14.3 GB | Whole board (690 folds) in 2.1 h on mostly one GPU | Yes | Cache 2.7–4.0 GB per session (39 GB for 12) |
| DIVER head training (frozen encoder) | Not logged | 13–14 min per session for 15 tasks × 2 folds, including filtering and encoding | Likely [INF] | Temporary `voltage.npy` of channels × windows × 2048 × 4 B (≈ 13.4 GB for 1_1), deleted after encoding. Host features ≈ 8.4 GB bf16 [computed]. |
| PopT fine-tuning | **6,978–7,724 MiB per worker** (`nvidia-smi`, 15:27:59); host RSS 3.9–6.0 GB, peak up to 21 GB (`MemoryPeak`) | 4.3 min per fold per worker; 11–15 min STFT preparation per recording | **No / marginal** (7.0–7.7 GiB used against ≈ 7–7.5 GiB free) | STFT 5.3–7.5 GB per recording (32 GB for 5) |

Host and disk: 251 GB RAM with 100 GB available, of which mapa-board holds about 60 GB. `/` has 523 GB free (`df`, 15:35). mapa-board keeps at most one pair of 74.6 GB per-contact caches (`mapa_sweep.py:60-61`).

## 9. What I could not verify

- **Progress of the within-session MAPA readout.** `mapa_cpu.py` prints only at the end. I estimated progress from `/proc/2243347/io` write volume and temp-file sizes. Cross-session cost is unmeasured because no cross-session shard has run.
- **PopT remaining time.** Fold time depends on window and contact counts per recording. I measured it only on 3_0 and 1_1.
- **VRAM for DIVER, per-contact MAPA encodes and the GPU ridge.** No log records it. The gpu-load CSV covers one minute (12:50–12:53), and I could not map its cards to jobs with certainty.
- **Which cards ran `mapa-encode0/1`** (13:26–13:46). Their logs do not name a UUID.
- **Inference's 56.4 GiB per card.** It comes from their notes (`neuroprobe-reproduction.md:7`); inference has been off since 10:59, so I could not measure it.
- **Why the first PopT run crashed.** I read the traceback only down to the PopT forward call (`logs/popt-board0.log:8-30`). It was relaunched and is progressing.
- **The CNN 4_0 speech fold-0 published value.** Whether 0.592 is an anomaly in the published file, or our 0.837 is the anomaly, is unverified.
- **No adapter code exists** locally or on the server, so §3.4 describes the plan only.

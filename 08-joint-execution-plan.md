# Joint execution plan for the two Neuroprobe workstreams

November 25, 2026, about 15:45 Central. Draft for Jake's decision; nothing here has been executed. "Peer" is the workstream running in `~/runs/neuroprobe-reproduction` (code in `experiments/neuroprobe/`, notes in `research/`). "Ours" is this folder. Facts about the peer's jobs and results come from `07-peer-workstream-audit.md`, which was read from logs and result files between 15:27 and 15:35; times and durations below are estimates from that audit unless marked otherwise.

## Where each direction stands

| Direction | Owner | Evidence so far | Status |
|---|---|---|---|
| Classical features stacked on MAPA | Ours | Indirect: leaderboard per-task gaps (waveform entries beat MAPA on word position; Ridge 6-band beats MAPA on subject 7) | Staged, not run |
| Ensemble of MAPA, Ridge 6-band, DIVER-1, CNN | Ours | Indirect: different winners per task; best-per-task ceiling only +0.004 | Staged, not run |
| Stimulus-supervised adapter | Peer | Literature only | Not started; no code |
| Parcel-transfer pooling | Peer | Measured: −0.002 to −0.008 across four arms, one session | Finished, negative |
| Multiscale blend (MAPA frontend + last layer) | Peer | Measured: −0.005, 1 of 5 participants up | Finished, negative |
| Temporal filter bank | Peer | Measured: −0.048, 0 of 3 up; both controls lose as much | Finished, negative |

- Ours rank higher only because the peer's three measured directions are negative and ours are untested. Neither of ours has a measured gain.
- The peer's results are all cross-subject and were measured on validation labels of evaluation sessions.
- Two of the peer's results lower expectations for our cross-subject regime. Adding a weak kernel at equal weight cost about 0.045 there (the temporal controls), and blending MAPA's frontend with its last layer lost 0.005. Our within- and cross-session regimes are not covered by those measurements.

## Peer items: keep, change, stop

| Item | State | Recommendation | Reason | What is lost |
|---|---|---|---|---|
| Population Transformer board (4 units, all 4 cards) | 75 of 690 folds; about 12 h left | Stop now | It is 0.06–0.09 below the published scores, so it does not reproduce; the rulings exclude its weights from any entry | At most 4 minutes of work per worker; results are saved per fold |
| MAPA board, within- and cross-session (CPU readout) | 0 of 24 shards; 34–60 h left; about 1.1 TB of temporary writes in 78 minutes | Let the first session pair finish as a parity check, then stop and replace with the shared kernel job below | Same computation our tracks need; the GPU kernel path is far faster and bounded on disk | Nothing, once the kernel job reproduces the first pair |
| Temporal filter bank, multiscale blend, uniform parcel pooling | Finished | Close | Negative, with controls | Nothing |
| MAPA cross-subject reproduction (0.6082 vs 0.6083) | Done | Keep as the cross-subject baseline | Exact | — |
| DIVER-1 within-session reproduction (+0.0015 macro) | Done | Keep; reuse the code for our member with per-trial export | Reproduces | — |
| CNN reproduction (macro within 0.004; cells scatter 0.02–0.04) | Done | Keep; reuse the code for our member with per-trial export and three seeds | Reproduces at macro level | — |
| Stimulus-supervised adapter | Not started | Continue under the shared protocol | Allowed only on films no evaluation session shows; needs a permitted-session screen with a held-out film | — |

No reproduction saved per-trial predictions, so every ensemble member needs a rerun with export.

## One protocol for both workstreams

- Both use the tiers, thresholds, ledger and test lock in `06-classical-and-ensemble-plan.md`, implemented in `common/`.
- Design decisions are made on the 13 permitted sessions (Tier P). Evaluation-session validation labels choose a ridge penalty or an early-stopping point and nothing else. Tier V is a single veto.
- The peer's finished experiments chose designs on evaluation-session validation labels, and the multiscale run included subjects 7 and 10. They go in the ledger as prior looks, are disclosed in any submission, and are not counted as Tier P or Tier V evidence.
- Amendment to our plan from the peer's temporal-control result: the weight of each added kernel relative to the base is chosen once per regime on Tier P from {0.1, 0.25, 0.5, 1.0} and then frozen. It is never chosen on evaluation sessions.

## Division of work

| Area | Owner |
|---|---|
| Shared harness (`common/`: splits, ridge, predictions, statistics, ledger, lock) | Ours |
| Tier P data download and preparation | Ours |
| MAPA kernel job (per-contact features reduced to kernels, storage-bounded) | Ours builds it; the peer checks it against its first within-session pair and its cross-subject scores |
| Reproduction baselines and published-score parity | Peer |
| Classical track, ensemble track | Ours |
| Stimulus-supervised adapter; cross-subject transfer ideas | Peer, on the shared harness |
| GPU queue | One owner at a time per card, per the table below; nobody launches on a card they do not hold |

## GPU allocation

Measured so far: MAPA encode at most 3.1 GiB and about 12 minutes per session per-contact; CNN at most 2.6 GiB and 2.1 hours for a whole board; DIVER head training 13–14 minutes per session (memory not logged); Population Transformer 7.0–7.7 GiB. Memory for the GPU ridge and for DIVER is not measured yet.

Two ways to run, depending on whether inference comes back:

- **Option A, restore inference after Population Transformer stops.** Inference holds about 56 GiB per card and leaves about 7 GiB. MAPA encode and CNN fit; DIVER and the GPU ridge probably fit but must be measured first. The peer's restore script waits for every `spark-neuroprobe-*` and `spark-npexp-*` unit to exit, so restoring inference while jobs run means starting it by hand.
- **Option B, keep the exclusive window** until Tier P decisions are in (about a day), then restore.

| Card | First block (tonight) | Second block (tomorrow) |
|---|---|---|
| 0 | Ours: MAPA kernels, evaluation sessions (also yields the peer's within- and cross-session reproduction) | Ours: Tier V and Tier T member exports, if a candidate survives |
| 1 | Ours: MAPA kernels, Tier P sessions, as each recording finishes downloading | Ours: classical Gram products if the GPU path is needed |
| 2 | Ours: DIVER member export, Tier P then evaluation sessions | Peer: stimulus adapter screen |
| 3 | Ours: CNN member export, three seeds | Peer: stimulus adapter controls, or idle |
| CPU | Ours: Tier P download, classical feature caches, C0 parity | Ours: block screen, fusion, statistics |

- The peer's CPU within-session readout uses about 59 GB of RAM and writes heavily to disk; it should not run at the same time as our kernel job. `/` had 523 GB free at the audit.
- Wave 2 member exports (DIVER, CNN) are computed early because the cards would otherwise idle. Whether wave 2 enters a decision still follows the plan's trigger.

## Timeline

Clock starts when Jake approves and Population Transformer is stopped. Durations are estimates; the staging agents' hand-off files will replace them with per-job figures.

| When | What runs | Decision it produces |
|---|---|---|
| Hour 0–3 | Staging finishes; Tier P download; peer's first MAPA session pair completes; kernel job checked against it | Kernel job trusted or not. If not, fix before anything depends on it |
| Hour 3–10 (overnight) | MAPA kernels for evaluation and Tier P sessions; classical feature caches; Ridge 6-band parity run; DIVER and CNN exports | Ridge 6-band reproduces within 0.005 or is labelled "our reimplementation"; members pass the export check |
| Day 1 morning | Classical block screen (no MAPA) and uniform-ensemble screen, both on Tier P, with controls | First refutation point. Classical-only claim: go, hold or stop. Ensemble claim: stop if no subset gains +0.005 over its best member |
| Day 1 afternoon | Classical stack on MAPA with controls; stacker rung; self-ensemble and leave-one-out controls | One candidate per regime frozen in a commit, or the tracks end here with the negative result recorded |
| Day 2 morning | Tier V veto check on the frozen candidate | Veto or proceed. A vetoed regime ends |
| Day 2 afternoon | Tier T, once per surviving regime | Final scores; submit only where the score beats MAPA; ledger published either way |
| Day 2 onward | Peer's stimulus adapter on Tier P with a held-out film and a shifted-target sham | Go, hold or stop under the same thresholds |

What comes next depends on the outcome:

- **Classical stack survives:** the next lineage is a pretrained encoder that takes waveform or complex input (latent-prediction objective), since the gain would show MAPA's encoder is missing that signal.
- **Classical stack fails but the ensemble survives:** the members disagree for reasons other than phase; study which member carries the gain before building anything new.
- **Both fail:** MAPA's features already contain what these methods add. Effort moves to the stimulus adapter and to cross-subject alignment fit on permitted sessions.

## Decisions needed from Jake

1. Stop the Population Transformer run now.
2. Replace the peer's CPU MAPA board with the shared kernel job after its first session pair.
3. Option A or B for inference.
4. Confirm the card table and that each workstream launches only on cards it holds.

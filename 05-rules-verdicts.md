# Neuroprobe rules: yes/no verdicts for our entry

November 25, 2026. This is read-only research: nothing was posted, trained or downloaded. The sources are the clone of `azaho/neuroprobe` at `b282acd` (the current `main`, 2026-10-07), the checkout of unmerged PR #20, the clones of MAPA, DIVER-1, Population Transformer (PopT) and BrainBERT, the Neuroprobe paper ([arXiv 2509.21671](https://arxiv.org/abs/2509.21671)), the MAPA paper ([arXiv 2609.13507](https://arxiv.org/html/2609.13507)), the DIVER-1 paper, and GitHub issues, PRs, reviews and commit history read with `gh`.

## Verdict table

| # | Question / sub-case | Verdict | Basis | Confidence | Required disclosure (YES) or what would flip it (NO) |
|---|---|---|---|---|---|
| Q1a | Stimulus annotations (transcripts, word timings, GPT-2 features, volume, pitch, frame features) from permitted sessions whose film is in no evaluation session, used to pretrain or supervise a model | **YES** | Rule text | Medium | Name the supervised targets, the sessions and films they come from, and the objective. State that no annotation of a film shown in an evaluation session was used. |
| Q1b | The same, for annotations of a film that an evaluation session also shows (6_0 *Megamind* → 4_1, 7_1; the partial sessions 7_100–102 and 10_100–101 come from *Cars 2*, *Megamind* and *Spider-Man: Far From Home*, all evaluation films) | **NO** | Rule text | Medium-high | Flip: a maintainer statement that annotations of evaluation films from permitted sessions do not count as "data that intersects with any data of Neuroprobe", plus an explanation under the attestation's "[If the data intersects, please explain here]" line. |
| Q2a | Labels of other tasks from the evaluation session that fall inside the decoded task's val or test span (within-session), or any target-session labels in the cross-session/cross-subject regimes | **NO** | Rule text | High | Flip: none available; it trains on held-out labels. |
| Q2b | Within-session: other tasks' labels restricted to windows inside the decoded task's own training block | **NO** | Silent | Medium | Flip: a maintainer statement that the training split is the time block, not the task's item list. |
| Q2c | Cross-session/cross-subject: other tasks' labels from the training session (sibling session, or anchor 2_4) | **NO** | Silent | Medium | Flip: a maintainer statement that every label of the training session counts as training data for every task. |
| Q3a | Val labels choose a regularization strength | **YES** | Merged-entry precedent (MAPA) | High | State that the penalty is chosen per task and session (and fold) on validation AUROC over a fixed grid, and give the grid. |
| Q3b | Val labels drive early stopping / checkpoint selection | **YES** | Merged-entry precedent (DIVER-1) | Medium | State that validation AUROC selects the checkpoint, with the patience and the maximum number of epochs. |
| Q3c | Val labels choose a few ensemble weights or which members to include | **NO** | Silent | Medium | Flip: a maintainer statement, or a merged entry, that uses val to weight or select ensemble members. |
| Q3d | Val labels fit a stacking model or any other model parameter | **NO** | Rule text | High | Flip: none available; it turns val into training data. |
| Q3e | Refit the final model on train+val | **NO** | Rule text | High | Flip: none available. |
| Q3f | Fit a cross-subject alignment map with val labels | **NO** | Rule text | High | Flip: none available (same as Q3d). |
| Q4a | Explicitly feed neural data from outside 0–1 s after onset | **NO** | Rule text | High | Flip: none available; the leaderboard bin is fixed at 0–1 s. |
| Q4b | Whole-session filtering or STFT whose kernels cross the window edges | **YES**, within MAPA's extent | Merged-entry precedent (MAPA) | Medium | State that filtering and the STFT run on the continuous session. Give each kernel's length and how far it reaches outside 0–1 s. Keep spectral windows within MAPA's reach of 250 ms before onset. |
| Q4c | Normalization statistics computed over the whole target session, including its unlabelled test half | **YES** | Merged-entry precedent (MAPA) | Medium-high | State that the statistics are fit, without labels, per channel (and frequency bin) over the entire session, including its test half. |
| Q5 | Ensemble member whose pretraining data overlapped evaluation sessions (public PopT or BrainBERT weights) | **NO** | Rule text | High | Flip: a maintainer statement that such checkpoints are allowed in a competitive entry, together with a filled-in intersection explanation. |
| Q6a | Transductive per-channel normalization statistics over the whole target session | **YES** | Merged-entry precedent (MAPA) | Medium-high | Same as Q4c. If MAPA features are reused, also disclose MAPA's per-session Guard 1 contact exclusions on evaluation sessions. |
| Q6a′ | Readout-feature standardization refit on the target's val+test windows | **NO** | Silent | Medium | Flip: a merged entry, or a maintainer statement, permitting it. MAPA explicitly used training-only statistics. |
| Q6b | Unsupervised alignment fit on target-session neural data, including the test half | **NO** | Silent | Medium-high | Flip: a maintainer statement that unlabelled target-session data may fit multivariate alignment parameters. |
| Q6c | Self-supervised adaptation on the target session | **NO** | Rule text | High | Flip: none available; the session is off-limits for pretraining. Self-supervised training on the five provided partial sessions is allowed. |

## Standard applied

- **YES** requires explicit rule text, or a merged entry that verifiably does the same thing.
- **NO** applies when the rules prohibit the practice, or when it would give us label or signal information *about the evaluation data* that no merged entry used. I read the second clause as concerning evaluation data. Otherwise any supervised pretraining on permitted data would fail it, and the rule text allows that pretraining.
- **Silent evidence** gives NO.
- **Unmerged and off-leaderboard work** (PR #20, torch_brain PRs) serves as context only, never as precedent.

## Facts used by several answers

**How the splits are built.**
- Each task has its own dataset. Items alternate positive and negative, and each class's items are sorted by time (`neuroprobe/datasets.py:260-262, 292-302`).
- In cross-session and cross-subject, val is `range(size//2)` of the target session's task dataset and test is the remainder (`neuroprobe/train_test_splits.py:66-71, 153-158`).
- Within-session uses unshuffled 2-fold KFold, then halves the held-out fold into val and test (`train_test_splits.py:219-237`).
- The val/test time boundary therefore differs by task and by class. MAPA's code states the same: "Each task's interleaved dataset order determines its validation/test split on that union axis" (`MAPA mapa/probe/labels.py:1-5`).

**The October 2025 split change.**
- Commit `a741139` (2025-10-23, "added val split") and `324a8c2` (2025-10-24) carved val out of the first half of what had been the test set.
- Commit `a090cd2` (2025-10-24) added the `## Train/Val/Test Splits` section to `SUBMIT.md`. Its whole content is "you MUST use the exact train/val/test splits that are provided by the Neuroprobe package" (`SUBMIT.md:10`).
- None of these commits, nor any later one, says what val may be used for. I read every commit touching `SUBMIT.md`, `README.md`, `train_test_splits.py` and `config.py`.
- DIVER-1's metadata records the change: "previous neuroprobe splits (25 SEP ver … no validation set, just 1:1 train test splits; we used 10% of train set as validation set in that version). The current results are based on the official neuroprobe splits with validation sets.(25 OCT ver)" (`leaderboard/DIVER-1(0.1s,tiny,frozen)_…/metadata.json:8`).

**Maintainer activity.**
- No maintainer has written anything about validation use, leakage, transductive preprocessing, stimulus overlap or pretraining overlap.
- Issues #3, #4 and #18 are code bugs.
- PRs #5, #6, #7, #9, #10, #11 and #16 have no reviews and no inline comments. Their commit messages are routine ("Use public brainbert weights" in #10; "Rename directory" in #11).
- The only maintainer comment on any leaderboard PR is azaho on #16 (MAPA): "Perfect. Thank you Ben." ([PR #16](https://github.com/azaho/neuroprobe/pull/16)). azaho's commits carry the author name Andrii Zahorodnii.
- PR #20 is open, with no review since 2026-10-13. It was last updated on 2026-11-16 ([PR #20](https://github.com/azaho/neuroprobe/pull/20)).
- `insight-neuro/neuroprobe` lists the same issues and PRs.
- I also searched the issues of bentang18/MAPA (none), DIVER-1 (#1, pretraining code), PopulationTransformer (#2, #3), BrainBERT (#1–#12), iMINDBench (PR #1 only), and torch_brain PRs #115, #134, #139, #187 and #362 and brainsets#133. None of them addresses these questions.

**Submission checks.**
- `tests/test_submission_format.py:247-271` requires only the phrase "I attest that the training and test splits of Neuroprobe were respected" and one "SIGN". It skips the check entirely when the file reads exactly "N/A" ("Allow 'N/A' as a placeholder for now", line 259).
- The pretraining attestation is not tested.

## Q1. Stimulus annotations from permitted pretraining sessions

**Decisive evidence.**
- `SUBMIT.md:32`: "For the validity of the evaluation on Neuroprobe, **no models can be pretrained on the same data that underlies the Neuroprobe evaluation**. So, the following sessions are off-limits…"
- `SUBMIT.md:40-42` lists the 14 sessions "that pretraining is allowed on". It does not restrict the objective to self-supervision.
- `SUBMIT.md:44-46` permits the partial sessions 7_100–102 and 10_100–101, "taken from the same trials as Neuroprobe but during the times that do not intersect with Neuroprobe indices". This shows the maintainers define "intersect" at the level of benchmark windows, not sessions.
- `SUBMIT.md:125-127`, the attestation: "not pretrained on any data that intersects with any data of Neuroprobe … [If the data intersects, please explain here]".

**Film overlap** (`neuroprobe/config.py:113-140`).
- *Megamind*: 6_0 (permitted), 4_1 and 7_1 (evaluation).
- *Cars 2*: 3_0, 7_0 and 10_0, all evaluation sessions.
- *Fantastic Mr. Fox*: 1_0 and 5_0, both permitted. No evaluation session shows it.
- The partial sessions come from trials 7_0 (*Cars 2*), 7_1 (*Megamind*), 10_0 (*Cars 2*) and 10_1 (*Spider-Man: Far From Home*). All four films are evaluation films.
- The word, pitch and volume annotations of every session ship inside the package (`neuroprobe/braintreebank_features_time_alignment/`, `neuroprobe/pitch_volume_features/`).
- The evaluation labels are percentile thresholds of these movie-level features within a session (paper §3, "Decoding tasks"). The 6_0 *Megamind* annotations at movie time *t* are therefore the same feature values that define the 4_1 and 7_1 labels at time *t*.

**What entries did.**
- MAPA's pretraining is masked reconstruction: "No task labels are used during pretraining" (`MAPA MODEL_CARD.md:84-88`).
- MAPA did use labels from permitted sessions to set its schedule: "During the constant phase we fit the readout on four tasks in seven pretraining sessions, none of which Neuroprobe evaluates, and we start the cooldown once the readout stops improving" (MAPA paper, Appendix B). It does not say which seven sessions, so whether 6_0 was among them is unknown.
- DIVER-1 pretrained on non-Brain-Treebank iEEG (metadata attests no intersection). PopT and BrainBERT pretraining is self-supervised.
- No merged entry trains weights on stimulus labels.

**Reasoning.**
- **Q1a.** The permitted-session list is explicit and unconditional. Annotations of films that no evaluation session shows do not intersect evaluation data, so the attestation can be signed truthfully. The answer is YES, with disclosure, because no entry has done supervised pretraining and reviewers should see it stated.
- **Q1b.** The *Megamind* annotations of 6_0 are, time point by time point, the source values of the 4_1 and 7_1 labels. A model supervised on them learns the mapping from the neural response at a given *Megamind* scene to that scene's label, and the evaluation sessions show those same scenes. That is label information about evaluation items, and it intersects "data of Neuroprobe". The partial sessions are evaluation films by construction. The answer is NO. Self-supervised use of the *neural* data of 6_0 and of the partial sessions stays allowed (`SUBMIT.md:40-46`).

## Q2. Multi-task training with other tasks' labels from evaluation sessions

**Decisive evidence.**
- `SUBMIT.md:10`: "you MUST use the exact train/val/test splits."
- The paper explains why the folds are contiguous: "the train examples are taken from a single contiguous block and the validation examples are taken from a separate block. This is done to prevent models from overfitting to temporal auto-correlation" (§3, "Evaluation Splits").
- Splits are generated per task (`train_test_splits.py:174-252`), so task A's training items can lie inside task B's val or test time span (see "How the splits are built").
- In cross-session and cross-subject, every target-session label of every task is val or test.

**What entries did.**
- MAPA encodes a task-agnostic union of windows once, without labels (`MAPA evals/neuroprobe/encode.py:391-408`). It then fits each task's ridge only on that task's own `train` indices and selects only on that task's `val` (`MAPA evals/neuroprobe/readout.py:255-287, 289-311, 333-371`).
- DIVER-1 trains one model per task (`DIVER-1 scripts/finetune_neuroprobe.sh`, one `--neuroprobe_task` per run).
- No merged entry shares labels across tasks.

**Reasoning.**
- **Q2a.** This trains on labels that lie in the decoded task's held-out time span, or in the target session's evaluation halves. It is NO.
- **Q2b.** Restricting the other tasks' labels to the decoded task's own training block removes the temporal leak. It still trains on items and labels that are not in that task's train dataset, and nothing in the rules or the entries covers that. It is NO.
- **Q2c.** The training session (the sibling session, or 2_4) is an evaluation session and off-limits for pretraining (`SUBMIT.md:33-38`). A multi-task model fit on all 15 of its label sets amounts to supervised pretraining on an off-limits session, and no entry does it. It is NO.

## Q3. Uses of the validation labels

**Decisive evidence.**
- `SUBMIT.md:10` provides `val_dataset` separately from `train_dataset` and says nothing more. The paper does not discuss the val split.
- The repository's example script never reads `fold["val_dataset"]` (`examples/eval_population.py:206-208`). Its MLP path passes the *test* split as the early-stopping set: `clf.fit(X_train, y_train, X_val=X_test, y_val=y_test)` (`examples/eval_population.py:273`, consumed in `examples/eval_utils.py:745-821`). I could not establish whether any merged entry was produced with that path.
- The example `CNNClassifier` early-stops on the last 20% of the training split (`examples/eval_utils.py:510-588`).

**What entries did.**
- **MAPA (merged):**
  - Val selects only the ridge penalty, the first maximum over a fixed 25-point grid (`readout.py:78, 204-221`). Standardization uses training statistics only (`readout.py:93-130`).
  - The paper says: "The regularization strength is the only quantity we select in our readout … We use the value with the highest AUROC on the validation trials" (Appendix C).
  - The model card says: "Validation labels select the ridge regularization strength; test labels are used only for scoring" (`MODEL_CARD.md:101-102`).
  - The encoder tap is fixed to block 12 for the submission (`evals/README.md:174-176`).
- **DIVER-1 (merged):**
  - The public script runs `--early_stop_criteria val_auroc --early_stop_patience 10` (`DIVER-1 scripts/finetune_neuroprobe.sh:34-35`). The metadata states that the leaderboard numbers use "the official neuroprobe splits with validation sets".
  - The public loader is the September version, which takes val as a shuffled 10% of train (`datasets/datasets_loaders.py:825-833`). The October code is not published.
- **GLIS-GNN (merged):** the Neuroprobe paper says it used "early stopping based on validation AUC" (Appendix D). The entry's own notes say "A random 20% of training samples is held out for validation" ([eg-99/neuroprobe `examples/GNN_METHODS.md`](https://github.com/eg-99/neuroprobe)).
- **Caltech CNN/MLP (merged, `PUBLICATION.bib` "N/A"):** the code those entries were made with is not identified. The same author's closed torch_brain PR [#187](https://github.com/neuro-galaxy/torch_brain/pull/187) early-stops on the official val split, with patience 10 (`examples/neuroprobe_eval/torch_runner.py:363-476`). For logistic models it scores val and test merged (`utils/fold_helpers.py:75-95`). Linking this code to the merged entries is an inference.
- **Ridge 6-band (PR #20, unmerged):** "The ridge penalty is chosen on the official validation half."

**Reasoning.**
- **(a)** MAPA does exactly this and was merged. YES.
- **(b)** DIVER-1 declares official val sets and uses val-AUROC early stopping. The maintainers' own paper describes an accepted entry the same way. YES. Confidence is medium because DIVER's October code is unpublished.
- **(c)** Choosing ensemble weights or members is a different quantity from any merged entry's selection, and the rules are silent. NO.
- **(d), (f)** Fitting parameters on val labels treats val as training data, and the provided split defines it as separate. MAPA's paper counts "target-session validation labels" as labels the method consumed (Appendix G). NO.
- **(e)** Refitting on train+val trains on target-session labels. In cross-session and cross-subject that changes the regime. Within-session it trains on the block adjacent to test. NO.

## Q4. Neural signal outside the 0–1 s window

**Decisive evidence.**
- `neuroprobe/config.py:27-28`: "START_NEURAL_DATA_BEFORE_WORD_ONSET = 0 … NOTE: for the 1-second evaluation on the leaderboard, this is overridden to 0" and "END_NEURAL_DATA_AFTER_WORD_ONSET = 1 … overridden to 1".
- Results must sit under `"one_second_after_onset"` with `"time_bin_start": 0.0, "time_bin_end": 1.0` (`SUBMIT.md:68-70`).
- The README invites custom pipelines: "In case you'd like to use your own pipeline for extracting and preprocessing data, feel free to set `dataset.output_indices = True`", which returns "indices into the session's h5 file" (`README.md:88-94`). Custom pipelines therefore read the raw session file directly.
- No text addresses filter kernels or statistics that extend past the window.

**What entries did.**
- **MAPA (merged)**, `mapa/preprocessing/frontend.py`:
  - It filters the continuous session: shaft-mean reference, then MNE notch at 60–300 Hz and a 0.5 Hz high-pass, "MNE default FIR; zero phase" (lines 49-60, 78-83, 251-253). [INFERENCE] MNE's defaults give a high-pass FIR of about 6.6 s.
  - It computes centered STFTs over the whole session, "periodic Hann", `center: True`, reflect padding only at the true recording ends (lines 89-95, 256-270; `mapa/preprocessing/stft.py:71-84`).
  - It slices clips as cached STFT frames: "Extract a clip from the cached STFT frames, not another FFT window" (lines 136-146; `evals/neuroprobe/encode.py:121-144`).
  - It snaps window starts to the 32 Hz grid, `round(seconds * 32)` (line 103), a shift of up to ±15.6 ms.
  - With the stem strides (8, 2, 1) (`mapa/models/stem.py:24`), the first Slow frame, a 500 ms window, is centered on onset. It therefore covers 250 ms of pre-onset signal. The last decimated frames end at about 1.0 s.
  - It fits median/MAD normalization over "entire session, separately per contact and frequency bin" (lines 96-101, 271-276). The paper says the same (§2, "Frontend").
- **Reference code:** `examples/eval_utils.py:52-90` runs `torch.stft(..., center=True)` on the 1-s clip only, so it uses no outside signal. Its z-score step is commented out (lines 115-119). Its classifiers use `StandardScaler` fit on train.
- **DIVER-1:** it reads exactly 0–2048 samples (`datasets_loaders.py:805-806`). It builds the subject with `BrainTreebankSubject(..., DIVER_preprocess=True, clip=1)`. That argument does not exist in the public Neuroprobe package, so its session-level behaviour is unknown.
- **Ridge 6-band (unmerged):** "each normalized with median/MAD per channel over the session" (PR #20 `metadata.json`).

**Reasoning.**
- **(a)** A longer explicit window contradicts the fixed leaderboard bin. NO.
- **(b)** MAPA verifiably runs session-level filters and STFT windows that reach 250 ms before onset. Its code and preprocessing config document this, and it was merged. YES, bounded by that extent. A kernel built to reach further, or designed to capture pre-onset or post-window content, would be (a) in another form.
- **(c)** MAPA verifiably fits normalization over the whole session, including the test half, without labels, and was merged. YES.

## Q5. Third-party checkpoints whose pretraining overlapped evaluation sessions

**Decisive evidence.**
- `SUBMIT.md:32-38` makes the 12 sessions "off-limits, with pretraining on them NOT allowed". `SUBMIT.md:125` asks submitters to attest "not pretrained on any data that intersects with any data of Neuroprobe".
- The public PopT pretraining split (`PopulationTransformer trial_selections/pretrain_split_trials.json:1`) includes 1_2, 2_0, 2_4, 3_1, 4_1, 7_1 and 10_1: 7 of the 12 evaluation sessions.
- BrainBERT's `data/pretrain_split_trials.json:1` is identical. PopT builds on the public BrainBERT STFT weights (`PopulationTransformer README.md:58-80`).
- The Neuroprobe paper says of its PopT baseline: "We use the publicly available weights which were pretrained on data from 10 iEEG subjects" (Appendix D).

**What the accepted entries attested.**
- **PopT, global z-scoring (Geeling Chau, PR #11, merged 2026-05-02):** `ATTESTATION.txt` has only lines 1-2, the splits attestation and "SIGN Geeling Chau". The pretraining attestation is absent. The metadata does not name the checkpoint.
- **BrainBERT, frozen, global z-scoring (Geeling Chau, PR #10, merged 2026-05-02; commit "Use public brainbert weights"):** the same; only the splits attestation is signed (`ATTESTATION.txt:1-2`). Its `PUBLICATION.bib` begins with "N/A".
- **PopT off-the-shelf, BrainBERT off-the-shelf and BrainBERT untrained (Andrii Zahorodnii):** each `ATTESTATION.txt` is the single line "N/A". The format test skips such files.
- By contrast, MAPA, DIVER-1, CNN, MLP, GLIS-GNN, GRU and the Linear entries sign both attestations.
- The PopT and BrainBERT entries were submitted by Neuroprobe paper co-authors and merged without review comments.

**Reasoning.** The rules prohibit pretraining on these sessions. The precedent entries were accepted only without signing the non-overlap attestation. Including such a checkpoint would give our entry signal from the test halves of 7 evaluation sessions, which we would have to declare. NO.

## Q6. Transductive use of unlabelled target-session data

**Decisive evidence.**
- The off-limits rule (`SUBMIT.md:32-38`) forbids pretraining on target sessions. The partial-session clause (`SUBMIT.md:44-46`) permits only the five specified files from evaluation trials.
- Nothing addresses normalization or alignment fit on target-session data.

**What entries did.**
- **MAPA, normalization:** frontend median/MAD statistics come from the whole session (see Q4c).
- **MAPA, contact exclusion:** "Guard 1" removes contacts from evaluation sessions using whole-session voltage statistics. "Guard 1 runs on each session separately … Guard 1 removes … 25 of the 1,414 evaluation contacts" (MAPA paper, Appendix H). The frozen per-session lists for all 12 evaluation sessions are in `mapa/data/anatomy.py:181-222`, and `evals/README.md:13-17` says "these exclusions affect evaluation inputs". The same paper's Appendix C states "Trials, labels, splits, and evaluation contacts come from Neuroprobe unaltered".
- **MAPA, readout:** "We standardize each input dimension by the mean and standard deviation of the training trials. In the cross-subject regime the training trials come from the anchor, so the anchor's statistics are applied to the test subject" (Appendix C; `readout.py:93-104`).
- **Other entries:** no merged entry fits alignment or self-supervised adaptation on target sessions.

**Reasoning.**
- **(a)** Per-channel (and per-frequency) statistics over the whole session match MAPA. YES, with disclosure.
- **(a′)** Refitting readout standardization on the target's evaluation windows goes beyond MAPA, which deliberately used training statistics. NO.
- **(b)** Multivariate alignment fit on target data, including the test half, has no precedent and no rule text. NO.
- **(c)** Self-supervised adaptation is pretraining on an off-limits session. NO.

## Consequences for the experiment plan

| Experiment | Cleared? | Condition / reason |
|---|---|---|
| Stack classical features onto MAPA features in one ridge, strength chosen on val | **Cleared** | One shared penalty from a fixed grid, selected on val AUROC (Q3a). Training-only readout standardization. Classical features must be in-window, or follow the Q4b/Q4c rules. Separate penalties per feature block are not covered by precedent (Q3c logic): **not cleared**. Reusing MAPA features inherits its session normalization and Guard 1 exclusions; disclose both. |
| Ensemble MAPA, Ridge 6-band, DIVER-1, CNN with (i) uniform weights | **Cleared** | Each member is trained on train, with only Q3a/Q3b use of val. The average is fixed in advance. The DIVER-1 checkpoint has no Brain Treebank pretraining. Ridge 6-band has no released code, so it must be reimplemented from PR #20 and credited. Any DIVER-style preprocessing must meet Q4. |
| (ii) weights chosen on val | **Not cleared** | Q3c. |
| (iii) stacker fit on out-of-fold *training* predictions | **Cleared** | Inner folds must be contiguous time blocks within the training split. The stacker sees no val or test labels. Its own penalty is fixed or chosen on val as a single λ (Q3a). |
| Population Transformer (or BrainBERT public weights) as an ensemble member | **Not cleared** | Q5. |
| Whole-session filtering vs strictly in-window feature extraction | **Both cleared; in-window is the conservative choice** | Whole-session processing must stay within MAPA's extent, with spectral windows reaching at most 250 ms before onset. Disclose kernel lengths and reach. Explicit longer windows are not cleared (Q4a). |
| Stimulus-aligned pretraining | **Cleared only on films absent from evaluation** | Use annotations of 1_0, 2_1, 2_2, 2_3, 2_5, 2_6, 3_2, 4_2, 5_0, 6_1, 6_4, 8_0 and 9_0. Exclude the annotations of 6_0 and of all partial sessions (Q1b). Their neural data may still be used self-supervised. Disclose the targets and sessions. |
| Functional cross-subject alignment | **Cleared only if fit on permitted data or training labels** | Allowed: alignment learned during pretraining on permitted sessions, including the target subjects' own permitted sessions (1_0, 3_2, 4_2, 7_100–102, 10_100–101); anchor-side training data; atlas/coordinate-based region pooling (MAPA and the paper's baseline); and per-channel session normalization (Q6a). Not cleared: alignment fit on target-session neural data (Q6b), on target val labels (Q3f), or on the target subject's other evaluation session (off-limits). |

## Not verified

- **Code behind the Caltech CNN, MLP, PopT and BrainBERT entries.** `PUBLICATION.bib` is "N/A" for CNN and MLP. The closest code is the same author's closed torch_brain PR #187, and its link to the entries is inferred. I could not determine what "global z-scoring" was computed over.
- **Which PopT checkpoint the Chau PopT entry used.** The metadata does not say. The paper's baseline section says "publicly available weights".
- **DIVER-1's October 2025 training code** and its `BrainTreebankSubject(DIVER_preprocess=True, clip=1)` preprocessing, which the public Neuroprobe package does not contain. The public repository holds only the September loader.
- **Which seven permitted sessions MAPA probed** to set its cooldown, and whether 6_0 was among them. The pretraining loop is unreleased.
- **MAPA's Guard 1 detector**, which is unreleased. I relied on the paper's description and the frozen lists.
- **MAPA's split reimplementation.** It uses `mapa/data/word_events.py:386-469` rather than calling `train_test_splits.py`, although its attestation names that function. I did not run its parity test (`tests/test_readout.py`).
- **MNE filter length.** The ~6.6 s high-pass length comes from MNE's documented defaults. I did not run MNE.
- **Ridge 6-band filtering.** Whether its band filtering and envelopes are whole-session is unknown; no code is released.
- **The two Q3 example-code observations.** I could not establish whether any merged entry used `examples/eval_population.py`'s MLP path, which early-stops on test, or torch_brain's merged val+test logistic scoring.
- **Maintainer statements.** None exists on any of Q1–Q6 in the sources listed under "Maintainer activity". The website source (`website/`) contains no rules text.

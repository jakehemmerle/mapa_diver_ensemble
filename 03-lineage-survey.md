# Neuroprobe: architecture and training lineages no current entry uses, with verdicts and a plan to validate them

No training was run and nothing was downloaded. The statistics below come from `/tmp/np_agg.py` and a paired-statistics script I wrote at `/tmp/np_stats.py`, both reading the leaderboard JSONs. Anything marked **[INF]** is my inference; everything else is cited.

## 1. What the leaderboard numbers say

- **Subjects, not sessions, are the unit of analysis.** For MAPA minus Ridge 6-band, the per-session differences of two sessions from the same subject correlate at r = 0.81 / 0.91 / 0.85 (within / cross-session / cross-subject). So the real n is 6 subjects (5 in cross-subject), not 12 sessions.
- **Noise floor.**
  - Subject-level paired SE of MAPA minus Ridge 6-band is 0.0060 / 0.0089 / 0.0102.
  - MAPA's leads over Ridge 6-band are +0.0126 / +0.0181 / +0.0237. On an exact one-sided sign-flip test over subjects these give p = 0.047 / 0.047 / 0.063.
  - With 6 subjects, the smallest p the test can return is 1/64 = 0.016, and only if every subject improves.
  - So a new #1 needs roughly +0.011–0.015 macro within-session, about +0.02 cross-session and about +0.025 cross-subject over MAPA, unless the gain is positive for every subject.
  - Per-task SE is about 0.008–0.011, so single-task claims need gains of 0.02 or more.
- **Picking the best entry per task adds only +0.004.** Wins have to come from new information, not from selecting between existing systems.
- **Where MAPA loses (within-session):**

  | Task | Winner | Margin over MAPA |
  |---|---|---|
  | word_index | DIVER-1 frozen | +0.017 |
  | word_length | DIVER-1 frozen | +0.004 |
  | delta_volume | DIVER-1 frozen | +0.003 |
  | gpt2_surprisal | Ridge 6-band | +0.004 |
  | frame_brightness | CNN | +0.031 |

  - Ridge 6-band beats MAPA on both subject-7 sessions within-session (+0.014 on each) and cross-session (+0.017 / +0.012).
  - Linear on raw voltage scores 0.742 on word_index against 0.651 for linear on spectrogram.
  - Ridge 6-band is the only top entry that keeps a 1–30 Hz waveform stream (its `metadata.json`).
- **Cross-subject is crude.** The readout uses only the anchor-and-test atlas-region intersection, 3–7 regions per pair (`/tmp/mapa_probe/evals/neuroprobe/readout.py:230`; MAPA Appendix C). The anchor is session btbank2_4 (`neuroprobe/config.py:37`).
- **MAPA's encoder is a usable base.** Its encoder checkpoints and eval code are released under Apache-2.0 (`/tmp/mapa_probe/MODEL_CARD.md`, `NOTICE:8`, `evals/neuroprobe/{encode,readout,build_submission}.py`), but its pretraining loop is not. Its encoder adds +0.021 / +0.025 / +0.021 over its own frontend (MAPA Table 6).

## 2. Lineage table

All costs are for the first experiment. "Dev" means the harness in §3. The go/no-go column is the **GO** threshold; anything below +0.005 is **STOP** (§3).

| # | Lineage | Closest prior instance (reported result) | Public code/weights | Prior Neuroprobe / Brain Treebank result | Why it might help here | First-experiment cost | Go/no-go (dev) | Verdict |
|---|---|---|---|---|---|---|---|---|
| L1 | Classical features stacked on MAPA: low-frequency waveform/ERP, phase, cross-frequency coupling, filter-bank Riemannian tangent-space covariance | MOABB: Riemannian ≈ CNNs across regimes (J. Neural Eng. 2024); ECoG Riemannian features +19% cross-session (iScience 2025) | pyRiemann; MAPA readout code | Ridge 6-band (has a 1–30 Hz waveform stream) 0.683; linear raw voltage word_index 0.742 | MAPA is magnitude-only and loses word_index and delta_volume to waveform entries; CNF paper says surprisal, word length and head position need phase | ~2 GPU-h (A100/H100, MAPA encoding) + ~20 CPU-h | Δmacro ≥ +0.010 vs MAPA features alone; ≥7/8 subjects up; Δword_index ≥ +0.015 | **QUALIFIED** |
| L2 | Ensembling MAPA + Ridge 6-band + DIVER-1 frozen (score or stacked) | — | All three reproducible (Ridge 6-band from its metadata) | Best-of-per-task 0.699 | Different winners per task and per subject (subject 7) | ~3 GPU-h + CPU | Same as L1 | **QUALIFIED** |
| L3 | Latent-prediction objectives: JEPA, data2vec, contrastive forecasting (BaRISTA is the masked-latent instance) | CNF (BrainBERT-mini, matched): contrastive on voltage 0.653–0.664 vs MSE on voltage 0.638 (14 tasks); EEG-JEPA frozen 40.5→50.4% vs masked reconstruction (arXiv 2608.00114) | BaRISTA repo + braindecode; EEG-JEPA official; CNF code "upon publication" | BaRISTA: 0.716 vs DIVER 0.728 in iMINDBench (Main units); 0.678/0.745 vs DIVER-BTB 0.770/0.859 speech/onset (DIVER Table 3); CNF-1 "only slightly" above linear | MAPA is an MSE MAE; matched ablations favour latent or contrastive targets | ~50–70 H100-h, plus reimplementing MAPA pretraining from its Appendix B | Latent arm ≥ MSE arm +0.010 macro; ≥7/8 subjects | **QUALIFIED** (BaRISTA's specific recipe disqualified) |
| L4 | Stimulus-aligned supervised pretraining (contrastive or regression to LM, audio or video embeddings; Evanson 2025) | Evanson: supervised contrastive to wav2vec 2.0 beats SSL fine-tuning; log-linear in hours (3 subjects, retrieval rank) | Défossez-style brain module; GPT-2/wav2vec public | Low-rank LLM alignment on Brain Treebank Cars 2 sEEG (arXiv 2610.40511); none on Neuroprobe | 5 language tasks sit at 0.62–0.64; dense targets carry more supervision than binary labels | ~10 A100-h (text targets) | Δlanguage-5 mean ≥ +0.015 and Δmacro ≥ +0.008 when added to MAPA | **QUALIFIED** (text variant; rules question Q1). Audio/video: **UNDECIDED** |
| L5 | Multi-task supervised training on the 15 binary labels | PopT-v2 joint supervised training | iMINDBench (to be released) | iMINDBench: 3–18× more same-task labelled units gives +0.001 on Neuroprobe | — | ~10 GPU-h | — | **DISQUALIFIED** |
| L6 | Discrete tokenizer + masked/AR modelling (LaBraM, NeuroLM, TOTEM, Du-IN) | NeuroLM does not beat LaBraM | LaBraM, NeuroLM, TOTEM public | TOTEM+Linear on BTB 0.58/0.66/0.78/0.79 vs iBrain 0.68/0.89/0.86/0.89 (iBrain Table 1) | — | ~30 GPU-h | — | **DISQUALIFIED** |
| L7 | Spike/event-tokenized perceivers (POYO, POSSM) | POYO+ value tokens on calcium imaging; REALM for LFP | torch_brain | None found | Learned per-electrode embeddings (CNF Appendix B) | — | — | **DISQUALIFIED** standalone (idea folded into L3 as an ablation) |
| L8 | Speech-SSL raw-waveform encoder (wav2vec 2.0 / HuBERT conv front end) | ECoG wav2vec SSL (arXiv 2405.18639); BENDR | fairseq; BENDR | DIVER-1 (raw-waveform CNN, masked reconstruction) frozen 0.678; CNF-1 | Phase information MAPA lacks | Free inside L1 | — | **UNDECIDED** |
| L9 | State-space / Mamba models and general time-series FMs (MOMENT, Chronos, Mantis) | Frozen Mantis on EEG "dramatically hurts" (arXiv 2510.27522); frozen MOMENT not consistently better (arXiv 2606.30104) | Public | iMINDBench could not adapt long-context MVPFormer/Brant to 1-s windows | — | ~5 GPU-h | — | **DISQUALIFIED** |
| L10 | Functional cross-subject alignment (CCA, shared-response/hyperalignment in stimulus-feature space, per-subject linear input maps) | Spalding et al., Nat. Commun. 2026 (CCA shared latent improves cross-patient μECoG decoding); Seegnificant (subject heads) | Seegnificant; CCA trivial | None | Cross-subject uses only 3–7 shared regions | <10 CPU-h after the MAPA cache | Δcross-subject macro ≥ +0.015; ≥5/6 subjects up | **QUALIFIED** (cross-subject regime only) |
| L11 | Scale only: more external iEEG, larger models | DIVER-1: 5,310 h; 15.8× BTB-scale external data ≈ 63× on speech/onset (Table 3) | DIVER-1 code | DIVER-1 frozen 0.678 < MAPA 0.695 (27.9 h); Tiny ≥ Small/Large/XL frozen (DIVER appendix) | — | >100 GPU-h | — | **DISQUALIFIED** as the first move |
| L12 | Physiology-guided targets + masked latent prediction (PHASE; found in this survey) | PHASE-ST frozen BTB volume 0.945 / pitch 0.882 (BaRISTA split, PopT protocol) | No code link found | Not on Neuroprobe | Explicit band/physiology targets | 0 until the fact is settled | — | **UNDECIDED** |
| L13 | Functional-clustering tokens (NeuroCLUS, ICML 2026; found in this survey) | BTB onset/speech/volume/pitch 0.96/0.99/0.92/0.83 (PopT protocol, as quoted by PHASE) | Not released (per a blog digest) | Not on Neuroprobe | Functional rather than anatomical grouping | 0 until the fact is settled | — | **UNDECIDED** |

**Facts that would settle the UNDECIDED rows:**
- **L4 audio/video:** can the films be bought and aligned to Brain Treebank word timings? Test on one permitted-session film against `transcripts/<movie>/features.csv` onsets. Audio is not in the release, which lists transcripts, features and `movie_frames.zip`.
- **L8:** if L1's waveform arm recovers less than +0.005 macro, disqualify L8. If it recovers at least +0.010, run a 1-subject wav2vec-style pilot (~20 A100-h).
- **L12:** PHASE's Neuroprobe-protocol score and its release status. Ask the authors. Also confirm its 17 BTB pretraining recordings exclude the 12 evaluation sessions.
- **L13:** code plus a Neuroprobe-split result, via the authors' contact address.

## 3. Shared validation harness

**Code base**
- Use MAPA's `evals/neuroprobe` (`encode.py`, `readout.py`) with Neuroprobe pinned at `f9b0842`, as MAPA's README specifies.
- Register the dev sessions by extending `NEUROPROBE_LITE_SUBJECT_TRIALS` in-process (`config.py:41-49`). The dataset asserts membership at `datasets.py:95-100`.
- That keeps Lite electrodes (`config.py:143`, defined for all 10 subjects), the 3,500-trial cap (`config.py:39`, `datasets.py:255-256`), 2-fold contiguous within-session splits, and val = first half of the test fold (`train_test_splits.py:233`).
- MAPA's Guard-1 lists exist for every permitted session except 5_0 (`mapa/data/anatomy.py:181-216`). Run 5_0 with no extra drop and flag it.

**Dev data.** The permitted sessions carry the same annotations (`braintreebank_features_time_alignment/` has words/nonverbal CSVs for all 26).
- **Within-session:** 14 sessions (1_0, 2_1, 2_2, 2_3, 2_5, 2_6, 3_2, 4_2, 5_0, 6_0, 6_1, 6_4, 8_0, 9_0) from **8 subjects**. That is more subjects than the test set has.
- **Cross-session:** ordered pairs within subjects 2 and 6, the only subjects with 2 or more permitted sessions.
- **Cross-subject:** anchor 2_6, tested on the 9 sessions of the other 7 subjects, with the same region-intersection readout.

**Contamination tiers**
- **Tier A** (no Brain Treebank pretraining, or the released MAPA checkpoint): all dev sessions.
  - MAPA pretrained on every permitted session except 5_0 (MAPA Table 3), so MAPA-based arms see their own pretraining data.
  - **[INF]** The bias mostly cancels in MAPA-vs-MAPA+X comparisons. Report 5_0 separately as the only clean session.
- **Tier B** (new Brain Treebank pretraining): pretrain only on {2_1, 2_2, 2_3, 2_5}, the 7/10 partial sessions and external data. Evaluate on the other 10 permitted sessions (8 subjects); cross-subject still uses anchor 2_6.
  - This mirrors the benchmark, which never pretrains on its anchor 2_4, and MAPA's held-out-subject test.

**Baselines, re-run inside the harness**
- B0: linear on raw voltage (`examples/eval_population.py`).
- B1: MAPA frontend (tap 0) + ridge.
- B2: MAPA `mapa_vits384.pt` with the paper's taps + ridge.
- B3: Ridge 6-band, reimplemented from its `metadata.json`.
- B4: DIVER-1 tiny frozen (for L2).
- A candidate must beat max(B2, B3) in each regime. Tier-B pretraining candidates must also beat a matched-compute MSE arm on the same pool.

**Statistics**
- Average folds and sessions per subject. Use an exact one-sided sign-flip permutation over subjects, the method in MAPA Appendix D. Add a session-level paired bootstrap 95% CI.
- Per-task claims use Holm correction over the 15 tasks.
- Every trained arm runs at least 3 seeds and reports seed SD. Seed variance on Neuroprobe is unknown, since each leaderboard entry reports one run.

**Thresholds**
- Assuming dev noise like the leaderboard's (subject SD ≈ 0.011–0.015, n = 8, so SE ≈ 0.004–0.005), the one-sided 80%-power detectable effect is about +0.010–0.013.
- **GO:** Δ ≥ +0.010 within-session with at least 7/8 subjects positive (exact p ≤ 0.035). For cross-subject: Δ ≥ +0.015 with at least 6/7 positive.
- **CONFIRM** (new seed or alternate pool): +0.005 to +0.010.
- **STOP:** below +0.005.
- Hyperparameters, stacking weights and ensemble weights are chosen on the val half only. Test labels are touched once per final candidate.

## 4. Per-lineage experiments (QUALIFIED and UNDECIDED)

| # | Smallest decisive experiment | Data | Compute | Main way it could mislead |
|---|---|---|---|---|
| L1 | Ridge on [B2 features ‖ 1–30 Hz waveform 100 ms bins ‖ theta/alpha Hilbert phase (cos/sin) ‖ filter-bank tangent-space covariance, shrinkage]. Ablate one block at a time. | Tier A dev | ~2 GPU-h + ~20 CPU-h | Dimensionality imbalance makes ridge favour one block. Fix with a per-block penalty chosen on val. MAPA's pretraining-session bias on dev. |
| L2 | Val-weighted average of B2/B3/B4 logits, plus a stacked variant. | Tier A dev | ~3 GPU-h | Gains concentrated in subject 7, which has no permitted full session; dev may understate or overstate. |
| L3 | MAPA architecture, two arms on the Tier-B pool, same steps: MSE vs data2vec/JEPA EMA-target latent prediction. Optional third arm: learned per-electrode embeddings (CNF-style). Frozen ridge readout. | Tier B pool, ~10 h + partials | ~50–70 H100-h | Shortened runs may rank objectives differently than full runs. The MAPA reimplementation may not reproduce B2, so check MSE-arm parity against B2's gain over B1 first. |
| L4 | Small head on frozen MAPA tokens, trained on permitted sessions to regress or contrast GPT-2 mid-layer word embeddings (+ volume/pitch). Then add its features to B2. | Tier B pool, transcripts | ~10 A100-h | Targets encode the task labels themselves (surprisal, POS, length are transcript functions), so it is near-supervised. Rules exposure (Q1). |
| L10 | Per test subject, linear map from all-channel MAPA features into the anchor's space, fit by CCA on encoding-model weights to shared stimulus features (word onset, volume, pitch, LM), estimated on permitted sessions. Compare with region-intersection pooling. | Permitted sessions + partials | <10 CPU-h | Dev cross-subject transfer is effectively only subjects 2↔6 (the others have one permitted session). Subjects 7/10 have only ~20 min to fit maps (SUBMIT.md:46). |
| L8 / L12 / L13 | Run only once the facts in §2 are settled. | — | 0 now | — |

## 5. Ranked shortlist

1. **L1 + L2 together, as one Tier-A batch of about 5 GPU-hours.**
   - This targets MAPA's documented blind spot: magnitude-only input. Waveform entries beat it on word_index (+0.017 for DIVER-1 frozen), and linear on raw voltage reaches 0.742 vs 0.651 on spectrogram.
   - It also targets the subject-7 sessions, where Ridge 6-band wins.
   - It reuses released MAPA weights and eval code, so cost is the lowest of any lineage.
   - **[INF]** Expected gain is +0.005–0.012 within-session and cross-session.
2. **L10 cross-subject functional alignment.**
   - CPU-only on the same cache. Cross-subject is the weakest regime (0.608) and discards most channels.
   - A CCA shared space from the MAPA authors' own lab improved cross-patient μECoG decoding.
   - Its dev signal is thin, so treat it as a screen rather than a final answer.
3. **L3 latent-prediction objective.**
   - Clearly allowed. Matched ablations on Brain Treebank (CNF, +0.015–0.026 for contrastive over MSE) and on EEG (EEG-JEPA) favour it over MAPA's MSE target.
   - It costs about 10× L1, so start it only after L1 shows how much headroom an encoder upgrade could add.

L4 has comparable promise but sits behind the answer to Q1.

## 6. Disqualified, one line each

- **L5 (15-label multi-task):** iMINDBench found +0.001 on Neuroprobe from 3–18× more same-task labelled data. On target sessions it also risks using other tasks' trials inside a task's val/test time range. [partly INF]
- **L6 (discrete tokenizers):** TOTEM on Brain Treebank trails continuous iBrain by 0.10–0.23 AUROC on volume, onset and speech. **[INF]** Quantization discards the amplitude detail that gives MAPA its volume/pitch lead.
- **L7 (POYO/POSSM):** there are no spikes to tokenize, and causal real-time decoding has no value here. What remains is patch tokens plus learned unit embeddings, which L3 already tests.
- **L9 (SSM/TSFM):**
  - The leaderboard fixes the 0–1 s window (`config.py:27-28`), so long-context advantages are unusable.
  - Long-context iEEG models could not be adapted to 1-s windows (iMINDBench I.1).
  - Generic time-series FMs underperform when frozen on EEG.
- **L11 (scale only):** DIVER-1's 190× data still scores below MAPA, its frozen Tiny model is best, and its gains saturate between 15.8× and 63×.
- **BaRISTA recipe (within L3):** it lost to DIVER-1 on Neuroprobe in iMINDBench and under matched Brain Treebank pretraining.

## 7. Rules section

| Lineage | Status | Basis |
|---|---|---|
| L1, L2 | Clearly allowed | No pretraining. Session-wide normalization is already used by MAPA and Ridge 6-band. Code must be published (`SUBMIT.md` publication requirement); MAPA weights are Apache-2.0. Choosing ensemble weights on val is **[INF]** allowed. |
| L3, L6, L7, L8, L9, L11 | Allowed if pretraining excludes the 12 sessions | `SUBMIT.md:32-46,125`. Third-party checkpoints must be checked: **CNF-1 is disallowed**, because it trained on "20 sessions from all 10 subjects" and subjects 7 and 10 have only evaluation trials. PHASE's split overlap is unknown. |
| L4 | **Ask maintainers (Q1)** | Labels are transcript-derived. Session 6_0 (permitted) shares *Megamind* with evaluation sessions 4_1 and 7_1, and 1_0/5_0 share *Fantastic Mr. Fox*. Rules restrict neural data, not stimulus annotations. MAPA used readouts on permitted sessions only to choose training steps. |
| L5 on target sessions | **Ask (Q2); likely disallowed** | Per-task fold boundaries differ, so one task's training trials can fall inside another task's val/test span. |
| L10 | Allowed if fit on permitted sessions or unlabelled test-session data; **ask (Q3)** if it uses test-session annotations or val labels for anything beyond hyperparameters | Current entries use val only for regularization. |
| Pre-onset context (L9 variant) | **Ask (Q4); likely disallowed** | `config.py:27-28` window override. |

## 8. Not verified

- **CNF:** protocol equality with the Neuroprobe splits, and code/weight release. I read only the anonymized PDF on azaho.org, which promises release "upon publication"; there is no CNF leaderboard PR (`gh pr list`).
- **NeuroCLUS code status:** only a blog digest says it is unreleased.
- **PHASE:** no code link in its arXiv HTML, and its BTB split overlap with Neuroprobe is unchecked.
- **Evanson:** effect sizes exist only as retrieval ranks; there are no AUROC-comparable numbers.
- **Ensemble gain size:** the leaderboard JSONs hold AUROCs, not per-trial predictions, so I could not compute it.
- **Dev-set noise:** all dev thresholds assume test-like noise; seed variance is unknown.
- **Stimulus media:** audio availability is inferred from the release listing; the density of `movie_frames.zip` was not checked.
- **POYO on iEEG/LFP:** no instance found beyond REALM (LFP).
- **iMINDBench numbers:** the BaRISTA, DIVER-1 and PopT-v2 scores use "Main units", so they are not on the leaderboard's scale.
- **PopT-protocol Brain Treebank numbers** (PHASE, iBrain, NeuroCLUS): mutually inconsistent across papers. PopT is 0.72 onset in iBrain's table and 0.90 in PHASE's.
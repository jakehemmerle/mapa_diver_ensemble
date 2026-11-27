# Synthesis: architecture differences, untried lineages, validation

November 25, 2026. Condensed from `02-architecture-comparison.md` and `03-lineage-survey.md`. Those two reports were written by subagents reading code and papers; their citations have not been independently re-checked.

## Core differences between the top architectures

| Axis | MAPA | DIVER-1 (0.1s tiny, frozen) | Population Transformer |
|---|---|---|---|
| Input | STFT magnitude at three resolutions, 2–160 Hz; phase lost | Raw waveform at 500 Hz; phase kept | STFT magnitude, 0–200 Hz; phase lost |
| Normalisation | Per contact and frequency bin over the whole session (median/MAD) | Fixed scale, no data statistics | "Global" z-score (no code found) |
| Token | Contact × band × frame, 52 per contact per second | Contact × 0.1 s, 10 per contact per second | One token per channel; time collapsed |
| Spatial information | Atlas region embedding plus relative contact position on the shaft | 3D coordinates | 3D coordinates |
| Attention | Joint, but only within one shaft | Joint across all contacts | Time within channel, then across channels |
| Pretraining | Masked reconstruction, 27.9 h of Brain Treebank | Masked reconstruction, 5,310 h external, no Brain Treebank | Discriminative tasks on BrainBERT embeddings |
| Readout | Flatten last layer, closed-form ridge | Flatten, trained linear layer | CLS token, fine-tuned |

All top entries share local re-referencing, no per-window normalisation, bandwidth to at least 150 Hz, and a per-session head selected on the validation half.

Differences most likely to explain the score gaps, in order of confidence:

1. MAPA's pretrained encoder with both spatial encodings adds +0.021 to +0.025 over its own frontend (ablation-backed). Without the encodings it scores below the frontend.
2. Keeping waveform or phase helps word position and delta volume. Linear on raw voltage beats linear on spectrogram by +0.091 on word position.
3. Ridge on frozen high-dimensional features beats fine-tuning. DIVER frozen beats DIVER fine-tuned by +0.012.
4. Region-averaged cross-subject alignment beats Population Transformer's coordinate-based fine-tuning.

Two caveats. Population Transformer's public weights were pretrained on 7 of the 12 evaluation sessions, and the leaderboard entry does not say what it used. MAPA's slow band probably includes about 250 ms of pre-onset signal (the subagent's inference from the code).

## Untried lineages, with verdicts

| Lineage | Verdict | Reason |
|---|---|---|
| Classical features stacked on MAPA (low-frequency waveform, phase, covariance) | Qualified | Targets MAPA's magnitude-only blind spot; about 2 GPU-hours |
| Ensemble of MAPA, Ridge 6-band and DIVER-1 | Qualified | Different winners per task and on subject 7; about 3 GPU-hours |
| Functional cross-subject alignment (CCA / shared response) | Qualified, cross-subject only | Current readout uses only 3–7 shared regions; CPU only |
| Latent-prediction objectives (JEPA, data2vec, contrastive) | Qualified | Matched ablations favour them over MSE; 50–70 H100-hours plus reimplementing MAPA pretraining |
| Stimulus-aligned pretraining to language-model embeddings | Qualified, pending a rules ruling | Five language tasks sit at 0.62–0.64; some permitted sessions share films with evaluation sessions |
| Raw-waveform speech-style encoder | Undecided | Decide after the stacking experiment shows how much the waveform adds |
| PHASE and NeuroCLUS | Undecided | No code or Neuroprobe-protocol result found |
| Multi-task training on the 15 labels | Disqualified | iMINDBench found +0.001 from 3–18× more labelled data |
| Discrete tokenizers | Disqualified | Trail continuous models by 0.10–0.23 on Brain Treebank |
| Spike-tokenized perceivers | Disqualified | No spikes; the remaining idea is covered by the latent-prediction test |
| State-space and general time-series models | Disqualified | The one-second window removes their long-context advantage |
| Scale alone | Disqualified as a first move | DIVER-1 has 190× the data and still scores below MAPA |

## Validation approach

- **Development set.** The 14 permitted pretraining sessions carry the same task annotations and cover 8 subjects, so evaluation-session labels are not needed during development. Cross-session development signal exists only for subjects 2 and 6. Subject 7 has no full permitted session.
- **Contamination.** MAPA pretrained on all of those sessions except 5:0, so comparisons involving MAPA features there are biased and need a confirmation on evaluation sessions.
- **Baselines.** Every candidate must beat reproduced MAPA and a reimplemented Ridge 6-band in the same harness.
- **Statistics.** Subjects are the unit: the two sessions of a subject correlate at 0.81–0.91. Exact sign-flip test over subjects; at least three seeds for any trained arm.
- **Thresholds.** Go at +0.010 with 7 of 8 subjects improving; hold between +0.005 and +0.010; stop below +0.005.
- **Bar for a credible first place.** About +0.011 to +0.015 over MAPA within-session, +0.02 cross-session, +0.025 cross-subject.

## Rules questions

Four questions were open at the time of this synthesis: stimulus annotations from permitted sessions, other tasks' labels on target sessions, validation labels beyond hyperparameters, and pre-onset context. `05-rules-verdicts.md` answers them.

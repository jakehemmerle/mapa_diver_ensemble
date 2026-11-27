# Neuroprobe leaderboard findings

November 25, 2026. Source: the leaderboard at https://neuroprobe.dev, the result JSONs in `azaho/neuroprobe` (`leaderboard/`), unmerged PR #20, and the papers and repositories of each entry. No model was run for this document. The per-task tables come from `analysis/np_agg.py`.

## Standings

Macro AUROC over 15 binary tasks, 12 sessions from 6 subjects, one-second windows after word onset.

| Entry | Within-session | Cross-session | Cross-subject |
|---|---|---|---|
| MAPA (Duke) | 0.695 (#1) | 0.691 (#1) | 0.608 (#1) |
| Ridge 6-band (Purdue, PR #20, unmerged) | 0.683 | 0.673 | 0.585 |
| DIVER-1 0.1s tiny frozen (SNU) | 0.678 (#2) | not submitted | not submitted |
| Population Transformer, global z-scoring (Caltech) | 0.670 (#3) | 0.663 (#3) | 0.575 (#3) |
| CNN Laplacian + spectrogram (Caltech) | 0.669 (#4) | 0.670 (#2) | 0.578 (#2) |
| MAPA frontend only, no encoder (paper, not on the board) | 0.674 | 0.666 | 0.587 |

## What each entry uses

- **MAPA** ([paper](https://arxiv.org/abs/2609.13507), [code](https://github.com/bentang18/MAPA), Apache-2.0). ViT-Small masked autoencoder, 21.3M parameters, 75% masking. Input is STFT magnitude in three bands covering 2–160 Hz, shaft-mean reference, per-session median/MAD normalisation. Spatial encodings are a DKT atlas region embedding and RoPE over contact number and time. Pretrained on 27.9 h of Brain Treebank (13 sessions, 7 subjects), 55k steps, 32 h on 2×H100. Readout is a frozen encoder, last layer flattened, closed-form ridge with the penalty chosen on the validation half. Encoder, four checkpoints and evaluation code are released; the pretraining loop and artifact detectors are not.
- **DIVER-1** ([paper](https://arxiv.org/abs/2512.19097), [code](https://github.com/DIVER-Project/DIVER-1), MIT). 12.7M parameters, width 256, depth 12, raw waveform at 500 Hz in 0.1 s patches, Laplacian re-referencing. Pretrained on 5,310 h of mostly private iEEG with no Brain Treebank data. Linear probe on flattened tokens. Its full fine-tune scores lower (0.665).
- **Population Transformer** ([code](https://github.com/czlwang/PopulationTransformer)). BrainBERT per-channel embeddings feeding a transformer over electrodes, on Laplacian STFT with global z-scoring. The checkpoint behind the leaderboard entry is not identified.
- **CNN baseline.** No pretraining. Reference code is `examples/eval_population.py --classifier_type cnn --preprocess.type laplacian-stft_abs` in the Neuroprobe repository; the exact submission script was not released.
- **Ridge 6-band.** No pretraining and no code; the PR metadata specifies it. Adjacent bipolar pairs, a 1–30 Hz waveform plus theta, alpha, beta, low-gamma log envelopes and 70–170 Hz log high-gamma amplitude, averaged in ten 100 ms bins, ridge readout.

## Where MAPA is beatable

- **The encoder adds little.** Pretraining contributes +0.021 to +0.025 over MAPA's own frontend. Ridge 6-band beats that frontend by about +0.009 with no pretraining.
- **It discards phase.** Within-session, waveform-based entries beat it on word position (DIVER 0.798, Ridge 0.791, MAPA 0.782), delta volume (DIVER 0.803, MAPA 0.800), surprisal and word length. The linear raw-voltage baseline scores 0.742 on word position against 0.651 for linear spectrogram.
- **Subject 7 is weak.** Ridge 6-band beats MAPA on both subject-7 sessions within-session (0.603/0.635 vs 0.589/0.621) and cross-session. MAPA held subjects 7 and 10 out of pretraining and did not use the permitted partial sessions or session 5:0.
- **Unused headroom.** One layer read out, one seed, a small model, under 28 hours of data.
- **Selecting among existing entries is not enough.** The per-task best across current entries is 0.699 / 0.691 / 0.610.

## Within-session per-task scores

| Task | MAPA | Ridge 6-band | DIVER frozen | PopT | CNN | Best |
|---|---|---|---|---|---|---|
| onset | 0.933 | 0.927 | 0.930 | 0.930 | 0.886 | 0.933 |
| speech | 0.917 | 0.906 | 0.897 | 0.919 | 0.888 | 0.919 |
| volume | 0.799 | 0.754 | 0.719 | 0.766 | 0.757 | 0.799 |
| delta_volume | 0.800 | 0.782 | 0.803 | 0.781 | 0.762 | 0.803 |
| pitch | 0.628 | 0.595 | 0.592 | 0.596 | 0.618 | 0.628 |
| word_index | 0.782 | 0.791 | 0.798 | 0.758 | 0.734 | 0.798 |
| word_gap | 0.640 | 0.640 | 0.636 | 0.620 | 0.583 | 0.640 |
| gpt2_surprisal | 0.630 | 0.634 | 0.629 | 0.603 | 0.597 | 0.634 |
| word_head_pos | 0.625 | 0.623 | 0.623 | 0.601 | 0.614 | 0.625 |
| word_part_speech | 0.628 | 0.607 | 0.627 | 0.604 | 0.621 | 0.628 |
| word_length | 0.643 | 0.640 | 0.646 | 0.622 | 0.636 | 0.646 |
| global_flow | 0.674 | 0.638 | 0.628 | 0.621 | 0.638 | 0.674 |
| local_flow | 0.675 | 0.643 | 0.616 | 0.618 | 0.630 | 0.675 |
| frame_brightness | 0.505 | 0.517 | 0.495 | 0.496 | 0.536 | 0.536 |
| face_num | 0.552 | 0.544 | 0.525 | 0.517 | 0.530 | 0.552 |
| Macro | 0.695 | 0.683 | 0.678 | 0.670 | 0.669 | 0.699 |

Cross-session and cross-subject tables: run `python3 analysis/np_agg.py` (expects the clones described in that script's paths).

## Benchmark rules that constrain a new entry

- The official train/val/test splits must be used. In cross-session and cross-subject the val split is the first half of the target session; current entries use it to choose regularization.
- No pretraining on the 12 evaluation sessions. Pretraining is allowed on 14 other Brain Treebank sessions, 5 partial sessions of subjects 7 and 10, and external data.
- A submission needs a reproducible publication or code and a signed attestation.

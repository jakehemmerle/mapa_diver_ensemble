# Neuroprobe top-entry architecture comparison: MAPA, DIVER-1, PopT, CNN, Ridge

I read the code for each model, ran `python3 /tmp/np_agg.py`, and ran a paired per-task/per-session analysis of the leaderboard JSONs (`/tmp/np_ana.py`). I trained nothing and changed no repos. I wrote the DIVER, MAPA and PopT rows with help from three sub-agents and spot-checked their key citations myself.

Path prefixes: M=`/tmp/mapa_probe`, D=`/tmp/diver_probe`, P=`/tmp/popt_probe`, B=`/tmp/brainbert_probe`, N=`/tmp/np_probe`. **Ref code** means N's reference baseline code. The CNN, MLP and PopT result JSONs contain `val_roc_auc` keys, which `N/examples/eval_population.py:312-317` never writes. So those entries came from a different script that was not released, and the CNN column describes the reference code, not the exact submission.

## Side-by-side

| Axis | MAPA | DIVER-1 0.1s tiny (frozen) | PopT (Laplacian STFT, global z) | CNN Laplacian+spectrogram | Ridge 6-band |
|---|---|---|---|---|---|
| Re-reference | Subtracts the shaft mean over surviving Lite contacts (`M/mapa/preprocessing/frontend.py:63-68,251`) | Laplacian, done in an external Neuroprobe fork (`D/scripts/finetune_neuroprobe.sh:50`); pretraining used no reference (DIVER paper App. E.3) | Laplacian that needs **both** n±1 neighbours; other contacts are dropped (`P/data/h5_data_reader.py:52-57,92-109`). That leaves 69–103 of the Lite contacts per subject (my count from `N/neuroprobe/config.py:143`) | Laplacian with 1–2 neighbours; contacts that can't be re-referenced stay raw (`N/examples/eval_utils.py:219-265`) | Adjacent bipolar pairs on a shaft (metadata.json) |
| Filtering, sample rate | 2048 Hz. Zero-phase FIR notches at 60–300 Hz and a 0.5 Hz high-pass on the whole session (`frontend.py:49-60`). Code references then filters; paper App. H says the reverse | 0.5 Hz high-pass and 60/120/180 Hz notches at 2048 Hz (`D/preprocessing/preprocessing_for_neuroprobe.py:39-41`), then 500 Hz | 2048 Hz. Causal IIR notches at 60–360 Hz (`h5_data_reader.py:24,64-69`). The paper's 0.1 Hz high-pass is not in the code | 2048 Hz, **no notch** in `laplacian-stft_abs` (`eval_utils.py:275-287`) | Line noise removed, 200 Hz |
| Input | \|STFT\| with no log, 3 bands: 2–14 Hz (500 ms window), 16–56 Hz (125 ms), 64–160 Hz (62.5 ms). Phase is lost (`frontend.py:33-36,256-270`) | **Raw waveform**, 50-sample patches (`D/models/diver.py:13,77`). CNN embedding plus \|rFFT\| of the *CNN feature* (`D/models/layers.py:58-124`). Phase is kept | \|STFT\|, 400-sample window, 50-sample hop, 0–200 Hz in 40 bins, no log. Phase is lost (`P/preprocessors/stft.py:26,37,57`) | \|STFT\|, 512-sample window, 75% overlap, 0–150 Hz, giving 17×38 per contact, no log (`eval_utils.py:52-113`) | 1–30 Hz **waveform** plus **log** envelopes for θ, α, β, 30–70 Hz, and log high gamma 70–170 Hz |
| Normalization | Per contact×bin over the **whole session**: median and 1.4826·MAD, clipped at ±15/15/20 (`frontend.py:271-274`; `M/mapa/models/stem.py:25-31`) | Fixed scale only (÷2, clip=1, "200 µV"; `D/datasets/datasets_loaders.py:703,749`). No statistics from the data | "Global" z of the STFT. No code exists; the repo only has per-window z (`stft.py:11-18,39-40`) | Training-split StandardScaler per feature (`eval_population.py:248-250`) | Per-channel median/MAD over the session |
| Token | One contact×band×frame: 52 per contact per second, 6,240 total; each attention block is ≤832 tokens (one shaft) (`stem.py:55-95`) | One contact×0.1 s: 10 per contact per second, 1,331 total with special tokens (`D/models/special_token.py:60-62`) | BrainBERT: one STFT frame (≈41/s). PopT: **one token per channel**, the mean of the 10 middle BrainBERT frames (`P/preprocessors/multi_elec_spec_pretrained.py:119-127`). ≤121 total | None; 120 input channels × 17 × 38 | None; 60 features per pair (10 × 100 ms bins × 6 streams) |
| Spatial info | Learned DKT region embedding plus RoPE on the contact index along the shaft (`M/mapa/models/pe.py:9-130`) | Sinusoidal 3D coordinates plus a depth-type embedding (`layers.py:126-182`; `D/models/embedder.py:111-124`) | Sinusoidal integer L/I/P coordinates (`P/models/transformer_encoder_input.py:165-199`) | Each electrode is a conv input channel | Each channel has its own weight |
| Attention | Joint, but **within one shaft only** (`M/mapa/models/attention.py:9-15`) | Joint over all contacts × patches, with an any-variate bias (`layers.py:344-400`) | Factorized: time within a channel (`B/models/masked_tf_model.py:12-23`), then across channels with no time axis | 3×3 conv over time×frequency; all electrodes mix in layer 1 | — |
| Time position | RoPE on a 32 Hz clock (`pe.py:49-88`) | RoPE plus STCPE (`layers.py:359-364`) | BrainBERT 1-D sinusoid; PopT has none | Implicit | Implicit |
| Pretraining | MAE with 75% random masking and MSE on normalized bins (paper §2; the loop is not released) | 50% patch masking; targets are raw MSE plus log1p FFT amplitude (`diver.py:15,80-88`) | BrainBERT: masked spectrogram, L1 loss. PopT: consecutive-segment and replaced-token *discrimination* (`P/criterions/nsp_replace_only_pretrain.py:367-384`) | — | — |
| Pretraining data | Brain Treebank only: 27.9 h, 13 allowed sessions, 7 subjects. Excludes subjects 7 and 10 and their partial sessions (paper App. A) | 5,310 h from 37 subjects (AJILE12 + private); no Brain Treebank (paper Table 5) | Brain Treebank, 10 subjects, 19 sessions in the public weights (`P/trial_selections/pretrain_split_trials.json`). This **includes 7 of the 12 evaluation sessions**; the leaderboard entry does not say what it pretrained on | — | — |
| Size | 21.3M parameters, width 384, 12 layers (`M/MODEL_CARD.md:63-68`) | 12.7M parameters, width 256, 12 layers | About 20M (6 layers × 512) plus a frozen BrainBERT, about 43M [INFERENCE] | About 0.39M [INFERENCE, computed from `eval_utils.py:543-553`] | About 6.4k weights [INFERENCE] |
| Readout | Frozen. Block-12 output, flattened to about 2.4M dims, closed-form ridge; λ picked from a 25-point grid by **validation AUROC** (`M/evals/neuroprobe/readout.py:162-221`) | Frozen. Flattened to 307k dims, then `nn.Linear` trained with AdamW and early-stopped on validation AUROC (`D/models/finetune_model.py:52-65`) | CLS token → `Linear(512,1)`, fine-tuned for 1000 steps (Neuroprobe paper App. D). Frozen vs fine-tuned is not stated for this entry | Trained end to end, early-stopped on validation AUROC (`eval_utils.py:573-669`) | Ridge; λ picked by validation AUROC |
| Cross-subject | Encoder features averaged per DKT region; 3–7 shared regions per subject pair; anchor session 2:4 (`readout.py:230-311`) | Not submitted | Coordinates only (Neuroprobe paper App. D) | DK-34 region averaging (`eval_utils.py:856-903`) | DKT region averaging |
| Score (within / cross-session / cross-subject) | 0.695 / 0.691 / 0.608 | 0.678 / – / – | 0.670 / 0.663 / 0.575 | 0.669 / 0.670 / 0.578 | 0.683 / 0.673 / 0.585 |

## Shared by all (probably table stakes)
- **Local re-referencing.** Laplacian, bipolar or shaft mean. Two sources back this up:
  - Linear spectrogram → Linear Laplacian is the same pipeline with only the Laplacian added: +0.035 / +0.025 / +0.011.
  - DIVER Table 13: speech goes from 0.862 to 0.900 with the Laplacian.
- **No per-window normalization.** PopT with global z vs per-window z: 0.670 vs 0.549 within-session (Neuroprobe paper, Supp. Tables). BrainBERT shows the same: 0.626 vs 0.585. This is the clearest result an ablation backs.
- **Bandwidth reaches at least 150 Hz** (high gamma), and the readout is a supervised head per target session, chosen or stopped on the official validation half.
- **One cross-subject anchor.** Every cross-subject entry trains on session 2:4 (`N/neuroprobe/config.py:37`).

## Unique to one
- **MAPA:**
  - in-domain MAE pretraining;
  - attention only within a shaft;
  - three STFT resolutions;
  - region embedding plus contact-index RoPE;
  - a readout that flattens about 2.4M dims.
  - The STFT is computed on the continuous session with `center=True`, and clips start at the onset frame (`M/evals/neuroprobe/encode.py:121-144`; `M/mapa/probe/labels.py:30`). So the first slow-band frame covers −250 to +250 ms around onset [INFERENCE]: some pre-onset signal gets in.
- **DIVER:** raw-waveform tokens in a pretrained model, 3D coordinates, joint attention across all contacts, a large out-of-domain corpus, and no per-session normalization.
- **PopT:**
  - time is collapsed to one token per channel;
  - pretraining discriminates rather than reconstructs;
  - contacts without two neighbours are dropped;
  - the public weights overlap 7 evaluation sessions.
- **CNN:** dense spatial mixing in the first layer and no line-noise removal.
- **Ridge:** log compression, an explicit low-frequency waveform stream, 100 ms averaging, and no learned features.

## Ranked differences most likely to explain the gaps

1. **MAPA's pretrained within-shaft encoder on a multi-resolution spectral frontend. Confidence: high.**
   - Ablation (MAPA paper Table 5): the encoder adds +0.021 / +0.025 / +0.021 over the same frontend with the same ridge readout.
   - With neither the region nor the relative-position encoding, the encoder scores −0.013 below the frontend, so both are needed.
   - Gains are largest on cross-subject onset (+0.070) and speech (+0.060), on word_index (+0.034 to +0.050) and on the optical-flow tasks (+0.026 to +0.032 within-session) (paper Table 6).
   - Caveat: one pretraining seed, and the cross-subject result is not significant (q=0.0625).
2. **Keeping the waveform or phase lifts word_index, delta_volume and the word-timing tasks. Confidence: medium-high** (near-ablation plus cross-entry pattern; the mechanism is inference).
   - Raw-voltage vs spectrogram linear models share one pipeline. Raw gains +0.091 on word_index and +0.031 on delta_volume while losing 0.021 overall.
   - word_index is sentence-initial word vs second word (`N/neuroprobe/datasets.py:183-189`) [INFERENCE: an evoked response to sound onset after silence].
   - Ridge beats the MAPA frontend on word_index (+0.043), word_length (+0.024) and gpt2_surprisal (+0.019), and still beats full MAPA within-session on word_index (−0.009 for MAPA). DIVER, also raw-input, has the best within-session word_index and delta_volume (0.798–0.814).
   - Subject 7 is the only subject where Ridge beats MAPA in all three regimes. Its within-session gap is dominated by word_index (0.69 vs 0.60 and 0.65 vs 0.58).
   - [INFERENCE] A second factor for subject 7: it has 8 long shafts (median 16 contacts), so MAPA's shaft-mean reference acts almost like a common-average reference, while Ridge's bipolar pairs stay local.
3. **Readout regularization picked on validation, applied to high-dimensional frozen features. Confidence: medium.**
   - The two entries that tune a ridge λ on validation AUROC are #1 and #2 in every regime. The MAPA frontend alone (no learned weights) scores 0.674, above PopT and the CNN.
   - Within one model, DIVER frozen beats DIVER fine-tuned by +0.012 (SE 0.002), with the gain mostly on volume, pitch and flow (+0.03 each).
   - This mixes evidence types: the DIVER pair is close to an ablation, while the ridge pattern is cross-entry and confounded with the features.
4. **How cross-subject alignment is done. Confidence: medium (cross-entry).**
   - Region averaging plus a linear readout (MAPA, Ridge, CNN) beats PopT's coordinate-based fine-tuning on cross-subject onset (0.711 vs 0.78–0.815) and speech.
   - MAPA's region embedding alone adds +0.018 cross-subject (ablation, not significant).
   - MAPA's largest margin over Ridge is in the cross-subject regime (+0.024).
5. **Spectral detail: fast high-gamma resolution, no log, robust session z. Confidence: low-medium (inference).**
   - The MAPA frontend (62.5 ms windows, 32 Hz frames) beats Ridge (100 ms bins) on volume by +0.023, and full MAPA leads every entry on volume, pitch and flow.
   - DIVER, with no session normalization, does worst on volume (0.719).
6. **PopT throws away time and contacts. Confidence: low-medium.**
   - PopT is 0.02–0.025 below MAPA, DIVER and Ridge on every within-session word task. One token per channel removes within-window dynamics, and the two-neighbour rule drops 15–30% of contacts.
   - Contamination check: the PopT−CNN gap is no larger on the 7 overlapping sessions than on the 5 clean ones (within-session −0.002 vs +0.006; cross-session −0.017 vs +0.005). So overlap doesn't explain its score.

## Axes no entry explores
- A pretrained model whose input carries both session-normalized power *and* phase (complex STFT or analytic signal). Neuroprobe's own code supports `stft_absangle` and `stft_realimag` (`eval_utils.py:95-104`), but no entry uses them.
- In-domain pretraining with attention across shafts. MAPA blocks it; DIVER has it but is out-of-domain.
- Pretraining on the allowed partial sessions of subjects 7 and 10. MAPA excluded them, and subject 7 is its weak spot.
- Structured masking (by contact, shaft or time block), or any masking-ratio, width or depth ablation in-domain.
- Log compression in a pretrained model.
- Readouts shared across tasks or sessions, and stacking feature families. The best-per-task ceiling of 0.699 means gains must be broad.
- Explicit control of pre-onset context, which is currently leaked implicitly by MAPA.
- Transductive use of target-session statistics: MAPA and Ridge normalize with whole-session statistics, which include the unlabeled test period. Nobody studies this.

## Could not verify (and what I tried)
- **Ridge 6-band:** no code exists, only `metadata.json`. Filter types, envelope method and λ grid are unknown.
- **CNN, MLP and PopT submission scripts:** the JSON keys show a script that was not released.
  - The CNN architecture and its normalization come from the reference code only.
  - PopT's global-z code, window length, frozen vs fine-tuned setting, and whether it was re-pretrained without evaluation sessions: I grepped both repos for `global`, `neuroprobe` and `z-scor` and found none of it.
- **DIVER:** the Laplacian, 500 Hz resampling and clip step live in a Neuroprobe fork that isn't in the repo, and the frozen-run command was never released (the released script sets `--frozen False`).
- **MAPA:** the pretraining loop is not released, and the order of reference and filter differs between code and paper. The size of the pre-onset effect is not ablated anywhere.
- **Neuroprobe paper vs code:** the paper says sklearn `LinearRegression`; the code uses `LogisticRegression` with default C (`eval_population.py:257`).
# Separation Backend Experiments

Status: **Phase 1 complete** (2026-03-14)
Decision: Integrate DrumSep only. Defer vocal model replacement.

## What we tested

### Models

| Model | Filename | Architecture | Primary output | Published SDR |
|---|---|---|---|---|
| Demucs htdemucs (baseline) | `htdemucs.yaml` | Demucs v4 | vocals/drums/bass/other | V:9.9 D:9.4 B:11.6 |
| MelBand Roformer Vocals | `vocals_mel_band_roformer.ckpt` | MDXC | vocals + other | V:12.6 |
| BS-Roformer Viperx-1296 | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | MDXC | vocals + instrumental | V:12.1 I:16.3 |
| MelBand Roformer Bleedless | `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt` | MDXC | vocals + other | Unknown |
| kuielab_b_drums | `kuielab_b_drums.onnx` | MDX | drums + no drums | D:7.1 |
| DrumSep | `MDX23C-DrumSep-aufr33-jarredou.ckpt` | MDXC | kick/snare/toms/hh/ride/crash | N/A |

### Songs tested

| Song | Character | Demucs stems existed |
|---|---|---|
| Gorillaz - Feel Good Inc (feat. De La Soul) | Dense electronic/hip-hop mix | Yes |
| Arctic Monkeys - Do I Wanna Know? | Sparse guitar-driven rock | Yes |

### Tool

- `audio-separator` v0.41.1 (`pip install "audio-separator[cpu]"`)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA via PyTorch, ONNX CPU-only)
- Python 3.10.12, PyTorch 2.10.0+cu128

## Results

### Experiment A: Vocal front-end bake-off

#### Feel Good Inc (dense mix)

| Metric | Demucs | MelBand Roformer | BS-Roformer | Bleedless |
|---|---|---|---|---|
| Voiced frames | 3200 | **3371** | 3346 | 3348 |
| Pitch jumps >3st | **24** | 31 | 29 | 27 |
| Pitch jumps >5st | 11 | 6 | 6 | **4** |
| Vocal↔instr RMS corr | **0.043** | 0.078 | 0.066 | 0.076 |

Bleedless won on this song: fewest extreme pitch jumps, near-best voiced coverage.

#### Do I Wanna Know (sparse mix)

| Metric | Demucs | Bleedless |
|---|---|---|
| Voiced frames | **6026** | 5325 (-11.6%) |
| Pitch jumps >3st | **57** | 64 |
| Pitch jumps >5st | **32** | 38 |
| Vocal↔instr RMS corr | 0.164 | **0.147** |

Demucs won on this song. Bleedless lost 700 voiced frames and had more pitch jumps.

#### Vocal conclusion

**Song-dependent.** Bleedless works better on dense mixes where cross-stem bleed is the main problem. On sparse mixes, it over-strips legitimate vocal content. No single model wins everywhere.

Listening tests confirmed: Roformer-family models produce noticeably cleaner vocals on Feel Good Inc, but the advantage is less clear on Do I Wanna Know.

### Experiment B: DrumSep representation

#### Feel Good Inc

| Band | Heuristic↔DrumSep correlation |
|---|---|
| Kick | 0.88 |
| Snare | 0.63 |
| Hats | **0.09** |

Kick onset sharpness: 32.7 (DrumSep) vs 13.6 (full drum stem) = 2.4x sharper.
Kick energy at kick onsets: 0.549 (DrumSep) vs 0.357 (heuristic) = 1.5x more defined.

#### Do I Wanna Know

| Band | Heuristic↔DrumSep correlation |
|---|---|
| Kick | 0.86 |
| Snare | 0.60 |
| Hats | 0.57 |

Kick onset sharpness: 29.5 (DrumSep) vs 20.5 (full drum stem) = 1.4x sharper.
Kick energy at kick onsets: 0.543 (DrumSep) vs 0.169 (heuristic) = 3.2x more defined.

#### DrumSep conclusion

**Consistent win on both songs.** The frequency-band heuristic's hats band is essentially wrong on dense mixes (r=0.09). Kick detection is dramatically sharper and more defined with real component separation. The improvement holds across both song types.

## Decision

- **Integrate now:** DrumSep as optional post-pass on Demucs drum stem
- **Keep:** Demucs htdemucs as default separation backend for all four stems
- **Defer:** Vocal model replacement (needs per-song model selection strategy)

## What remains unresolved

### Vocal separation
- No single model beats Demucs on all song types
- Bleedless is promising on dense mixes but harmful on sparse ones
- Possible approaches for later:
  - Mix density classifier → model selector (bleedless for dense, demucs for sparse)
  - Vocal-first cascade: run specialized vocal model, subtract from mix, then run Demucs on remainder
  - Test additional models optimized for sparse vocals (e.g., `mel_band_roformer_vocal_fullness_aname.ckpt`)
  - Test `htdemucs_ft` (fine-tuned Demucs, V:10.8 D:10.0 B:12.0) as a low-risk upgrade

### Demucs variant comparison
- `htdemucs_ft` was identified but never locally validated on our songs
- `hdemucs_mmi` (V:10.2 D:9.6 B:12.2) also untested
- Should be tested before any Demucs variant swap

### DrumSep deeper evaluation
- Per-component onset timing accuracy not yet compared against manual annotations
- Ride/crash/toms separation quality not deeply evaluated (only kick/snare/hh mapped to our 3-band scheme)
- Could explore richer than 3-band representation using all 6 components

## Reproducibility

### Commands used

```bash
# Install
.songviz/venv/bin/python -m pip install "audio-separator[cpu]"

# Vocal separations (Feel Good Inc)
.songviz/venv/bin/audio-separator "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac" \
  --model_filename vocals_mel_band_roformer.ckpt \
  --output_dir experiments/results/feel_good_inc/melband_roformer/ --output_format WAV

.songviz/venv/bin/audio-separator "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac" \
  --model_filename model_bs_roformer_ep_368_sdr_12.9628.ckpt \
  --output_dir experiments/results/feel_good_inc/bs_roformer_viperx/ --output_format WAV

.songviz/venv/bin/audio-separator "songs/Gorillaz - Feel Good Inc (featuring De La Soul).flac" \
  --model_filename mel_band_roformer_kim_ft2_bleedless_unwa.ckpt \
  --output_dir experiments/results/feel_good_inc/melband_bleedless/ --output_format WAV

# DrumSep (run on Demucs drum stem, not full mix)
.songviz/venv/bin/audio-separator "outputs/Gorillaz - Feel Good Inc (featuring De La Soul)/stems/drums.wav" \
  --model_filename MDX23C-DrumSep-aufr33-jarredou.ckpt \
  --output_dir experiments/results/feel_good_inc/drumsep/ --output_format WAV

# Do I Wanna Know (Bleedless + DrumSep only)
.songviz/venv/bin/audio-separator "songs/Arctic Monkeys - Do I Wanna Know_.flac" \
  --model_filename mel_band_roformer_kim_ft2_bleedless_unwa.ckpt \
  --output_dir experiments/results/do_i_wanna_know/melband_bleedless/ --output_format WAV

.songviz/venv/bin/audio-separator "outputs/Arctic Monkeys - Do I Wanna Know_/stems/drums.wav" \
  --model_filename MDX23C-DrumSep-aufr33-jarredou.ckpt \
  --output_dir experiments/results/do_i_wanna_know/drumsep/ --output_format WAV
```

### Comparison scripts

- `experiments/compare_stems.py` — Feel Good Inc (all 4 vocal models + DrumSep)
- `experiments/compare_stems_diwk.py` — Do I Wanna Know (Demucs + Bleedless + DrumSep)

### Output locations

```
experiments/results/
├── feel_good_inc/
│   ├── comparison.json          # v1 metrics (Demucs vs MelBand+kuielab)
│   ├── comparison_v2.json       # v2 metrics (4 vocal models + DrumSep)
│   ├── demucs_htdemucs/         # pitch_track.json, onset_times.json, drum_band_energy.npy
│   ├── melband_roformer/        # vocals.wav, instrumental.wav, pitch_track.json
│   ├── bs_roformer_viperx/      # vocals.wav, instrumental.wav, pitch_track.json
│   ├── melband_bleedless/       # vocals.wav, instrumental.wav, pitch_track.json
│   ├── kuielab_b_drums/         # drums.wav, no_drums.wav, onset_times.json, drum_band_energy.npy
│   └── drumsep/                 # kick.wav, snare.wav, hh.wav, toms.wav, ride.wav, crash.wav
│                                # drumsep_3band_energy.npy, heuristic_3band_energy.npy
│                                # onset_times_{component}.json
└── do_i_wanna_know/
    ├── comparison.json
    ├── demucs_htdemucs/         # pitch_track.json
    ├── melband_bleedless/       # vocals.wav, instrumental.wav, pitch_track.json
    └── drumsep/                 # same structure as above
```

### Metric definitions

- **Voiced frame ratio:** fraction of pYIN frames with valid (non-NaN) pitch
- **Pitch jump count (Nst):** frames where pitch changes by more than N semitones between consecutive voiced frames
- **Vocal↔instrumental RMS correlation:** Pearson correlation between RMS envelopes of vocal and instrumental stems (leakage proxy)
- **Onset sharpness:** peak-to-mean ratio of librosa onset_strength envelope
- **Band correlation:** Pearson correlation between heuristic frequency-band energy and DrumSep component-based energy per band
- **Kick energy at onsets:** mean/median of energy curve value at detected kick onset times

### Runtimes (GPU — RTX 4050)

| Operation | Feel Good Inc | Do I Wanna Know |
|---|---|---|
| Bleedless vocal | ~33s separation | ~41s |
| BS-Roformer vocal | ~1m19s | not run |
| DrumSep on drum stem | ~1m07s | ~1m22s |
| MelBand Roformer vocal | ~13m (CPU), ~33s (GPU est.) | not run |

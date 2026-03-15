> **Reference material.** Canonical pipeline: `docs/05_lyrics_playbook.md`. This file covers alternatives not yet implemented.

# Extracting Lyrics With Word and Phoneme Timestamps From Singing Audio

## Executive summary

Accurate lyric timing for singing is usually **not a single-tool problem**. The most reliable (free/open-source) approach is a **two-stage pipeline**: (a) **transcribe** (or supply known lyrics), then (b) **force-align** to obtain **word + phoneme boundaries**. Speech-trained tools can work surprisingly well on clean isolated vocals, but **melisma, vibrato, breaths, backing vocals, and long sustained vowels** still cause systematic errors; specialized *singing* aligners can outperform generic speech aligners when their language/assumptions match your material. citeturn14search16turn25search6turn7search20

For “implementable today” pipelines, the best general-purpose recommendation for most users is:

- **Whisper → manual lyric cleanup (optional but strongly recommended) → Montreal Forced Aligner (MFA)** to produce **Praat TextGrid** (phones + words) and/or JSON/CSV outputs. Whisper is robust and easy to run locally; MFA produces explicit **word and phone tiers** and supports model/dictionary/G2P workflows, but it is **not designed for single long files or singing**, so you’ll typically get better results by splitting into utterances and, when needed, adapting/training models. citeturn29view0turn27view2turn23search3turn6view0turn23search20

For higher accuracy on singing alignment (especially when you already have lyrics), two open-source options are worth special attention:

- **lyrics-aligner** (phoneme-level lyrics alignment + text-informed singing voice separation; MIT) — designed for singing mixtures and can output phoneme/word onsets; good when your music resembles its training conditions (notably MUSDB-like material). citeturn24view0turn25search2  
- **SOFA (Singing-Oriented Forced Aligner)** (MIT) — a singing-oriented forced aligner that converts transcripts to phonemes via G2P/dictionary and exports **TextGrid/HTK** plus optional confidence; however, its default resources/example usage are strongly oriented to **Chinese/pinyin-style inputs** and may not directly fit other languages without custom G2P/dicts/models. citeturn8view2turn9search0

Unspecified details that materially affect “best tool” choice: **language(s)**, whether you have **ground-truth lyrics text**, how much **melisma** (one syllable stretched over many notes), presence of **backing vocals**, amount of **reverb/delay**, and whether you need **phoneme offsets** (full durations) vs only **onsets**. (Defaults assumed below: you can provide or obtain lyrics text, lead vocal dominant, moderate reverb, one main language, and you want both phone start/end times.)

## Candidate tools and web options prioritized for singing lyrics alignment

The table below emphasizes tools that can produce **word timestamps** and/or **phoneme-level timing**, and notes how they behave on singing (vibrato, consonants, backing vocals). “Accuracy” is relative and depends heavily on: (1) whether you already have correct lyrics, (2) vocal isolation quality, and (3) language support. citeturn14search16turn25search6turn23search3turn10view3

| Tool / pipeline (priority order) | What it’s best at | License | Inputs | Outputs | Dependencies / setup | Ease | Singing-specific behavior (notes) |
|---|---|---|---|---|---|---|---|
| **lyrics-aligner** citeturn24view0 | Singing-focused alignment (phoneme/word onsets) from provided lyrics; includes a VAD-like option via vocals estimate | MIT citeturn24view0 | Audio formats supported by librosa; lyrics as words or phonemes citeturn24view0 | Phoneme and/or word **onsets** (and alignment artifacts) | Conda env (CPU/GPU), PyTorch stack; uses librosa citeturn24view0 | Medium | Explicitly designed for audio containing solo singing or singing in mixtures; expects 39 ARPAbet phonemes if phoneme input; can use VAD threshold to handle long instrumental gaps citeturn24view0 |
| **SOFA (Singing-Oriented Forced Aligner)** citeturn8view2 | Singing-oriented forced alignment with G2P + confidence export | MIT citeturn9search0 | `.wav` + transcript `.lab` per segment citeturn8view2 | TextGrid / HTK / “trans” (+ confidence) citeturn8view2 | Conda + PyTorch; dictionary/G2P modules citeturn8view2 | Medium | Strongly tuned to singing; default example/dict uses pinyin-like input; best when your language + phoneme inventory match its resources citeturn8view2turn9search0 |
| **Whisper → (optional) WhisperX** | Fast local transcription; WhisperX improves word-level timestamps by adding VAD + forced alignment with phoneme model | Whisper: MIT citeturn3search6turn3search2; WhisperX: BSD-2 citeturn11view1turn6view1 | Whisper reads FLAC/MP3/WAV citeturn27view3 | Whisper: text + segment timestamps; WhisperX: improved word-level timestamps citeturn6view1turn26view1 | PyTorch + ffmpeg; WhisperX adds VAD + wav2vec2 alignment models citeturn27view2turn6view1turn26view1 | High | Great for **getting the lyric text**; word timestamps can drift on singing/mixtures; best on isolated vocals; still not a true phoneme-boundary tool unless you add a forced aligner after citeturn23search3turn25search6turn6view1 |
| **Montreal Forced Aligner** citeturn6view0 | **Phone + word boundaries** (TextGrid/JSON/CSV) given transcript + dictionary; can use pretrained models and G2P workflows | MIT citeturn11view0turn6view0 | Corpus of audio + text files (TextGrid input supported in some workflows) citeturn23search4turn23search29 | TextGrid w/ word & phone tiers; JSON/CSV output formats available citeturn23search29turn23search5turn27view2 | Conda install; Kaldi-based internals citeturn2search26turn6view0 | Medium | Produces explicit phone tiers (excellent for phonemes), but docs warn it’s not intended for single long files or atypical styles such as **singing**—segmenting/adapting helps citeturn23search3turn27view2turn23search20 |
| **Gentle** citeturn10view2 | “Lenient” forced alignment (handles imperfect transcripts); returns per-word timings plus per-phone durations | MIT citeturn10view2 | Audio (e.g., mp3) + transcript text citeturn10view2turn22view1 | JSON with word start/end and phone durations per word citeturn18view0turn22view1 | Docker or install.sh; internally resamples to **8 kHz** and runs Kaldi pipeline citeturn15view0turn22view1 | Medium | Resampling to 8 kHz can reduce sibilant detail, but alignment is robust/lenient; phoneme durations exist (phone times are reconstructable within each word) citeturn22view1turn18view0 |
| **WebMAUS (BAS / CLARIN web service)** | Convenient **web-based** forced alignment including phonemes/words/sentences; good for quick trials & manual correction workflows | Free web service (not fully open-source as a packaged aligner) | Audio + transcript upload citeturn1search14turn1search37 | Time-aligned phonemes/words/sentences (Praat-friendly) citeturn1search14turn1search18 | Browser upload; privacy constraints (uploads to third party) citeturn1search22 | High | Useful sanity-check baseline; may degrade on heavy music beds; best on isolated vocals; be mindful of consent/privacy and retention terms citeturn1search22turn1search14 |
| **SPPAS** citeturn4search8 | Phonetic/word/syllable annotations from transcription; strong tooling ecosystem + file-format support | AGPLv3 citeturn10view3turn9search3 | Audio + manually prepared orthographic transcription (it explicitly is *not* ASR) citeturn2search39turn7search3 | Praat TextGrid + many formats; can produce syllabification from aligned phonemes citeturn7search23turn4search0 | Python package + external tools; rich formats | Medium | Primarily for speech; can still be used for singing if transcript is clean and vocals are isolated, but expect failures on melisma/backing vocals citeturn2search39turn7search3 |
| **Aeneas** citeturn8view3 | Text/audio “sync map” via TTS+DTW-like approach; quick word/phrase alignment when ASR models are poor | AGPLv3 citeturn8view3turn4search3 | Audio + text fragments citeturn4search35turn14search3 | Sync maps / subtitle-like outputs; word-level possible but not the design target citeturn8view3turn4search35 | Python/C; requires TTS engines for best results | Medium | Authors explicitly note audio is assumed to be spoken and **not suitable for song captioning** (YMMV). Use mainly as fallback. citeturn8view3turn14search3 |
| **Torchaudio forced alignment / CTC segmentation family** | Modern CTC forced alignment using wav2vec2/HuBERT/MMS; can handle missing tokens via `<star>` in some APIs | Open-source (PyTorch ecosystem) | Audio + transcript citeturn2search1turn14search18 | Frame-to-token alignment paths; can derive token/word times citeturn2search1turn14search18 | Python + torch; note API deprecations in torchaudio 2.8/2.9 citeturn2search5 | Medium | Potentially strong for multilingual alignment; singing still hard unless phoneme models match singing acoustics; watch torchaudio forced-align API deprecation path citeturn2search5turn2search1 |

Notable free (but not open-source) web/API demos: major cloud ASR providers sometimes offer free tiers, but **phoneme-level timing is typically not exposed** and alignment quality on singing varies. Use them mainly to bootstrap rough transcripts, and keep legal/ToS constraints in mind. (No single authoritative “best free tier” applies across regions and dates, so treat this as optional.) citeturn14search16turn3search0

## Algorithms and approaches that matter for singing alignment

### ASR vs forced alignment: why you usually need both

- **ASR** predicts text from audio; even strong models can struggle with singing because vowels are prolonged, consonants are reduced, and rhythm is constrained by melody. Research and benchmarks treat **lyrics transcription** as a specialized challenge distinct from speech ASR. citeturn14search16turn7search24turn7search20  
- **Forced alignment** assumes you already know (or hypothesize) the transcript and finds **where** in the audio each unit occurs. This is typically how you get **phoneme boundaries**: you map words → phonemes via a pronunciation dictionary (or G2P), then align phoneme sequence to acoustics. citeturn6view0turn27view2turn23search4

In practice: **use ASR to get a draft transcript**, correct it (or supply known lyrics), then **force-align** to obtain high-resolution timings. citeturn6view0turn25search6

### Classical forced alignment: GMM/HMM, MFCCs, and why it can break on singing

Many forced aligners (including MFA) are built on an ASR pipeline with **MFCC features** and a **GMM/HMM** acoustic model; MFA’s published description includes 25 ms windows and 10 ms frame shifts, and uses Kaldi with triphone modeling and speaker adaptation. citeturn6view0turn4search9

Strengths:
- Produces explicit **phone boundaries** (exactly what you need for phoneme-level timing). citeturn23search29turn13search5  
- Works well when transcript and pronunciations are correct and acoustics resemble training conditions (speech). citeturn6view0turn23search4

Weaknesses on singing:
- Long sustained vowels can “smear” boundaries; consonants may be too short/soft to anchor.  
- Background vocals and reverb cause competing phonetic evidence.  
- MFA docs explicitly warn it is not intended to align single long files and “different style such as singing,” and recommend strategies like aligning larger sets (speaker adaptation) when possible. citeturn23search3turn27view2

### Neural alignment: CTC segmentation, wav2vec2-style acoustic models, and WhisperX

Modern pipelines often use a strong ASR model plus **CTC-based forced alignment** or other neural alignment strategies.

- **CTC segmentation / CTC forced alignment**: compute framewise token posteriors (often from wav2vec2/HuBERT/MMS) and then align a transcript to those posteriors via dynamic programming; torchaudio’s tutorial explicitly uses **CTC segmentation** and provides a forced alignment API, including a mechanism to handle missing transcript regions via a `<star>` token. citeturn2search1turn2search5turn14search18  
- **WhisperX**: explicitly adds (i) VAD pre-segmentation, (ii) cut-and-merge into ~30s chunks, and (iii) forced alignment using an external phoneme model to yield accurate word timestamps (as described in its paper figure and text). citeturn6view1turn6view1turn26view1

Strengths for singing:
- If you have a phoneme/character model that generalizes to singing, CTC alignment can be resilient to mild transcript errors (especially with “blank” capacity).  
- VAD-style segmentation can reduce timestamp drift by avoiding long music-only spans. citeturn6view1turn25search6turn25search6

Weaknesses for singing:
- Alignment quality is dominated by whether the acoustic model “understands” singing phonetics; speech-only phoneme models still confuse melisma and breathy onsets. citeturn14search16turn7search20

### Phoneme recognition and “text-independent” alignment

Some tools aim to recognize phonemes directly from audio, then align without a strict lexicon:

- **Charsiu** advertises phoneme recognition, forced alignment, and “text-independent alignment” capabilities (neural phonetic aligner) — useful when lyrics text is unreliable, but accuracy depends heavily on domain match and language. citeturn14search2

### Grapheme-to-phoneme (G2P) and pronunciation resources: crucial for phoneme-level timing

Phoneme timestamps depend on a correct mapping from words → phonemes:

- **MFA** includes workflows for G2P training and generation; its docs describe training G2P models (via FSTs / Pynini) and using G2P to generate pronunciations for OOV words. citeturn12search11turn27view2  
- General-purpose phonemization tools include **Phonemizer** (supports multiple backends such as espeak-ng with IPA output) and **Epitran** (MIT; transliteration to IPA for many languages). citeturn12search0turn12search2

Singing-specific complication: performers often modify pronunciations (dropped consonants, stylized vowels). A “canonical” lexicon may be slightly wrong, which can shift boundaries; some aligners provide “unknown word/noise” handling or confidence outputs to flag unreliable regions. citeturn6view0turn7search10turn8view2

## Practical pipelines for isolated vocal FLAC or full mix

### Recommended preprocessing (with defaults you can tune)

Because singing alignment is fragile, preprocessing should aim to (1) maximize vocal intelligibility, (2) avoid distorting consonant bursts, and (3) keep the audio consistent with model expectations.

**Audio decode + normalization**
- Convert FLAC/full mix to a consistent working format (mono or dual-mono), and resample to model-native sample rates where possible; Whisper’s own loader resamples to **16 kHz mono** via ffmpeg, and hard-codes 30s chunks with 10 ms frames (hop length 160 at 16 kHz). citeturn29view0turn27view2  
- Gentle internally resamples for alignment (its CLI logs “converting audio to 8K sampled wav,” and the code uses an 8 kHz pipeline), so feeding it high-rate audio won’t preserve extra high-frequency detail. citeturn15view0turn22view1

**Isolated vocals vs full mix**
- If you have isolated vocals (e.g., Demucs stems), prefer them: benchmarks and recent work show source separation can improve lyric transcription and can be used as VAD/segmentation guidance. citeturn25search6turn25search2

**Denoise / dereverb (light touch)**
- Use mild denoise/dereverb if the vocal stem has “warbly” artifacts; aggressive denoise can destroy consonant cues and harm alignment. (This is a practical heuristic; the strong evidence is that alignment models rely on phonetic cues and VAD boundaries, so preserving attacks matters.) citeturn6view1turn23search3

**De-essing**
- Apply only if sibilants are severely exaggerated by separation. Over-de-essing can erase /s/ and /t/ transients (key anchors for word boundaries). Confidence outputs (SOFA `--save_confidence`, or MFA phone confidence utilities) are useful to test whether you helped or harmed alignment. citeturn8view2turn7search10

**Segmentation**
- Split long songs into short vocal regions before forced alignment. WhisperX’s design explicitly uses VAD + chunking to avoid drift, and MFA notes single-file alignment is not its core design. citeturn6view1turn23search3

### Mermaid flowchart: end-to-end “lyrics + phonemes + pitch per word” pipeline

```mermaid
flowchart TD
  A[Input audio\n- isolated vocal FLAC (preferred)\n- or full mix] --> B[Decode & resample\nffmpeg -> mono\n16 kHz for ASR\n(optional: keep 44.1/48k for pitch)]
  B --> C{Do you have lyrics text?}
  C -- Yes --> D[Normalize lyrics text\n(case, punctuation, repeats)\n(optional: mark backing vocals)]
  C -- No --> E[ASR transcription\nWhisper (local)\n+ manual correction pass]
  E --> D
  D --> F[Forced alignment\nChoose:\n- MFA (word+phone tiers)\n- Gentle (lenient JSON)\n- lyrics-aligner/SOFA (singing-oriented)\n- WebMAUS (web)]
  F --> G[Post-process alignments\n- convert TextGrid/CTM/JSON -> unified JSON\n- derive syllables (optional)\n- compute confidence flags]
  B --> H[Prosody extraction\nPitch & vibrato (pYIN/torchcrepe/Praat)\nEnergy per word]
  H --> I[Align pitch to words/phones\naggregate stats per word/phoneme]
  G --> J[Final JSON schema\nwords, phonemes, confidence,\nsyllables, pitch contour per word]
  I --> J
```

Primary rationale: ASR is mainly for **text**, forced alignment is for **timing**, and pitch/prosody should be extracted from audio and then **joined by timestamps**. citeturn6view0turn6view1turn29view0turn13search3

## Output formats and a JSON schema example for kinetic typography

### Common alignment outputs you will encounter

- **Praat TextGrid**: standard for word+phone interval tiers; Praat documents the TextGrid file format. citeturn13search1turn23search29  
- **CTM** (Kaldi-style): time-marked token outputs; Kaldi docs discuss CTM and timestamp extraction; many pipelines convert CTM → TextGrid/JSON. citeturn1search13turn1search32  
- **Gentle JSON**: words with `start`, `end`, and per-word `phones` with durations. citeturn18view0turn22view1  
- **lyrics-aligner outputs**: phoneme/word onsets; supports ARPAbet 39 phone set and optional VAD thresholding. citeturn24view0  
- **SOFA outputs**: configurable export formats including TextGrid/HTK and confidence. citeturn8view2  
- **Subtitles**: SRT/VTT/LRC are good downstream targets after you have word-level timing.

### JSON schema example (word + phoneme timing + pitch contour)

Below is a **reasonable default** schema for kinetic typography. It assumes you have (or can derive) phoneme segments with start/end times, and optionally attach pitch contours per word.

```json
{
  "metadata": {
    "audio": {
      "path": "vocals.wav",
      "sample_rate_hz": 44100,
      "channels": 1,
      "source": "isolated_vocals_or_mix"
    },
    "language": "en",
    "alignment_tool": "mfa|gentle|whisperx+mfa|lyrics-aligner|sofa|webmaus",
    "created_utc": "2026-02-15T00:00:00Z"
  },
  "segments": [
    {
      "type": "phrase",
      "start_s": 12.34,
      "end_s": 16.02,
      "text": "we are the sound",
      "confidence": 0.82,
      "quality_flags": ["possible_melisma", "backing_vocals_overlap"],
      "words": [
        {
          "word": "we",
          "start_s": 12.34,
          "end_s": 12.61,
          "confidence": 0.90,
          "phones": [
            { "ph": "W", "start_s": 12.34, "end_s": 12.42, "confidence": 0.85 },
            { "ph": "IY", "start_s": 12.42, "end_s": 12.61, "confidence": 0.92 }
          ],
          "syllables": [
            { "syllable": "we", "start_s": 12.34, "end_s": 12.61, "stress": 1 }
          ],
          "pitch": {
            "f0_hz": [
              { "t_s": 12.34, "hz": 220.0, "voiced_prob": 0.98 },
              { "t_s": 12.36, "hz": 221.5, "voiced_prob": 0.97 }
            ],
            "summary": {
              "median_hz": 221.0,
              "vibrato_rate_hz_est": 5.5,
              "vibrato_extent_cents_est": 35
            }
          }
        }
      ]
    }
  ]
}
```

Notes:
- If your aligner yields **TextGrid**, you can treat each interval as a `(label, start, end)` record (Praat defines these semantics). citeturn13search1turn23search29  
- For pitch contours: **librosa pYIN** returns `f0`, `voiced_flag`, and `voiced_prob`; you can sample pitch at frame times and then slice/aggregate by each word’s `[start, end]`. citeturn13search3turn13search7

## Step-by-step usage examples for major pipelines and tools

The commands below assume you can create a working directory with:
- `audio/` containing your `vocals.flac` (isolated vocal) or `mix.flac`
- optionally `lyrics.txt` containing the known lyrics (recommended when feasible)

### Whisper for transcription (local, free) + robust audio decoding

Whisper is open-source and can transcribe **audio.flac** directly via CLI. citeturn27view3turn3search6

**Install**
```bash
python -m pip install -U openai-whisper
# ffmpeg is required by Whisper's loader
```
Whisper’s loader uses ffmpeg to downmix to mono and resample to 16 kHz. citeturn29view0turn27view2

**Transcribe (draft lyrics)**
```bash
whisper audio/vocals.flac --model medium --language English --task transcribe
```
Whisper processes audio using a sliding 30-second window internally. citeturn27view2turn27view3

**Python snippet (get transcript text)**
```python
import whisper

model = whisper.load_model("medium")
result = model.transcribe("audio/vocals.flac")
print(result["text"])
```
(If you need strict 16 kHz audio for other steps, you can use Whisper’s own `load_audio()` behavior as reference: 16 kHz mono, PCM s16le via ffmpeg.) citeturn29view0

**Improving singing transcripts (practical tip)**
Recent research suggests **source separation** and using vocal stems for segmentation can reduce WER for lyric transcription with Whisper. citeturn25search6

### WhisperX for improved word-level timestamps (then add a phoneme aligner)

WhisperX’s paper describes a 3-stage augmentation to Whisper: VAD segmentation, cut/merge to ~30s chunks, and forced alignment with an external phoneme model to yield accurate word timestamps. citeturn6view1turn26view1

**Install**
```bash
pip install whisperx
# ensure ffmpeg is available
```
WhisperX installation guidance notes ffmpeg may be needed and references Whisper setup. citeturn26view1turn27view2

**Run**
```bash
whisperx audio/vocals.wav --model large-v2 --highlight_words True
```

**Key tuning knobs**
- Larger Whisper model can improve transcription; for timestamp accuracy, WhisperX suggests model choices and alignment model selection; tested languages have default phoneme alignment models. citeturn26view1turn6view1

**Phoneme-level timing**
WhisperX is mainly positioned as word-timestamping; for **phoneme boundaries**, chain it with MFA (below) using the corrected transcript. (This “WhisperX → MFA” strategy is common in practice because MFA outputs explicit phone tiers.) citeturn23search29turn23search4

### Montreal Forced Aligner for word + phoneme boundaries (TextGrid/JSON/CSV)

**Install (conda)**
```bash
conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
```
citeturn2search26

**Align a corpus (recommended over single long file in many cases)**
MFA’s command reference:
```bash
mfa align CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY
```
citeturn27view2turn23search4

**Align a single file**
```bash
mfa align_one SOUND_FILE_PATH TEXT_FILE_PATH DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_PATH
```
MFA supports output formats `long_textgrid | short_textgrid | json | csv`. citeturn27view2

**Singing caveat**
MFA troubleshooting explicitly warns it is not intended for single files with “different style such as singing,” and suggests improvements via aligning larger speaker sets for adaptation. citeturn23search3turn6view0

**Practical “song workflow” for MFA**
1. Split your vocals into **short utterances** (e.g., by VAD or manual phrase breaks).
2. Create one `.wav` + one `.txt` per utterance (same basename).
3. Run `mfa align` on that directory.
4. Merge resulting TextGrids back into a single timeline if needed.

**Converting TextGrid → JSON**
Praat defines the TextGrid structure precisely; you can parse it and emit `(label, start, end)` items. citeturn13search1turn23search29

### Gentle forced aligner (lenient; JSON output with phones)

Gentle provides CLI, local web UI, and REST API usage instructions. citeturn10view2turn15view1

**Install & run via Docker**
```bash
docker run -P lowerquality/gentle
# then open http://localhost:8765
```
citeturn10view2

**Run via CLI**
```bash
git clone https://github.com/lowerquality/gentle.git
cd gentle
./install.sh
python3 align.py audio/vocals.mp3 lyrics.txt > gentle_out.json
```
citeturn10view2turn15view0

**Output structure (what you can rely on)**
- Words contain `start`, `duration/end`, and a list of `phones`. citeturn18view0turn22view1  
- Each phone entry includes `phone` and `duration`; reconstruct phone start/end times by cumulative summation within the word. citeturn22view1turn18view0

**Singing notes**
Gentle is “lenient,” which helps when transcripts are imperfect, but it is still speech-alignment oriented and internally resamples to 8 kHz. citeturn15view0turn22view1

### lyrics-aligner (singing-focused phoneme/word onset alignment)

This tool is explicitly for aligning lyrics to (solo/mixed) singing signals and references a peer-reviewed paper; it supports lyrics as words or phonemes (ARPAbet), and provides a `--vad-threshold` parameter to handle long instrumental sections. citeturn24view0

**Install (conda)**
```bash
git clone https://github.com/schufo/lyrics-aligner.git
cd lyrics-aligner
conda env create -f environment_cpu.yml   # or environment_gpu.yml
conda activate <env-name>
```
citeturn24view0

**Prepare**
- `AUDIO_DIR/` — audio files
- `LYRICS_DIR/` — `.txt` lyrics files with same basenames as audio citeturn24view0

**Run (phoneme onsets from word lyrics)**
```bash
python align.py AUDIO_DIR LYRICS_DIR --lyrics-format w --onsets pw --dataset-name dataset1 --vad-threshold 0
```
The docs note `--vad-threshold` was used in experiments with values between 0 and 30 depending on voice loudness. citeturn24view0

**When this helps most**
- You already have accurate lyrics text.
- Your audio resembles MUSDB-like conditions or solo singing.
- You need robust alignment under instrumental overlap. citeturn24view0turn25search2

### SOFA (Singing-Oriented Forced Aligner)

SOFA expects `.wav` + `.lab` transcript pairs in a directory structure and converts transcripts to phoneme sequences via a G2P module (dictionary-based by default), then runs alignment; it can export TextGrid/HTK/trans and output confidence scores. citeturn8view2turn9search0

**Setup**
```bash
git clone https://github.com/qiuqiao/SOFA.git
cd SOFA
conda create -n SOFA python=3.8 -y
conda activate SOFA
# install torch (+ torchaudio optionally) and other deps as per repo instructions
```
citeturn9search0

**Data layout**
```
segments/
  singer1/
    segment1.wav
    segment1.lab
```
citeturn8view2

**Inference**
```bash
python infer.py --ckpt /path/to/model.ckpt --folder segments --out_formats TextGrid,htk,trans --save_confidence
```
citeturn8view2

**Singing notes**
SOFA is specifically singing-oriented; default dictionary examples look tailored to a pinyin-like workflow, so non-Chinese use usually requires custom G2P/dictionaries/models. citeturn8view2turn9search0

### WebMAUS (free web forced alignment with phonemes)

WebMAUS Basic is a CLARIN/BAS web service that aligns an uploaded speech signal and transcript and can produce time-aligned phonemes/words/sentences compatible with Praat workflows. citeturn1search14turn1search37turn1search18

**Workflow**
1. Upload audio + transcript.
2. Download alignment output (e.g., Praat TextGrid).
3. Convert TextGrid → JSON using a TextGrid parser.

**Privacy**
Third-party upload is required; some integrations warn uploaded data is sent to a third party and note retention constraints (example: documentation mentioning deletion after ~24 hours). Ensure you have consent if using real voices. citeturn1search22

## Phoneme-level prosody extraction aligned to words: pitch & vibrato contours

To drive kinetic typography (bounce on syllables, shader motion on vibrato, etc.), you’ll usually want **pitch and voicing probability** aligned with word/phone tiers.

### Pitch extraction options (free/open-source)

- **librosa pYIN** gives `f0`, `voiced_flag`, `voiced_prob` per frame; pYIN uses Viterbi decoding and can be quite stable for monophonic vocals. citeturn13search3turn13search7  
- **torchcrepe** provides neural pitch estimates and uses Viterbi decoding by default to reduce octave errors by penalizing large jumps. citeturn13search2  
- **Praat via Parselmouth** is a standard phonetics toolchain; Parselmouth is GPL and exposes Praat analysis in Python. citeturn13search4turn13search8

### Minimal Python sketch: join word intervals with pYIN pitch samples

```python
import json
import numpy as np
import librosa

# 1) Load audio for pitch (use original SR if you like, but be consistent)
y, sr = librosa.load("vocals.wav", sr=22050, mono=True)

# 2) Pitch track (choose fmin/fmax for the singer range; defaults here are generic)
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C6"),
)
times = librosa.times_like(f0, sr=sr)

# 3) Load alignment JSON you produced (example expects words with start_s/end_s)
alignment = json.load(open("alignment_words.json"))

# 4) For each word, slice pitch samples
for w in alignment["words"]:
    t0, t1 = w["start_s"], w["end_s"]
    mask = (times >= t0) & (times <= t1) & np.isfinite(f0)
    w["pitch_summary"] = {
        "median_hz": float(np.nanmedian(f0[mask])) if mask.any() else None,
        "mean_voiced_prob": float(np.nanmean(voiced_prob[mask])) if mask.any() else None,
    }

json.dump(alignment, open("alignment_words_with_pitch.json", "w"), indent=2)
```

This relies on librosa’s documented `voiced_prob` and the ability to convert frames to times. citeturn13search3turn13search7

## Evaluation criteria, failure modes, and benchmarking scripts

### Common failure modes in singing lyric timing

- **Melisma**: one syllable stretched across many frames; word boundaries become ambiguous and phoneme/phone durations inflate. citeturn14search16turn7search20  
- **Vibrato and pitch slides**: change spectral cues; aligners can shift vowel phone boundaries.  
- **Consonant masking**: consonants are short and often buried by reverb or accompaniment; aligners misplace stops/fricatives.  
- **Backing vocals**: competing phonetic evidence; “lyrics for the lead” no longer matches the acoustics cleanly (recent datasets explicitly annotate backing vocals as separate phenomena). citeturn25search38turn25search6  
- **Over/under-segmentation**: VAD splits phrases badly, causing drift or missed words (WhisperX’s emphasis on VAD-based segmentation is specifically to mitigate drift). citeturn6view1turn25search6  
- **Domain mismatch**: speech-trained aligners (MFA, SPPAS) can degrade on singing; MFA explicitly flags this in troubleshooting. citeturn23search3turn2search39

### Metrics to compute

You typically need **two evaluation layers**: transcription accuracy and alignment accuracy.

**Text accuracy (if you are transcribing)**
- **WER / CER** using `jiwer`. citeturn3search1turn3search5  
- For scoring with standard tooling, **SCTK / sclite** is a classic reference implementation from entity["organization","NIST","us standards institute"]. citeturn3search0turn3search16

**Timing accuracy (alignment)**
- **Boundary deviation**: mean/median absolute error of start/end times vs reference (e.g., mean |Δstart|).  
- **Within-tolerance accuracy**: % of word/phone boundaries within 20/50/100 ms of reference.  
- **Overlap IoU**: interval intersection-over-union for phones/words.  
- **Segmentation F1** for boundary events (treat starts as events with tolerance windows).

MFA also supports evaluating against a reference directory (gold TextGrids) via `--reference_directory` in `mfa align`. citeturn27view2

### Benchmarks and tests you can run

**Public datasets**
- **JamendoLyrics (MultiLang)** provides word-by-word *start and end* timestamps and is used for alignment evaluation. citeturn25search0  
- **DALI** provides time-aligned lyrics at multiple granularities (words/lines/paragraphs) and is widely used for lyrics alignment research. citeturn25search1turn25search29  
- If you want song stems with lyrics, the **MUSDB18 lyrics extension** is available (lyrics transcripts for MUSDB items). citeturn25search2turn24view0

**“Small labeled tests” for your own files (recommended)**
1. Pick 20–40 seconds from your song:
   - one easy verse (clear consonants)
   - one melismatic run
   - one section with backing vocals
2. Create a **Praat TextGrid** with manual word boundaries (and optionally phone boundaries for a smaller subclip), then treat that as “gold.”

**Timing evaluation script outline**
- Parse predicted and gold TextGrids into interval lists.
- Match words by label and order (or via dynamic programming if insertions occur).
- Compute boundary errors and tolerance hit rates.

### Concrete scoring tools and commands

**WER with jiwer**
```bash
pip install jiwer
```
Then:
```python
import jiwer
wer = jiwer.wer(reference_text, hypothesis_text)
```
citeturn3search1turn3search5

**WER with sclite (SCTK)**
SCTK includes `sclite` for scoring ASR outputs against references. citeturn3search0turn3search16

(Exact CLI varies by format; SCTK docs describe `sclite`’s purpose and behavior.)

## Integration tips for Demucs/FLAC workflows and DAWs

### Demucs/isolated vocal stems

- Prefer **isolated vocal stems** when possible; published work shows source separation can improve Whisper lyric transcription and can provide better segmentation boundaries (acting like a VAD). citeturn25search6turn6view1  
- Maintain **sample-accurate alignment** between vocal stem and mix: do not trim leading silence differently across stems; add offsets explicitly in your JSON if needed.

### FLAC handling and consistent timelines

- Whisper CLI accepts FLAC directly. citeturn27view3  
- For aligners that prefer WAV-only, convert via ffmpeg; Whisper’s own loader shows a robust pattern: `-ac 1 -ar 16000` and PCM s16le. citeturn29view0

### DAW / video pipeline exports

- **TextGrid** is excellent for analysis and manual correction (Praat), then export to JSON. citeturn13search1turn23search29  
- For video editors: generate **SRT/VTT** from your word timing, then use your JSON for richer kinetic typography parameters.  
- For lip-sync/phoneme-driven animation: keep phoneme inventory consistent (e.g., ARPAbet vs IPA vs language-specific sets). Tools like lyrics-aligner restrict to CMUdict/ARPAbet 39 phonemes in one configuration. citeturn24view0

## Recommended “best free pipeline” and alternatives

### Best default for most users

**Whisper (draft lyrics) → manual text cleanup → MFA (phoneme + word boundaries) → JSON assembly (+ pitch via pYIN/torchcrepe)**

Justification:
- Whisper is easy to run locally, supports common audio formats including FLAC, and uses a robust ffmpeg-based loader and chunking approach. citeturn27view2turn27view3turn29view0  
- MFA outputs explicit phone and word tiers (TextGrid) and can export JSON/CSV; it’s one of the most practical ways to get **phoneme-level timing** without building Kaldi recipes yourself. citeturn23search29turn27view2turn6view0  
- Pitch/voicing extraction is well-supported in open-source tooling (librosa pYIN, torchcrepe), and joining by timestamps is straightforward. citeturn13search3turn13search2

Key caveat: MFA warns that singing is not its core target; segmenting and (if you have enough data) adapting/training can help. citeturn23search3turn23search21turn6view0

### High-accuracy alternatives for singing-heavy material

- **lyrics-aligner** when you have lyrics and want a singing-oriented model that can cope with mixtures and long instrumental sections (via `--vad-threshold`). citeturn24view0  
- **SOFA** when your language/resources match its singing-oriented models and you want confidence + TextGrid/HTK exports. citeturn8view2turn9search0  
- **WebMAUS** as a quick external baseline or sanity check (watch privacy/consent). citeturn1search14turn1search22  
- **Gentle** when transcripts are messy and you need a lenient aligner that still yields phone durations. citeturn22view1turn18view0

### When to use Aeneas or SPPAS

- **Aeneas**: mainly a fallback when you cannot rely on ASR/phoneme models; authors state it is not suitable for song captioning and assumes spoken audio. citeturn8view3turn14search3  
- **SPPAS**: best when you have careful transcripts and want a full annotation environment with many supported formats; it explicitly requires manual transcription (not ASR). citeturn2search39turn7search23turn7search3
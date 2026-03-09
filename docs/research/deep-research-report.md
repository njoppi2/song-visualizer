# Stem Separation Options and an Integration Plan for SongViz

## Comparison table

| Option | Local/offline | Output types | Quality notes | Speed/perf + model size | Ubuntu 22.04 + Python 3.10 install | License + constraints | How to run (CLI / Python) | Cost notes |
|---|---|---|---|---|---|---|---|---|
| entity["organization","Demucs","music source separation"] | Yes | Default: 4 stems (drums, bass, other, vocals) saved as stereo WAV at 44.1 kHz. citeturn12view0 Optional 2‑stem “karaoke mode” via `--two-stems=<source>`. citeturn12view0 Experimental 6‑source model adds `guitar` and `piano`. citeturn12view0 | Demucs’ docs explicitly call out that the experimental 6‑source `piano` stem has “a lot of bleeding and artifacts.” citeturn12view0 The Demucs paper notes Hybrid Demucs reduces bleeding across sources compared to earlier baselines. citeturn21search7 | Official docs: CPU processing time “roughly equal to 1.5× the duration of the track.” citeturn12view0 GPU memory guidance: “at least 3GB” VRAM; “about 7GB” with default args; `--segment` can reduce memory. citeturn12view0 Quantized model variants are described as “smaller download and storage” with “slightly worse” quality, but upstream does not publish exact sizes. citeturn12view0 | `python3 -m pip install -U demucs` citeturn12view0 (Separation-only requirement is Python ≥3.8.) citeturn12view0 | MIT. citeturn8view2turn6view0 | CLI: `demucs <audiofile>`; outputs land under `separated/MODEL_NAME/TRACK_NAME`. citeturn12view0 Python: `demucs.separate.main([...args...])`. citeturn12view0 | Free (open-source). |
| entity["organization","Open-Unmix","pytorch music separation"] | Yes | 4 stems for pop music: vocals, drums, bass, other. citeturn5view2turn9view1 CLI “umx” separation produces multiple output files; optional `--residual` adds an extra stem for “everything not in the targets.” citeturn7view3 | Repo reports median SDR breakdowns for multiple pretrained variants (UMX/UMXHQ/UMXL) by source (e.g., UMXL vocals 7.21, drums 7.15, bass 6.02, other 4.89 in the listed table). citeturn7view3 | Model weights are published on Zenodo: four `.pth` files at 35.6 MB each (+ small logs/metadata), total listed “Files (143.6 MB).” citeturn9view0 | `pip install openunmix` citeturn7view1 Optional: install `stempeg` to “increase the number of supported input and output file formats.” citeturn7view1 | Repo is MIT; pretrained weights on Zenodo are also MIT. citeturn5view2turn9view0 | CLI: `umx input_file.wav` (README notes “wav, flac, ogg – but not mp3”). citeturn7view3 Python: `separator = openunmix.umxl(...)` or `torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device=device)` (example). citeturn7view3turn9view1 | Free (open-source). |
| entity["organization","Spleeter","deezer stem separation"] | Yes | 2 stems (vocals/accompaniment), 4 stems (vocals/drums/bass/other), 5 stems (adds piano). citeturn32view0 | The project positions its 2‑stem and 4‑stem models as “high performances” on MUSDB. citeturn32view0 Demucs’ own comparison table lists Spleeter “Overall SDR 5.9” (in that specific summary). citeturn12view0 | Claims GPU speed: “100× faster than real-time” for 4‑stem separation on GPU. citeturn32view0 (Upstream docs in-scope here don’t publish model archive sizes.) citeturn32view0turn7view2 | Quick-start commands: `conda install -c conda-forge ffmpeg libsndfile` then `pip install spleeter`. citeturn7view2 PyPI metadata: requires Python ≥3.8,<3.12 (includes 3.10). citeturn32view2 | MIT. citeturn8view0turn32view2 README includes a reminder to obtain authorization for copyrighted material. citeturn8view1 | CLI: `spleeter separate -p spleeter:2stems -o output audio_example.mp3` produces `vocals.wav` + `accompaniment.wav`. citeturn7view2 Python usage is supported (“Python library”). citeturn32view0 | Free (open-source). |
| entity["organization","Audio Separator","python uvr model runner"] | Yes | “Various stems” depending on model; package explicitly supports ONNX and PyTorch model formats and multiple UVR model families (MDX‑Net/VR Arch/Demucs/MDXC, etc.). citeturn23view0 Also supports listing/filtering models and seeing “Output Stems (SDR)” for models where available. citeturn23view0 | Quality is model-dependent; the package’s own `--list_models` output example shows per-stem SDR numbers for some models (example table rows shown for Demucs bags). citeturn23view0 | GPU install guidance includes supported CUDA versions (11.8 and 12.2) and notes that CUDA library mismatches can occur with ONNX Runtime. citeturn23view0 Example UVR ONNX model files are commonly tens of MB (e.g., 66.8 MB shown for a UVR MDX-Net ONNX model file in a public model pack). citeturn21search28turn21search2 | PyPI: `pip install audio-separator` (latest release Jan 24, 2026; requires Python ≥3.10). citeturn23view0 CPU: `pip install "audio-separator[cpu]"`. GPU: `pip install "audio-separator[gpu]"`. citeturn23view0 If using pip, install ffmpeg via `apt-get update; apt-get install -y ffmpeg`. citeturn23view0 | MIT (PyPI metadata). citeturn23view0 | CLI example: `audio-separator /path/to/input.wav --model_filename <model>`; `audio-separator --list_models`; model caching defaults to `/tmp/audio-separator-models/`. citeturn23view0 Python API section is included in the docs (as a dependency). citeturn23view0 | Free (open-source). |
| entity["organization","Ultimate Vocal Remover","uvr5 source separation"] | Mostly local, but model-dependent | UVR is a GUI app that bundles “state-of-the-art source separation models” and notes its core developers trained the bundled models “except for the Demucs v3 and v4 4-stem models.” citeturn5view3turn21search24 (UVR’s strength is breadth of model choices; output stems depend on model.) citeturn5view3turn21search24 | UVR is frequently used as a “model zoo” front-end; in Demucs’ docs, UVR is referenced as a self-contained GUI supporting Demucs. citeturn12view0 | UVR itself is GUI-first; for headless usage on Linux you’d typically use a CLI wrapper (for example, Audio Separator above) rather than the GUI bundle. citeturn23view0turn12view0 | UVR’s README emphasizes bundled installers; Linux-first steps are not documented in the excerpted sections here. citeturn5view3 | Repo-based; licensing depends on UVR plus model weights. (UVR repo exists on GitHub; consult repo license + model pack license for commercial redistribution decisions.) citeturn5view3turn21search24 | GUI application; also includes scripts like `separate.py` in repo tree (but GUI is primary interface). citeturn5view3 | Free (open-source), but operationally heavier than CLI-first libraries. citeturn5view3turn23view0 |
| entity["organization","LALAL.AI","stem separation service"] | No (cloud upload) | One “separation type” at a time yields “two stems per file” (e.g., vocal+instrumental, drums+drumless, bass+bassless). citeturn29view0 | Independent roundup testing noted strong quality for common stems (vocals/drums/bass) and weaker performance on some “extended” instruments, depending on song. citeturn27news24 | Pricing page explains minutes deducted as: `file length × number of stem separation types selected` and gives a worked example (5 min × 3 types = 15 min). citeturn29view0 Plan pricing shown: Starter free; Lite $7.5/month; Pro $15/month; upload limits up to 2GB/file on paid tiers. citeturn29view0 | Not installable locally; integration would be via their service/API. Pricing page references an “activation key” usable in desktop app and “our API.” citeturn29view0 | Proprietary service; you accept refund/privacy/ToS terms at purchase/upload. citeturn29view0 | Web/desktop workflow; API exists per pricing FAQ (activation key enables API usage). citeturn29view0 | Subscription minutes model; monthly tiers shown on pricing page. citeturn29view0 |
| entity["organization","Music AI","moises api platform"] | No (cloud API) | API offers multiple stem products with per‑minute pricing: e.g., “Cinematic stems (Dialogue, Music, Effects)” $0.05/min; “Clean up stems (Vocals, Bass, Drums, Guitars, Keys)” $0.07/min; “Drum stems (Kick, Snare, Toms, Hi-hat, Cymbals)” $0.15/min; plus individual “Musical stems” (Vocals/Drums/Bass/etc.) at $0.07/min. citeturn26view2 | Vendor-provided; quality varies by module. (No independent benchmark in the sources above specific to this API’s modules.) citeturn26view2 | Scales per audio minute (pay‑as‑you‑go). citeturn26view2 | Integrate over HTTP: API reference shows authentication endpoints including “Get temporary token,” “Refresh token,” and “Get token status.” citeturn2search16 | Proprietary API; treat as third‑party processing (upload). citeturn2search16turn26view2 | Use their REST API (auth flow in docs). citeturn2search16 | Pay‑as‑you‑go price list is public (per‑minute, per module). citeturn26view2 |
| entity["organization","AudioShake","stem separation company"] | No for web/API; Yes for SDK (commercial, negotiated) | Offers “instrument stem separation” via developer docs; SDK overview describes separating into multiple stems (example: vocals, drums, bass, other) and supports CPU/GPU. citeturn4search1turn4search4 | Vendor positions results as “high-quality” and “performance-quality stems” (marketing claim). citeturn4search10turn4search4 | Indie pricing page states “$5.00 / stem” (and indicates plan-style packaging). citeturn4search3 | API/SDK; SDK docs say to obtain credentials you contact them (Client ID/Secret). citeturn4search4 | Proprietary; API/SDK governed by vendor terms and credentialing. citeturn4search4turn4search22 | Developer docs: server-to-server stem separation API and a “Tasks API” concept for running one or more models on the same media. citeturn4search1turn4search22 | $5 per stem (indie pricing page). citeturn4search3 Broader enterprise pricing is typically “contact sales” in many vendor models (SDK requires contacting for access). citeturn4search4 |
| entity["organization","SpectraLayers Pro 12","steinberg spectral editor"] | No (not Linux; desktop app is Windows/macOS) | “Unmix Song” can produce Vocals, Drums, Bass, Guitar, Piano, Sax & Brass, Other; quality modes Fast/Balanced/High. citeturn31view0 | Independent roundup placed this tool near the top and noted many options but imperfect recognition (example: mislabeling piano content). citeturn27news24 | Not a Python library; performance is workstation-dependent. Steinberg forum guidance recommends dedicated GPU (8GB VRAM) for faster unmixing in some cases. citeturn4search12 | Not installable on Ubuntu: official system requirements list Windows and macOS only. citeturn4search2 | Proprietary commercial software. citeturn27search0turn27search1 | GUI-first. The manual includes command-line options section in ToC, but stem separation is primarily an in-app workflow. citeturn31view0 | Suggested retail price for Pro 12 is stated as 349€ / $349.99 in Steinberg’s press release, and the product page shows $349.99. citeturn27search0turn27search1 |

## Recommended path with concrete install and run steps

### Primary recommendation

Primary: entity["organization","Demucs","music source separation"], using its default 4‑stem model for an offline-first workflow with an explicit MIT license and documented CLI/Python entrypoints. citeturn12view0turn8view2 Demucs also documents practical operational expectations relevant to SongViz (fixed 44.1 kHz outputs, default 4 stems, and CPU runtime guidance). citeturn12view0

Fallback: entity["organization","Audio Separator","python uvr model runner"], because it is Python ≥3.10, CLI-first, can write outputs to a specified directory, and can switch among multiple model families (including Demucs bags and other UVR models) without you hard-coding a single architecture into SongViz. citeturn23view0

### Demucs on Ubuntu 22.04 + Python 3.10

Install (in your SongViz venv):

```bash
python3 -m pip install -U demucs
```

The above command is the Demucs project’s documented “for musicians” install. citeturn12view0

Run a separation:

```bash
demucs songs/foo.flac
```

Demucs documents that it writes separated stems under `separated/MODEL_NAME/TRACK_NAME`, producing four stereo WAVs at 44.1 kHz: `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`. citeturn12view0

Useful flags you can safely rely on from the official README:

- 2‑stem mode (still runs a full separation and then mixes stems):  
  ```bash
  demucs --two-stems=vocals songs/foo.flac
  ```  
  citeturn12view0
- Pick a different pretrained model name (example shown):  
  ```bash
  demucs -n mdx_q songs/foo.flac
  ```  
  Demucs describes `mdx_q` / `mdx_extra_q` as quantized variants with smaller storage and potentially slightly worse quality. citeturn12view0
- Avoid the “random shifts” trick if you want predictable runtimes; `--shifts` explicitly multiplies runtime and uses “random shifts.” citeturn12view0

Performance expectations you can use as a baseline:

- CPU time is documented as “roughly equal to 1.5× the duration of the track.” citeturn12view0
- GPU VRAM guidance is documented (≥3GB; ~7GB with default args), and Demucs suggests `--segment` to reduce memory. citeturn12view0

Python entrypoint (documented):

```python
import demucs.separate
demucs.separate.main(["--two-stems", "vocals", "songs/foo.flac"])
```

Demucs documents calling `demucs.separate.main([...])` as its simple Python API surface. citeturn12view0

### Audio Separator on Ubuntu 22.04 + Python 3.10

Install (CPU-only) inside your SongViz venv:

```bash
pip install "audio-separator[cpu]"
```

This is the package’s documented pip install for CPU-only usage. citeturn23view0

If you want NVIDIA GPU acceleration, the package documents:

```bash
pip install "audio-separator[gpu]"
```

…and provides additional troubleshooting steps if CUDA / ONNX Runtime versions mismatch. citeturn23view0

If you installed via pip, ensure ffmpeg is installed (package’s docs):

```bash
apt-get update
apt-get install -y ffmpeg
```

citeturn23view0

Run (example pattern from the docs):

```bash
audio-separator songs/foo.flac --model_filename htdemucs.yaml
```

The package documents `--model_filename`, automatic model download/caching, and `--list_models` / `--list_filter` for discovery. citeturn23view0

List available models / discover which ones output drums (documented examples):

```bash
audio-separator --list_models
audio-separator -l --list_filter=drums
```

citeturn23view0

Model caching location (default) is documented as `/tmp/audio-separator-models/`. citeturn23view0

## SongViz integration blueprint

### Goals and non-goals aligned to your repo constraints

Your constraints imply three concrete requirements for the integration layer:

- Input audio stays under `songs/` and remains uncommitted (gitignored); do not introduce any workflow that suggests committing copyrighted audio into git.
- Outputs must land under `outputs/<song_name>/...` and remain gitignored.
- Stem separation must be optional and cacheable, since it’s substantially more expensive than your current single-mix `librosa` analysis.

These are project design constraints (not external facts), so the rest of this section proposes a reproducible implementation shape that stays inside them.

### Proposed CLI shape

Add a new subcommand:

```bash
songviz stems songs/foo.flac
```

Recommended flags (design proposal):

- `--backend demucs|audio-separator` (default: `demucs`)
- `--model <name>` (backend-specific; e.g., Demucs `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx`, etc. are named in Demucs docs) citeturn12view0
- `--device cpu|cuda` (for Demucs you can implement by passing `-d cpu` when needed; Demucs documents using `-d cpu` as a fallback when GPU memory is insufficient) citeturn12view0
- `--force` to ignore cache and re-run
- `--keep-intermediate` for debugging (store backend logs and a “run manifest”)

### Output layout

For a song at `songs/foo.flac`, write:

- `outputs/foo/stems/`
  - `drums.wav`
  - `bass.wav`
  - `vocals.wav`
  - `other.wav`
  - (optionally `guitar.wav`, `piano.wav` if the backend/model produces them; Demucs documents a 6-source model adding guitar/piano) citeturn12view0
- `outputs/foo/stems/stems.json` (metadata + cache key)
- `outputs/foo/stems/_logs/<backend>.log` (stdout/stderr capture)

Note: Demucs’ default on-disk convention is `separated/MODEL_NAME/TRACK_NAME` with those four WAV stems at 44.1 kHz. citeturn12view0 SongViz should treat that as an intermediate and then copy/link into the stable `outputs/foo/stems/` target.

### Metadata schema for stems.json

A concrete JSON schema you can implement (design proposal):

```json
{
  "schema_version": 1,
  "input": {
    "path": "songs/foo.flac",
    "sha256": "…",
    "bytes": 12345678
  },
  "backend": {
    "name": "demucs",
    "backend_version": "demucs==X.Y.Z",
    "model": "htdemucs",
    "args": ["--two-stems", "vocals"],
    "device": "cpu"
  },
  "audio": {
    "sample_rate_hz": 44100,
    "channels": 2,
    "duration_seconds": 245.12
  },
  "stems": [
    {"name": "drums", "path": "outputs/foo/stems/drums.wav", "sha256": "…", "bytes": 1234},
    {"name": "bass",  "path": "outputs/foo/stems/bass.wav",  "sha256": "…", "bytes": 1234},
    {"name": "vocals","path": "outputs/foo/stems/vocals.wav","sha256": "…", "bytes": 1234},
    {"name": "other", "path": "outputs/foo/stems/other.wav", "sha256": "…", "bytes": 1234}
  ],
  "created_at": "2026-02-14T…Z"
}
```

Fields worth including for reproducibility:

- `backend_version` (e.g., `demucs` installed via pip) because Demucs is pip-installable and versions change behavior. citeturn12view0
- `model` because Demucs exposes multiple bags/models and warns that `htdemucs_ft` is ~4× slower than the default and that the 6‑source piano is artifact-prone. citeturn12view0
- recorded `sample_rate_hz` because Demucs outputs 44.1 kHz WAVs by default. citeturn12view0

### Cache key and “don’t re-separate” logic

Implement “content-addressed” caching:

- Cache key = SHA‑256 of the *input audio bytes* + backend name + backend version + model name + normalized args list.
- Before running separation:
  1. Check if `outputs/<song>/stems/stems.json` exists.
  2. If yes, validate:
     - `input.sha256` matches current input file hash
     - `backend` fields match requested backend+model+args
     - all stem files listed exist and their hashes match
  3. If all checks pass, skip separation.

This avoids re-separating when only rendering parameters change (e.g., visuals), and it avoids false cache hits when the input file is replaced.

### Feeding stems into analysis without exploding runtime

The expensive part is separation; once stems exist, you can keep analysis scalable via two tactics (design proposal):

- Compute per-stem features only for stems that are actually mapped to visual layers (e.g., compute onset/RMS for drums+bass only, not for “other” if unused).
- Downsample before analysis (e.g., resample stems to a lower rate for envelope/onset extraction) and cache the derived features alongside stems:
  - `outputs/<song>/analysis/features_stems_v1.npz` (or JSON + `.npy` arrays)
  - Include the same cache key strategy and store feature parameters (hop length, window, etc.)

This keeps a multi-stem workflow from multiplying your per-song analysis time by N stems on every run.

### Backend abstraction

A clean separation interface (design proposal):

- `songviz/stems/backends/base.py`
  - `class StemBackend: separate(input_path, out_dir, *, model, device, **kwargs) -> StemResult`
- `songviz/stems/backends/demucs_backend.py`
- `songviz/stems/backends/audio_separator_backend.py`

Implement the Demucs backend by invoking either:
- Demucs CLI (`demucs …`) with subprocess, and then normalizing/moving the outputs from Demucs’ default `separated/...` convention into `outputs/<song>/stems/`. citeturn12view0
- or Demucs’ Python entrypoint `demucs.separate.main([...])` (documented). citeturn12view0

Implement the Audio Separator backend using its CLI plus `--model_file_dir` and `--output_dir` (both documented). citeturn22view0turn23view0

## Cost-aware and privacy-aware options

### Cloud processing is optional, and should be opt-in

Your constraints prefer offline processing and explicitly say not to propose uploading purchased music unless clearly labeled optional and privacy implications are explained.

Accordingly, implement cloud backends only as a separate backend with an explicit opt-in flag (design proposal):

- `songviz stems --backend lalal --i-understand-this-uploads-audio`
- `songviz stems --backend musicai --i-understand-this-uploads-audio`

Also implement a “dry run” mode that prints which files would be uploaded and which endpoints would be called, without actually uploading anything.

### Pricing and scaling snapshots

entity["organization","LALAL.AI","stem separation service"]

- Plans shown on their pricing page: Starter free; Lite $7.5 billed monthly; Pro $15 billed monthly. citeturn29view0
- Billing is minute-based, and minutes deducted scale with audio duration and the number of separation types selected: `file length × number of stem separation types`. citeturn29view0
- Each separation type yields 2 stems per file (e.g., vocals+instrumental, drums+drumless). citeturn29view0

entity["organization","Music AI","moises api platform"]

- Public pay-as-you-go pricing lists stem-separation modules per minute, including (examples):  
  - Cinematic stems (Dialogue/Music/Effects) $0.05/min citeturn26view2  
  - Clean up stems (Vocals/Bass/Drums/Guitars/Keys) $0.07/min citeturn26view2  
  - Drum stems (Kick/Snare/Toms/Hi-hat/Cymbals) $0.15/min citeturn26view2  
  - Musical stems for single instruments (Vocals/Drums/Bass/etc.) $0.07/min citeturn26view2
- API docs show a token-based auth flow (temporary token, refresh, status). citeturn2search16

entity["organization","AudioShake","stem separation company"]

- Indie pricing page snippet indicates $5 per stem. citeturn4search3
- Developer docs describe a stem separation API and a Tasks API for running one or more models on the same media. citeturn4search1turn4search22
- SDK docs state you must contact them to access and obtain credentials, and they describe CPU/GPU support. citeturn4search4

### Safe API integration patterns

For any cloud backend (design proposal):

- Never upload by default; require explicit opt-in each run (or a config file setting that is knowingly enabled).
- Keep API keys in environment variables (e.g., `SONGVIZ_MUSICAI_TOKEN`) and never write them into `stems.json`.
- Avoid storing user audio on third-party servers unless explicitly chosen; by default, delete temporary upload artifacts after job completion and only keep local stems under `outputs/<song>/stems/`.
- For enterprise/vendor SDKs, isolate them behind an optional extra dependency (e.g., `pip install songviz[cloud]`) so offline users don’t pull cloud SDKs.

These are implementation recommendations rather than external facts.

## Testing and reproducibility strategy without copyrighted audio

### Unit tests (fast, deterministic)

Use `pytest` with synthetic audio you generate at test time (design proposal):

- Generate a WAV containing:
  - a click track (impulses) to simulate “drums”
  - a low sine wave for “bass”
  - a mid sine wave for “vocals” (as a proxy)
- Write it into a temporary directory via a standard audio writer (e.g., `soundfile`), then run `songviz stems` with a **mock backend**.

Key unit tests:

- `test_stems_command_writes_expected_paths`: ensure `outputs/<song>/stems/` and `stems.json` are created.
- `test_cache_hit_skips_backend`: run twice; second run should not call backend if hashes/settings match.
- `test_cache_miss_on_input_change`: modify the input file; ensure backend re-runs.
- `test_metadata_schema_fields_present`: validate keys and that listed stem files exist.

Mock backend approach:

- Implement a `DummyBackend` that writes fixed stems (e.g., copies input into each stem filename) and returns a predictable `StemResult`.
- This verifies SongViz’s IO, hashing, caching, and metadata without depending on large ML models.

### Integration tests (slow, opt-in)

Add `@pytest.mark.slow` tests that run a real backend only when the environment is configured:

- Demucs integration test:
  - Skip unless `demucs` is installed and `ffmpeg` is present.
  - Use a very short synthetic WAV (≤2 seconds) to minimize runtime.
  - Assert:
    - Demucs output is detected and normalized into SongViz’s `outputs/<song>/stems/` layout.
    - `stems.json` records backend name/model and sample rate consistent with Demucs doc (44.1 kHz). citeturn12view0

This keeps CI clean while still giving you an end-to-end test you can run locally.

### Reproducibility knobs you can expose

- Default to Demucs settings that do not explicitly add randomized ensembling; Demucs documents `--shifts` does “multiple predictions with random shifts” and makes prediction times slower. citeturn12view0
- Record backend/model/args in `stems.json` and treat any change in those fields as a cache miss.

These steps don’t guarantee bit-identical outputs across hardware, but they make “what produced these stems” auditable and keep reruns stable within a project.
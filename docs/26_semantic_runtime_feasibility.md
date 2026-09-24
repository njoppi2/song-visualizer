# Music Flamingo local runtime feasibility — Team 2

Audit date: 2026-09-14. **Status: complete — awaiting Team 1 integration review.**
Scope was deliberately read-only: no model download,
environment installation, inference, audio upload, cache change, GPU-driver change,
or paid/remote provisioning was attempted. This is an execution-feasibility result,
not a test of Music Flamingo's musical claims.

## Decision

**Do not start a local Music Flamingo probe on the current machine. No demonstrated
execution route exists under its current constraints.** GPU unavailability is a
measured immediate blocker, but repairing it alone is not a feasibility route:
the identified RTX 4050 Laptop GPU has a published standard 6 GB memory
configuration, already below the 15.40 GiB unquantized BF16 weight file. The
installed Transformers runtime is also incompatible and currently available system
memory is insufficient for the raw weights alone. A different hardware,
quantization, or offload plan would be a separate bounded feasibility question;
it is not assumed or authorized here. No paid or external-audio route is proposed,
and Hugging Face currently lists no Inference Provider for this model.

The checkpoint itself is publicly visible and not gated, so credentials are not
the present blocker. Its terms are nevertheless non-commercial research only;
any later use must remain within the [NVIDIA OneWay Noncommercial License](https://huggingface.co/nvidia/music-flamingo-2601-hf#license--terms-of-use).

## Checked model identity and documented constraints

The requested starting point, [NVIDIA's Music Flamingo model card](https://huggingface.co/nvidia/music-flamingo-2601-hf), currently resolves to revision
`6b5be086d52f65a1e204cb0faf70bf54e2741ecd` (Hub API observation below; model
last modified 2026-04-09). It reports an 8B-parameter BF16 model, built from an
AF-Whisper audio encoder, an MLP adaptor and Qwen2.5-7B; its model-file listing
contains one `model.safetensors` file. The model card's inference section names
PyTorch/Hugging Face Transformers, Linux and NVIDIA A100/H100 as supported/test
hardware (test hardware: A100 80 GB). It accepts WAV, MP3 or FLAC plus text,
processes 16 kHz mono audio in 30-second windows, and caps one sample at 40
windows / 20 minutes. The song is only 221.17 seconds, so its duration is within
that documented cap. [The current upstream Transformers model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/musicflamingo)
also describes 30-second / 16 kHz windowing and the 20-minute hard maximum.

At the resolved revision, the raw Hub file endpoint reported:

| Item | Observed value | Meaning |
| --- | --- | --- |
| `model.safetensors` | 16,534,531,504 bytes = 16.53 GB = 15.40 GiB | Actual repository weight-file size, not an estimate. |
| Model card | 8B parameters, BF16 | Publisher metadata; this implies a roughly 16 GB raw-weight order of magnitude. |
| Config producer | `transformers_version: 5.6.0.dev0` | Exact config metadata, not a claim that a released version is sufficient. |
| Text context | 32,768 positions; 24,000-token max text | Config/card constraint. |
| Audio context | 30 s windows, maximum 40 / 1,200 s | Publisher documentation. |

The publisher's card presently says to upgrade `transformers` and `accelerate`
and describes its then-required fork while the upstream documentation exposes the
`MusicFlamingoForConditionalGeneration` API. That is a moving dependency surface:
a future authorized setup must pin a revision known to contain this exact class,
rather than assuming the currently installed MuQ dependency is compatible.

## Read-only local observations

Commands below were run from the repository. They neither downloaded a model nor
initialized inference.

```bash
curl --fail --silent --show-error \
  https://huggingface.co/api/models/nvidia/music-flamingo-2601-hf

curl --fail --silent --show-error --location --range 0-0 -D - -o /dev/null \
  https://huggingface.co/nvidia/music-flamingo-2601-hf/resolve/6b5be086d52f65a1e204cb0faf70bf54e2741ecd/model.safetensors

.songviz/representation-venv/bin/python -c 'from transformers import MusicFlamingoForConditionalGeneration'
nvidia-smi
```

| Area | Result |
| --- | --- |
| CPU / memory | Intel Core i7-13620H; 16 logical CPUs / 10 cores; 31 GiB RAM total, about 11 GiB available. Swap is 2 GiB total, with about 1.8 GiB already used. |
| Storage | 40 GiB free of 277 GiB in the workspace filesystem. The 15.40 GiB weights could consume a large fraction but are not downloaded by this audit. |
| GPU identity / capacity | `/proc/driver/nvidia/gpus/0000:01:00.0/information` identifies an **NVIDIA GeForce RTX 4050 Laptop GPU**. [NVIDIA's current laptop comparison table](https://www.nvidia.com/en-au/geforce/laptops/compare/) lists a standard **6 GB GDDR6** configuration for that GPU. This is a published capacity, not a measurement of free VRAM on this machine. |
| GPU runtime | An NVIDIA PCI device exists, but `nvidia-smi` failed: `Failed to initialize NVML: Driver/library version mismatch`. The kernel NVRM reports 580.173.02 while the NVML user library reports 580.178. |
| PyTorch CUDA | Both project environments have torch 2.10.0+cu128. `torch.cuda.is_available()` is `False`, device count is 0, and CUDA emits error 804: `forward compatibility was attempted on non supported HW`; it also reports that NVML cannot initialize. |
| Main analysis venv | Python 3.10.12; no `transformers`, `accelerate`, `huggingface_hub` or `safetensors`. |
| Existing representation venv | Python 3.10.12; `transformers` 4.57.6, `huggingface_hub` 0.36.2 and `safetensors` 0.5.3; `accelerate` absent. Importing `MusicFlamingoForConditionalGeneration` raises `ImportError`, so this venv cannot load the checkpoint API. |
| Existing related cache | Only the pinned 1.3 GB MuQ model is present under `.songviz/representation-models/`; no Music Flamingo weights were found. |

## Memory assessment

These are transparent estimates, not measured Music Flamingo execution:

- The 16,534,531,504-byte safetensors file alone is **15.40 GiB**. That exceeds
  the roughly **11 GiB currently available** system RAM, before Python, model
  objects, audio features, activations, generation state or operating-system
  pressure. CPU fallback is therefore not a responsible smoke-test route here.
- A simple two-copy loading envelope is **30.80 GiB** (2 × raw file) before
  allocator overhead and runtime buffers. Some loaders may stream/shard and use
  less peak RAM; others may transiently need a second representation. This is
  explicitly an upper-risk envelope, not a measured requirement.
- The model card does not give a minimum VRAM number. Its A100 80 GB test hardware
  and A100/H100 support listing are stronger evidence than an invented exact
  minimum. The RTX 4050 Laptop GPU's published standard 6 GB capacity is below
  raw BF16 weights before activations, audio features or runtime buffers. Because
  PyTorch sees zero usable devices, its free VRAM and actual on-device peak cannot
  be measured safely in this audit. Driver restoration would only make that
  capacity observable; it would not make the stated no-offload, unquantized,
  all-on-GPU proposal viable.

## Conditional smoke-test shape — not a current recovery plan

This describes the smallest systems check only if a separately evidenced feasible
compute plan is selected. It must not run merely because the NVIDIA driver/CUDA
issue is resolved. It additionally requires demonstrated adequate memory for the
chosen precision/offload plan, a usable GPU visible to PyTorch, and human
authorization for the otherwise out-of-scope environment creation and model
download.

1. Create a **new isolated** Music Flamingo environment; pin the resolved Hub
   revision above and a Transformers revision that imports
   `MusicFlamingoForConditionalGeneration`. Keep MuQ's environment unchanged.
2. Download the checkpoint once into a new, provenance-recorded cache. Record
   file size/hash, package versions, CUDA visibility, GPU name/free memory and
   processor configuration before audio is processed.
3. Run exactly one original-mix **20-second** local excerpt (for example 119–132s
   plus neutral context, still under one 30-second window) with one neutral,
   title-free prompt: “Describe only audible instruments, vocals and changes in
   this excerpt. State uncertainty when unsure.” Use deterministic decoding and
   a small fixed output budget (for example 128 new tokens). Save only the local
   source interval, prompt, parameters, raw response, elapsed time and measured
   peak memory. Do not include annotations, role labels or expected answers in
   the prompt.

Success requires: checkpoint and dependency provenance captured; one local audio
input completes without CPU offload or remote service; measured peak memory stays
below available VRAM; the response and interval are saved; and no claim is
treated as validated until Team 1/3's later semantic design evaluates it against
audio and existing notes. Stop immediately if the class/import still fails, CUDA
is unavailable, model loading exceeds available memory, any new credential/paid
service/external audio transfer is requested, or the run tries to fall back to
CPU. Those are execution stops, not evidence against semantic utility.

## Handoff to Team 1

Local execution is presently infeasible for measured systems reasons. Team 1 can
use this report when reconciling Team 3's scientific design, but should not
schedule model execution or infer a semantic-model result. A driver repair can
be pursued only as a local systems decision; it does not clear the 6 GB versus
15.40 GiB capacity gate. Any alternative compute plan needs separate scoped
evidence and explicit human authorization; neither is implied by this audit.

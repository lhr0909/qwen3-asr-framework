# Q3ASR Worklog

## 2026-03-16

### Goal

Stand up the first runnable version of the library in this repo with:

- split-model loading (`text.gguf` + `mmproj.gguf`)
- Qwen3-ASR mel preprocessing and encoder execution owned by this library
- upstream `llama.cpp` used as the text decoder runtime
- a smoke test that runs a real local model + wav file
- a small CLI for direct wav-file transcription

### References Used

- `../qwen3-asr-llamacpp/docs/streamlined-qwen3-asr-library.md`
- `../qwen3-asr-llamacpp/qwen3-asr-llama-cpp.patch`
- `../qwen3-asr-llamacpp/qwen3asr-enc.cpp`
- `../qwen3-asr-llamacpp/convert_qwen3_asr_to_gguf.py`
- `../Qwen3-ASR/qwen_asr/inference/qwen3_asr.py`
- `../Qwen3-ASR/qwen_asr/inference/utils.py`
- `../qwen3-asr.cpp/src/mel_spectrogram.*`
- `../qwen3-asr.cpp/src/audio_encoder.*`
- `../qwen3-asr.cpp/src/gguf_loader.*`

### Current Implementation Notes

- The upstream decoder checkout has now been migrated into this repo at `third_party/llama.cpp`.
- Compatibility symlinks were left behind in the sibling reference repos so older local commands still resolve, but this repo is now the canonical home for the active GGUF models, active wav fixtures, and the decoder dependency.
- The encoder path is being rewritten against the split mmproj GGUF layout instead of the older monolithic `qwen3-asr.cpp` model layout.
- The first end-to-end path is now running successfully against local weights:
  - text model: `models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf`
  - mmproj model: `models/gguf/Qwen3-ASR-1.7B-mmproj.gguf`
  - sample audio: `testdata/asr_zh.wav`
- Observed sample output with the current prompt path:
  - raw: `language Chinese<asr_text>甚至出现交易几乎停滞的情况。`
  - language: `Chinese`
  - text: `甚至出现交易几乎停滞的情况。`
- The initial decoder integration bug was KV-position handling across prompt tokens and injected audio embeddings.
- The fix was to drive an explicit monotonic `n_past` / position sequence for every token and audio embedding chunk.
- `llama.cpp` logging is now quiet by default for this library path; set `Q3ASR_LLAMA_VERBOSE=1` to restore upstream logs while debugging.
- The Python reference and the patched `llama.cpp` mtmd runtime both confirmed that `conv_chunksize` is not the mel time chunk length.
- It is only a batching knob for how many already-split conv chunks are processed at once during convolution.
- The actual Qwen3-ASR time chunk length is `n_window * 2` mel frames, so interpreting `clip.audio.conv_chunksize` as a time split degrades transcripts while still producing plausible text.
- The transcript mismatch against the sibling `llama.cpp` implementation was ultimately fixed by aligning the mel path with the mtmd reference worker-threaded FFT implementation instead of the separate Apple Accelerate shortcut.
- After that mel parity fix, `testdata/q3asr-input.wav` now matches the sibling CLI output: `You can apparently promote on Sundays on /r/apple on Reddit.`
- The local WAV loader now accepts IEEE-float WAV files (`audio_format = 3`) in addition to 16-bit PCM, so the original `Recording_20260202131426.wav` test file can be transcribed directly without an ffmpeg conversion step.
- The active Hugging Face safetensor model directories now live under `../Qwen3-ASR/models/`:
  - `../Qwen3-ASR/models/Qwen3-ASR-1.7B`
  - `../Qwen3-ASR/models/Qwen3-ASR-0.6B`
- The converted GGUF artifacts now live locally in this repo under `models/gguf/`, including the migrated legacy `qwen3-asr-0.6b-f16.gguf`.
- The local Hugging Face safetensor model directory at `../Qwen3-ASR/models/Qwen3-ASR-1.7B` has now been exercised through the sibling Python reference project in `../Qwen3-ASR` using `uv`.
- The working local setup path was:
  - `cd ../Qwen3-ASR`
  - `uv venv --python 3.12 .venv`
  - `uv pip install --python .venv/bin/python -e . torch`
- `uv sync` did not work against the upstream `pyproject.toml` as-is because the project declares `requires-python = ">=3.9"` while pinning `accelerate==1.12.0`, which only resolves for Python `>=3.10`.
- The parity check used the local safetensor weights and the original float WAV:
  - model dir: `../Qwen3-ASR/models/Qwen3-ASR-1.7B`
  - audio: `testdata/Recording_20260202131426.wav`
- Observed Python reference output:
  - raw: `language English<asr_text>You can apparently promote on Sundays on /r/apple on Reddit.`
  - language: `English`
  - text: `You can apparently promote on Sundays on /r/apple on Reddit.`
- That output matches the current `q3asr-cli` output on the same file exactly, so there is no observed degradation on this regression sample between the local GGUF path and the unconverted safetensor reference path.
- Added root-level `README.md` and `AGENTS.md` so future agents can recover the project overview, architecture, validation workflow, and local-reference constraints from the repo itself instead of depending on thread history.
- Migrated the active artifact layout so future iteration can happen from this repo directly:
  - `third_party/llama.cpp`
  - `models/gguf/`
  - `testdata/`
- Left compatibility symlinks in `../qwen3-asr-llamacpp` and `../qwen3-asr.cpp` so the old reference paths still resolve.
- After the migration, a clean `cmake -S . -B build -DQ3ASR_LLAMA_CPP_SOURCE_DIR=/Users/simon/Documents/git-repo/oss/qwen3-asr-hax/third_party/llama.cpp` configure succeeded, `cmake --build build -j` succeeded, and `ctest --test-dir build --output-on-failure` passed `2/2`.
- The Python safetensor parity check also still works from the migrated base-model location `../Qwen3-ASR/models/Qwen3-ASR-1.7B` against `testdata/Recording_20260202131426.wav`, with the same `/r/apple` transcript as the local GGUF path.

### Remaining Work

- Validate long-audio behavior carefully against the patched `llama.cpp` parity runtime.
- Decide whether the migrated `third_party/llama.cpp` checkout should stay as a plain working copy or be converted into a more explicit vendored/submodule setup once the pipeline is stable.
- Add streaming/session-layer behavior from the official Python implementation.
- Add resampling and richer audio-format support beyond 16 kHz WAV input.
- Tighten the smoke test further with a stable transcript fixture once we decide whether exact text matching is acceptable across hardware/backends.
- Revisit prompt construction to decide whether the explicit ChatML prompt should stay or be replaced with model-template rendering plus audio placeholder injection.
- Add a structured decoder/session API rather than only the current one-shot C entrypoints.
- Add a targeted regression check for Qwen3-ASR encoder chunk metadata so `conv_chunksize` is not misinterpreted again.
- Add broader parity checks against the Python and patched `llama.cpp` references for more local wav fixtures, not just the current English regression sample.
- Decide whether to patch the local `../Qwen3-ASR/pyproject.toml` metadata for `uv sync` compatibility or keep relying on the working `uv pip install -e . torch` bootstrap.
- Add a scripted Python-reference parity command in this repo so the safetensor comparison does not depend on an inline one-off shell snippet.

### Gotchas

- The local repo at `../llama.cpp` appears older than the upstream snapshot bundled near `qwen3-asr-llamacpp`, so the latter is the safer base for decoder integration.
- The Qwen3-ASR text model chat template is present in GGUF metadata, but the initial library path is still using an explicit prompt builder so the audio placeholder injection is fully under library control.
- The split mmproj format stores the exact encoder tensor shapes, which is useful enough to load from GGUF metadata directly instead of reconstructing every tensor shape manually.
- `clip.audio.conv_chunksize` in the GGUF metadata looks like a conv time-chunk parameter, but for Qwen3-ASR it is really the conv batching size from the Python implementation; the encoder time chunk is still `n_window * 2`.
- The prompt format from the Python reference and the GGUF chat template were already effectively correct for this library path; the observed English-quality gap was not caused by the extra `s` hypothesis once an audio-only single-turn parity check was run.
- The first working smoke test currently proves "real transcription happened" by requiring a parsed non-empty text result and the detected language `Chinese`; it does not yet lock down an exact transcript.
- There is now an optional `q3asr-english-regression` test that is registered when `testdata/q3asr-input.wav` exists locally; it checks for detected language `English` and the `/r/apple` substring to catch this mel-front-end regression.
- The current wav loader path is intentionally narrow: mono/stereo WAV parsing is supported, but the transcription entrypoint rejects non-16 kHz input instead of resampling it.
- Float WAV support is now in place for `pcm_f32le` / format `3`, but the entrypoint still intentionally rejects non-16 kHz input instead of resampling it.
- The previous reconfigure issue seen before artifact migration is no longer reproducing with the current local `third_party/llama.cpp` layout; plain reconfigure/build works again in this workspace.
- The upstream `../Qwen3-ASR/pyproject.toml` currently claims Python `>=3.9`, but one pinned dependency (`accelerate==1.12.0`) forces Python `>=3.10`, so plain `uv sync` fails until that metadata mismatch is addressed.
- Importing the full top-level `qwen_asr` package was not necessary for the parity check and appeared to pull in heavier optional tooling; importing only `qwen_asr.core.transformers_backend` plus the inference utilities was enough to register the custom config/model/processor classes and run the local safetensor model.

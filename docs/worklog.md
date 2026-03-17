# Q3ASR Worklog

## 2026-03-17

### Goal

Bring the forced aligner into this repo as a first-class local subsystem with Python-reference parity for the short-sample path.

### What Changed

- Added a separate forced-aligner runtime in `src/forced_aligner.cpp` and `src/forced_aligner.h`.
- Added a separate C API surface in `include/q3asr.h` and `src/q3asr.cpp`:
  - `q3asr_aligner_context_create()`
  - `q3asr_align_wav_file()`
  - `q3asr_align_pcm_f32()`
  - `q3asr_alignment_result_clear()`
- Kept the aligner separate from the transcription context on purpose.
- Loaded the aligner GGUF through the same direct GGUF/ggml metadata style used elsewhere in this repo instead of reusing the older monolithic loader flow.
- Reused the local Qwen3-ASR mel front end for alignment.
- Ported the classifier-head decoder path and Python-style timestamp repair from the sibling C++ reference.
- Implemented Python-biased normalized-unit tokenization behavior for the common paths:
  - space-delimited languages strip punctuation and keep alnum/apostrophe units
  - Chinese mixed text splits CJK characters into separate units
  - Korean can use an optional dictionary path
  - Japanese currently uses a local fallback heuristic, not a `nagisa` port
- Converted the raw local HF forced-aligner model into a canonical local GGUF:
  - input: `../Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B`
  - output: `models/gguf/qwen3-forced-aligner-0.6b-f16.gguf`
- Extended `q3asr-cli` with direct aligner mode:
  - `--aligner-model`
  - `--align-text`
  - `--korean-dict`
- Added real-model alignment regressions in `tests/aligner_test.cpp` and wired them in `CMakeLists.txt`.
- The default CLI path is now quiet in aligner mode as well; the noisy ggml/Metal logs are muted through the same global logging hook already used for the decoder path.

### Validation

- Converted the local aligner model with:
  - `GGUF_PY_PATH=/Users/simon/Documents/git-repo/oss/qwen3-asr-hax/third_party/llama.cpp/gguf-py ../Qwen3-ASR/.venv/bin/python ../qwen3-asr.cpp/scripts/convert_hf_to_gguf.py --input ../Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B --output models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --type f16`
- Rebuilt successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Full test suite passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `5/5` passing
- Manual English aligner CLI spot check:
  - `./build/q3asr-cli --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/q3asr-input.wav --align-text "You can apparently promote on Sundays on /r/apple on Reddit." --language English`
  - observed units:
    - `You`
    - `can`
    - `apparently`
    - `promote`
    - `on`
    - `Sundays`
    - `on`
    - `rapple`
    - `on`
    - `Reddit`
- Manual Chinese aligner CLI spot check:
  - `./build/q3asr-cli --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/asr_zh.wav --align-text "甚至出现交易几乎停滞的情况。" --language Chinese`
  - observed units:
    - `甚`
    - `至`
    - `出`
    - `现`
    - `交`
    - `易`
    - `几`
    - `乎`
    - `停`
    - `滞`
    - `的`
    - `情`
    - `况`

### Remaining Work

- Use the aligner timestamps to build a real long-audio overlap-merging policy instead of plain text concatenation.
- Decide how raw decoder text spans should map back onto normalized aligner units once punctuation ownership matters.
- Port Japanese tokenization more faithfully if Japanese parity becomes important; the current fallback is not `nagisa`-equivalent.
- Add a higher-level `transcribe + align` helper once the merge policy is clear.
- Revisit whether the aligner should eventually share more code with the main audio encoder runtime or remain a separate graph implementation.

### Gotchas

- The forced aligner does not align raw display substrings; it aligns normalized lexical units, so `/r/apple` becomes `rapple` on the direct aligner path.
- The official Python `return_time_stamps=True` ASR path still concatenates per-chunk text and per-chunk aligned items; it does not yet solve long-audio overlap ownership for us.
- The sibling `../qwen3-asr.cpp` aligner graph was a useful structural reference, but its tokenizer behavior is not Python-parity and should not be copied blindly for languages beyond the currently validated paths.

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
- Added decoder raw-text streaming through the public `q3asr_transcribe_params` callback and the CLI `--stream-raw` mode.
- Added regression coverage for:
  - token streaming callback firing (`q3asr-streaming-regression`)
- Manual long-file validation now succeeds on `../qwen3-asr-llamacpp/i-ship-code-i-dont-read.wav` with:
  - `--max-tokens 1024`
- That 114-minute wav now completes and produces the expected podcast transcript instead of failing the old single-shot path.
- The decoder callback path was also tightened to suppress invalid UTF-8 partial updates during token streaming.
- The experimental long-audio chunk/window path was rolled back after manual inspection showed it was not reliable enough on longer material.
- The current repo state is intentionally back to one-shot transcription plus decoder token streaming only.
- Validation after the rollback:
  - `ctest --test-dir build --output-on-failure` passes the remaining `3/3` active tests
  - the final streamed raw text still matches the final raw transcript on the English regression sample
- The raw Hugging Face forced aligner model was downloaded locally into:
  - `../Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B`
- The Python forced aligner reference was exercised directly on local samples through `../Qwen3-ASR/.venv/bin/python` on MPS:
  - English sample: `testdata/q3asr-input.wav`
  - reference text: `You can apparently promote on Sundays on /r/apple on Reddit.`
  - observed aligned units: `You`, `can`, `apparently`, `promote`, `on`, `Sundays`, `on`, `rapple`, `on`, `Reddit`
  - Chinese sample: `testdata/asr_zh.wav`
  - reference text: `甚至出现交易几乎停滞的情况。`
  - observed aligned units were character-level: `甚`, `至`, `出`, `现`, `交`, `易`, `几`, `乎`, `停`, `滞`, `的`, `情`, `况`
- That direct Python run confirms an important integration constraint:
  - the forced aligner operates on normalized lexical units, not final display substrings with punctuation
  - punctuation and symbols can be removed or normalized inside an aligned unit, as shown by `/r/apple -> rapple`
- The Python `Qwen3ASRModel.transcribe(..., return_time_stamps=True)` path was also exercised locally with the base `0.6B` ASR model plus the new aligner model.
- On the same English wav, the returned transcript text and the timestamp units were related but not identical:
  - text: `You can apparently promote on Sundays on SlashR SlashApple on Reddit.`
  - aligned units near the punctuation site: `SlashR`, `SlashApple`
- The official Python implementation does not use the forced aligner to reconcile chunk overlap text.
- Its `return_time_stamps=True` path is still:
  - split audio to `<= 180s`
  - run ASR per chunk
  - run forced alignment on each chunk transcript
  - offset timestamps by chunk start
  - concatenate text and concatenate aligned items
- The sibling `../qwen3-asr.cpp/src/forced_aligner.cpp` implementation is useful for model loading, graph structure, classifier-head decoding, and timestamp post-processing, but its text tokenization is not Python-parity:
  - it only special-cases Korean and otherwise whitespace-splits text before BPE
  - that is materially different from the Python `Qwen3ForceAlignProcessor`, which handles Chinese mixed text character-by-character, Japanese via `nagisa`, Korean via a dictionary tokenizer, and punctuation stripping/normalization for space-delimited languages
- The current implementation direction for this repo should therefore be:
  - add the forced aligner as a separate optional post-pass subsystem first
  - port the Python aligner processor semantics into C++ instead of blindly copying the sibling C++ tokenizer path
  - expose a direct `align transcript against audio` API before tying it into long-audio transcription
  - represent aligned output as normalized units plus raw-text span mappings back into the original decoder text
  - use those aligned units and absolute timestamps to decide chunk-overlap ownership later, while preserving punctuation from the chosen raw text span rather than reconstructing final text from aligner units alone

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
- Implement the real push-audio session API from the design doc; the current library path is one-shot transcription plus decoder-token streaming, not microphone/live-session streaming.
- Capture the official Python streaming wrapper behavior in tests before reintroducing any long-audio/session policy.
- Convert and vendor the forced aligner model into this repo's local artifact layout once the runtime integration starts, instead of depending only on the Python-side raw safetensor checkout.
- Add a direct forced-aligner regression test that checks normalized unit output and monotonic timestamps on both a short English sample and a short Chinese sample.
- Build the future long-audio merger on aligned-unit timestamps plus raw-text span ownership; do not reintroduce overlap chunking by plain text concatenation.

### Gotchas

- The local repo at `../llama.cpp` appears older than the upstream snapshot bundled near `qwen3-asr-llamacpp`, so the latter is the safer base for decoder integration.
- The Qwen3-ASR text model chat template is present in GGUF metadata, but the initial library path is still using an explicit prompt builder so the audio placeholder injection is fully under library control.
- The split mmproj format stores the exact encoder tensor shapes, which is useful enough to load from GGUF metadata directly instead of reconstructing every tensor shape manually.
- `clip.audio.conv_chunksize` in the GGUF metadata looks like a conv time-chunk parameter, but for Qwen3-ASR it is really the conv batching size from the Python implementation; the encoder time chunk is still `n_window * 2`.
- The prompt format from the Python reference and the GGUF chat template were already effectively correct for this library path; the observed English-quality gap was not caused by the extra `s` hypothesis once an audio-only single-turn parity check was run.
- The first working smoke test currently proves "real transcription happened" by requiring a parsed non-empty text result and the detected language `Chinese`; it does not yet lock down an exact transcript.
- There is now an optional `q3asr-english-regression` test that is registered when `testdata/q3asr-input.wav` exists locally; it checks for detected language `English` and the `/r/apple` substring to catch this mel-front-end regression.
- There are now two different "streaming" concepts in the codebase:
- The only active streaming behavior in the codebase right now is raw token streaming during one decode.
- The live session/prefix-rollback behavior from the Python vLLM wrapper is still not implemented yet.
- The official Python repo still has two different long-audio behaviors to study for future work:
  - offline `transcribe()` splits long audio and concatenates per-chunk text
  - vLLM-only `streaming_transcribe()` re-feeds accumulated audio and prepends a rollback-trimmed decoded prefix on each step
- The current wav loader path is intentionally narrow: mono/stereo WAV parsing is supported, but the transcription entrypoint rejects non-16 kHz input instead of resampling it.
- Float WAV support is now in place for `pcm_f32le` / format `3`, but the entrypoint still intentionally rejects non-16 kHz input instead of resampling it.
- The previous reconfigure issue seen before artifact migration is no longer reproducing with the current local `third_party/llama.cpp` layout; plain reconfigure/build works again in this workspace.
- The upstream `../Qwen3-ASR/pyproject.toml` currently claims Python `>=3.9`, but one pinned dependency (`accelerate==1.12.0`) forces Python `>=3.10`, so plain `uv sync` fails until that metadata mismatch is addressed.
- Importing the full top-level `qwen_asr` package was not necessary for the parity check and appeared to pull in heavier optional tooling; importing only `qwen_asr.core.transformers_backend` plus the inference utilities was enough to register the custom config/model/processor classes and run the local safetensor model.

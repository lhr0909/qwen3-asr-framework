# Agent Notes

This file is project-specific guidance for agents working in `qwen3-asr-hax`.

## Project Intent

Treat this repo as a Qwen3-ASR bring-up and parity project first, not a cleanup or abstraction exercise.

Priority order:

1. keep the encoder and decoder path correct
2. validate behavior against the sibling references
3. only then refactor or broaden the surface area

When behavior changes, update `docs/worklog.md` with:

- what changed
- how it was validated
- remaining work
- any new gotchas or constraints

## Local References

Use these sibling directories before guessing:

- `../qwen3-asr-llamacpp`
  - streamlined design doc: `docs/streamlined-qwen3-asr-library.md`
  - patched mtmd runtime
- `../Qwen3-ASR`
  - official Python reference
  - prompt, processor, and safetensor parity checks
- `../qwen3-asr.cpp`
  - older monolithic mel/encoder reference

If a result diverges, confirm the behavior against at least one reference implementation before changing the code blindly.

Canonical local artifact locations in this repo:

- `third_party/llama.cpp`
- `models/gguf/`
- `testdata/`

Canonical base safetensor model locations for the Python reference:

- `../Qwen3-ASR/models/Qwen3-ASR-1.7B`
- `../Qwen3-ASR/models/Qwen3-ASR-0.6B`
- `../Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B`

Compatibility symlinks were left in the reference repos after migration. Do not add new canonical paths there.

## Architecture Map

- `src/mel_spectrogram.cpp`
  - wav parsing
  - mel filter generation
  - Qwen3-ASR log-mel front end
- `src/gguf_loader.cpp`
  - mmproj GGUF metadata parsing
  - tensor binding and mmap-backed weight access
- `src/audio_encoder.cpp`
  - ggml audio encoder execution
- `src/decoder_llama.cpp`
  - `llama.cpp` text model loading
  - prompt construction
  - audio embedding injection
  - greedy decode
- `src/forced_aligner.cpp`
  - forced-aligner GGUF loading
  - Python-style unit normalization/tokenization
  - audio encoder + classifier-head decoder
  - timestamp repair and structured aligned-item output
- `src/q3asr.cpp`
  - one-shot recognizer wiring
  - public C API
  - transcript parsing
- `tests/smoke_test.cpp`
  - transcription smoke/regression harness
- `tests/aligner_test.cpp`
  - forced-aligner real-model regression harness

## Validation Workflow

For any non-trivial change, prefer this order:

1. Build the repo.

```sh
cmake -S . -B build
cmake --build build -j
```

2. Run the local tests.

```sh
ctest --test-dir build --output-on-failure
```

3. Run the local CLI on a known sample.

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio testdata/Recording_20260202131426.wav \
  --show-raw
```

Token-streaming spot check:

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio testdata/q3asr-input.wav \
  --stream-raw
```

Forced-aligner spot checks:

```sh
./build/q3asr-cli \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/q3asr-input.wav \
  --align-text "You can apparently promote on Sundays on /r/apple on Reddit." \
  --language English
```

```sh
./build/q3asr-cli \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/asr_zh.wav \
  --align-text "甚至出现交易几乎停滞的情况。" \
  --language Chinese
```

Long-file spot check:

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio ../qwen3-asr-llamacpp/i-ship-code-i-dont-read.wav \
  --max-tokens 1024
```

4. If the change touches parity-sensitive areas, compare against the sibling patched `llama.cpp` CLI.

```sh
cd ../qwen3-asr-llamacpp
./llama.cpp/build/bin/llama-mtmd-cli \
  -m ../qwen3-asr-hax/models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj ../qwen3-asr-hax/models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio ../qwen3-asr-hax/testdata/Recording_20260202131426.wav \
  -ngl 99 -c 4096 -n 256 --temp 0
```

5. If the change may affect prompt construction or model-level quality, compare against the Python safetensor reference in `../Qwen3-ASR`.

Working setup in this workspace:

```sh
cd ../Qwen3-ASR
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . torch
```

Important:

- The active local safetensor model directories now live under `../Qwen3-ASR/models/`.
- `uv sync` currently fails against the upstream `pyproject.toml` because the project claims `requires-python = ">=3.9"` but pins `accelerate==1.12.0`, which requires Python `>=3.10`.
- For lightweight parity scripts, importing `qwen_asr.core.transformers_backend` plus inference utilities is sufficient. Importing the full top-level `qwen_asr` package is unnecessary for the current checks and may pull in heavier optional dependencies.

## Testing Conventions

Keep tests practical and parity-focused.

Current expectations:

- use real local model files instead of mocks for the main smoke path
- gate tests in `CMakeLists.txt` on the existence of local model/audio assets
- prefer language checks and stable substrings over exact full-transcript matches unless the output is known to be stable across backends and hardware
- add a regression test when fixing a parity bug that could plausibly return later
- if you touch token streaming, keep `q3asr-streaming-regression` passing
- if you touch aligner normalization or timestamp extraction, keep both `q3asr-aligner-english` and `q3asr-aligner-chinese` passing

If you add another transcript regression:

- use a short local audio sample
- prefer a stable substring that captures the actual bug
- keep the failure message readable through `tests/smoke_test.cpp`

If you add another aligner regression:

- prefer normalized units that were confirmed against the Python aligner
- check monotonic timestamps, not just token text
- favor short local samples whose unitization is stable across backends

## Conventions And Constraints

- Do not treat random interactive input typed into a reference CLI as part of the intended prompt. Confirm prompt structure from the Python reference or the model template.
- The current decoder prompt path is explicit in `src/decoder_llama.cpp`. Keep prompt changes justified and validated.
- The only active streaming behavior in this repo right now is token streaming from the decoder callback.
- This is not the same as the future live session API from the design doc.
- The forced aligner is a separate post-pass today. It does not yet participate in chunk ownership or long-audio stitching.
- Keep KV positions monotonic across:
  - prefix tokens
  - injected audio embeddings
  - suffix tokens
  - generated tokens
- The mel front end is sensitive. Do not swap in a different FFT or mel path without a regression check against the sibling runtime.
- `clip.audio.conv_chunksize` is not the time-axis encoder split for Qwen3-ASR. The active interpretation is correct: use `n_window * 2` mel frames for the actual time chunk.
- Long-audio chunking/session behavior was intentionally removed for now after incorrect results on longer files. Do not reintroduce it casually without validating against the Python reference semantics.
- The Python vLLM streaming path is still the reference algorithm for future session work: it re-feeds accumulated audio and prepends a rollback-trimmed decoded prefix on each step.
- The Python forced aligner reference is the semantics target for normalized units:
  - default languages use whitespace splitting plus punctuation stripping and Chinese-character splitting
  - Chinese mixed text is character-level for CJK spans
  - `/r/apple` normalizes to `rapple`
  - Japanese in this repo is still a best-effort fallback, not `nagisa` parity
- The entrypoint still requires `16 kHz` audio. Float wav is supported; resampling is not.
- Logging from `llama.cpp` is muted by default. Use `Q3ASR_LLAMA_VERBOSE=1` while debugging decoder issues.

## Dependency Constraints

- The build now depends on `third_party/llama.cpp`, which was migrated from the sibling reference repo.
- Do not replace or rebase that dependency casually while parity work is still ongoing.
- The migrated local layout now reconfigures successfully with `cmake -S . -B build`. If that changes again, note it in the worklog and do not assume the library code is at fault before checking the upstream dependency.

## When To Add Documentation

Update the root docs when the public behavior changes:

- `README.md`
  - build steps
  - CLI usage
  - API surface
  - current limitations
- `docs/worklog.md`
  - progress log
  - validation notes
  - open issues and gotchas

If you add a new validation path or a new required local asset, document it immediately.

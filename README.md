# q3asr-hax

`q3asr-hax` is an experimental C/C++ Qwen3-ASR library bring-up.

The current scope is narrow on purpose:

- own the Qwen3-ASR audio path inside this repo
- load the split Qwen3-ASR models (`text.gguf` + `mmproj.gguf`)
- run mel preprocessing and the audio encoder locally
- use upstream `llama.cpp` as the decoder runtime
- expose a small C API and a wav-file CLI
- keep behavior checked against the sibling Python and patched `llama.cpp` references

This is not a finished product yet. The encoder and decoder path runs end to end, but the project is still in the parity-and-bring-up phase.

## Current State

- One-shot transcription works with local split GGUF weights.
- Raw decoder text can now be streamed token-by-token through the C API callback and the CLI `--stream-raw` mode.
- Transcription now supports Python-style prompt context through one system-message string per request.
- Long-audio transcription now has an aligner-backed chunked mode with bounded decoder windows and forced-aligner-owned span merging.
- A separate console-oriented chunked streaming CLI now renders committed text plus provisional partial text during long-audio transcription.
- The public API supports:
  - `q3asr_transcribe_wav_file()`
  - `q3asr_transcribe_pcm_f32()`
  - `q3asr_align_wav_file()`
  - `q3asr_align_pcm_f32()`
  - `q3asr_align_wav_file_ex()`
  - `q3asr_align_pcm_f32_ex()`
- The CLI supports direct wav-file transcription through `q3asr-cli`.
- The CLI can transcribe long audio continuously when you provide `--aligner-model` plus `--audio-chunk-sec`.
- The repo also builds `q3asr-chunk-stream-cli` for console-first long-audio streaming with committed/partial rendering.
- The CLI also supports direct forced alignment with a local aligner GGUF through `--aligner-model` plus either `--align-text` or `--align-text-file`.
- The forced aligner now follows the Python timestamping reference at the audio-chunk level:
  - default `180s` max chunk length
  - low-energy boundary search
  - per-chunk timestamp offset and concatenation
- Real-model tests are wired through `ctest` when the expected local assets exist.
- The English parity regression sample currently matches both:
  - the sibling patched `llama.cpp` runtime in `../qwen3-asr-llamacpp`
  - the unconverted safetensor reference model through `../Qwen3-ASR`
- The canonical local artifact layout now lives in this repo:
  - `third_party/llama.cpp`
  - `models/gguf/`
  - `testdata/`

Current limitations:

- input must already be `16 kHz`
- wav parsing supports PCM and IEEE-float wav, but there is no resampler yet
- one-shot streaming is still the only true token-by-token mode; the new chunked streaming path is a committed/partial progress view built on top of chunk-level decode+align steps
- forced-align long-audio chunking currently assigns transcript ownership over normalized aligner units by duration, not over punctuation-preserving raw text spans
- long-audio transcription merging maps aligner-owned normalized units back into decoder text spans, but it is still heuristic and not yet exact-parity on every boundary
- the forced aligner currently targets Python parity first for English and Chinese; Japanese tokenization is still a best-effort fallback, not a `nagisa`-equivalent port
- diarization model conversion is experimental: the repo can now package selected ONNX diarization checkpoints into GGUF for C++/ggml-side inspection, but it does not execute those graphs in ggml yet

## Architecture

The runtime is split into five main pieces:

1. `src/mel_spectrogram.cpp`
   - Loads wav audio.
   - Generates mel filters.
   - Computes the Qwen3-ASR log-mel spectrogram.
   - This path is parity-sensitive and is aligned with the patched mtmd `llama.cpp` implementation.

2. `src/gguf_loader.cpp` and `src/audio_encoder.cpp`
   - Load the `mmproj.gguf` audio encoder weights and metadata.
   - Bind ggml tensors directly from the mapped GGUF payload.
   - Execute the Qwen3-ASR audio encoder graph and produce decoder-sized audio embeddings.

3. `src/decoder_llama.cpp`
   - Loads the `text.gguf` model through upstream `llama.cpp`.
   - Injects audio embeddings between the prompt prefix and suffix.
   - Greedy-decodes the transcript while keeping KV positions monotonic across prompt, audio, and generated tokens.

4. `src/q3asr.cpp`
   - Owns the public C API.
   - Connects wav loading, mel computation, encoder execution, decoder execution, and transcript parsing.
   - Parses raw model output such as `language English<asr_text>...` into structured fields.
   - Owns the aligner-backed long-audio transcription stitcher.

5. `src/forced_aligner.cpp`
   - Loads the local forced-aligner GGUF directly through ggml/GGUF metadata.
   - Reuses the Qwen3-ASR mel front end.
   - Runs the separate audio encoder + classifier-head decoder graph for timestamp-class prediction.
   - Applies Python-style timestamp repair and returns normalized aligned units with timestamps.
   - Uses Python-style audio chunking for long forced-alignment inputs.
   - For long material, mirrors the Python timestamping splitter and offsets/merges per-chunk alignments back into absolute time.

## Repository Layout

- `include/q3asr.h`: public C API
- `src/`: library implementation
- `cli/q3asr_cli.cpp`: general wav-file CLI
- `cli/q3asr_chunked_stream_cli.cpp`: console-focused long-audio chunked streaming CLI
- `tests/smoke_test.cpp`: real-model smoke/regression test entrypoint
- `third_party/llama.cpp`: local upstream decoder dependency
- `models/gguf/`: local GGUF model artifacts
- `testdata/`: local wav fixtures used by tests and manual runs
- `docs/worklog.md`: running implementation notes, gotchas, and remaining work

## Dependencies

Build requirements:

- CMake `>= 3.20`
- a C++17-capable compiler
- the local upstream decoder checkout at `third_party/llama.cpp`
- local GGUF model files, typically:
  - `models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf`
  - `models/gguf/Qwen3-ASR-1.7B-mmproj.gguf`
  - `models/gguf/qwen3-forced-aligner-0.6b-f16.gguf`

Reference model assets now live in:

- `../Qwen3-ASR/models/Qwen3-ASR-1.7B`
- `../Qwen3-ASR/models/Qwen3-ASR-0.6B`
- `../Qwen3-ASR/models/Qwen3-ForcedAligner-0.6B`

Apple-specific notes:

- Metal is enabled by default on Apple via `Q3ASR_ENABLE_METAL=ON`
- `Accelerate` is linked for the current build

## Build

Default build:

```sh
cmake -S . -B build
cmake --build build -j
```

If you need to point at a different `llama.cpp` checkout:

```sh
cmake -S . -B build -DQ3ASR_LLAMA_CPP_SOURCE_DIR=/absolute/path/to/llama.cpp
cmake --build build -j
```

Useful CMake options:

- `-DQ3ASR_BUILD_CLI=ON|OFF`
- `-DQ3ASR_BUILD_TESTS=ON|OFF`
- `-DQ3ASR_ENABLE_METAL=ON|OFF`

## Experimental Diarization Model Conversion

You can pull the official pyannote checkpoints from Hugging Face and package them into GGUF for local inspection.

```sh
uvx hf download pyannote/segmentation-3.0 \
  --local-dir models/hf/diarization/pyannote-segmentation-3.0

uvx hf download pyannote/wespeaker-voxceleb-resnet34-LM \
  --local-dir models/hf/diarization/pyannote-wespeaker-voxceleb-resnet34-LM

uvx --with torch,numpy,pyyaml,torchaudio,pyannote.audio==3.1.1 python scripts/convert_diarization_pytorch_to_gguf.py \
  --input-dir models/hf/diarization/pyannote-segmentation-3.0 \
  --output models/gguf/pyannote-segmentation-3.0-pytorch-f32.gguf \
  --kind speaker-segmentation \
  --source-repo pyannote/segmentation-3.0

uvx --with torch,numpy,pyyaml,torchaudio,pyannote.audio==3.1.1 python scripts/convert_diarization_pytorch_to_gguf.py \
  --input-dir models/hf/diarization/pyannote-wespeaker-voxceleb-resnet34-LM \
  --output models/gguf/pyannote-wespeaker-voxceleb-resnet34-LM-pytorch-f32.gguf \
  --kind speaker-embedding \
  --source-repo pyannote/wespeaker-voxceleb-resnet34-LM
```

If those PyTorch-derived GGUFs exist locally, the repo registers `q3asr-diarization-gguf` in `ctest` and validates the metadata/tensor layout through the C++ loader.

## Reference Pyannote Diarization

The repo currently has three diarization-oriented surfaces:

- `scripts/pyannote_diarization.py`
  - Python reference helper for quality checks and algorithm study
- `src/streaming_diarizer.cpp`
  - C++ online clustering and delayed aggregation derived from `diart`
  - expects external segmentation scores and speaker embeddings
- `src/offline_diarizer.cpp`
  - C++ asset/config wrapper for Community-1
  - validates the official PyTorch-derived GGUF models plus Community-1's bundled PLDA assets
  - does not execute raw-audio diarization natively yet

The Python reference helper supports two Community-1-based modes:

- `offline-community1`
  - runs the full offline `pyannote/speaker-diarization-community-1` pipeline
  - keeps the bundled VBx + PLDA clustering stage intact
- `streaming-community1`
  - runs a lower-latency approximation inspired by `diart`
  - uses overlapping rolling windows
  - remaps chunk-local speakers online with Community-1 speaker embeddings
  - commits a stable center strip from each chunk to reduce boundary churn
  - this is a practical reference path, not exact offline VBx parity

Suggested runtime setup:

```sh
uv venv --python 3.12 /tmp/pyannote-audio-4
uv pip install --python /tmp/pyannote-audio-4/bin/python pyannote.audio==4.0.0
```

The script includes a small `torch.load(..., weights_only=False)` compatibility shim for trusted local Community-1 checkpoints under newer PyTorch releases.

Offline example:

```sh
/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py \
  offline-community1 \
  --audio testdata/long-audio.wav \
  --num-speakers 2 \
  --output-json /tmp/q3asr-offline-community1-long-audio.json \
  --output-rttm /tmp/q3asr-offline-community1-long-audio.rttm
```

Streaming approximation example:

```sh
/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py \
  streaming-community1 \
  --audio testdata/long-audio.wav \
  --num-speakers 2 \
  --window-sec 20 \
  --step-sec 5 \
  --output-json /tmp/q3asr-streaming-community1-long-audio.json \
  --output-rttm /tmp/q3asr-streaming-community1-long-audio.rttm
```

Notes:

- the offline path is the quality reference
- the streaming path is intentionally simpler and lower-accuracy
- the C++ `streaming_diarizer` currently covers only the online clustering logic
- the C++ `offline_diarizer` currently covers only official-asset loading and config validation
- native C++ Community-1 execution still needs segmentation forward, speaker-embedding forward, PLDA scoring, and VBx decoding

## Run The CLI

Example:

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio testdata/Recording_20260202131426.wav \
  --show-raw
```

Context / hotword-hint example:

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --audio testdata/q3asr-input.wav \
  --context "PSPDFKit Claudebot /r/apple" \
  --show-raw
```

Long-audio transcription example:

```sh
./build/q3asr-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/long-audio.wav \
  --audio-chunk-sec 180 \
  --max-tokens 1024 \
  --show-raw
```

Chunked console streaming example:

```sh
./build/q3asr-chunk-stream-cli \
  --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/long-audio.wav \
  --context-file /tmp/q3asr-context.txt \
  --audio-chunk-sec 180 \
  --max-tokens 1024
```

That CLI renders progress on stderr as:

- `committed`: stable merged text that will not be retracted
- `partial`: provisional text from the current decode window that may still change at the next chunk boundary

Useful CLI flags:

- `--language <name>`: force a language hint
- `--context <text>`: pass Python-style prompt context / hotword hints through the system message
- `--context-file <path>`: read that context text from a file
- `--audio-chunk-sec <sec>`: max audio seconds per transcription chunk when using `--aligner-model`
- `--audio-overlap-sec <sec>`: overlap seconds between chunk decode windows; defaults to `5`
- `--max-tokens <n>`: cap decoder output length
- `--temp <value>`: decoder temperature, defaults to `0` for greedy decode
- `--threads <n>`: mel + decoder worker count
- `--batch <n>`: decoder batch size
- `--ctx <n>`: decoder context size
- `--no-gpu`: disable GPU offload
- `--stream-raw`: stream raw decoder text as it is generated
- `--show-raw`: print raw decoder output plus parsed fields

Forced-align example:

```sh
./build/q3asr-cli \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/q3asr-input.wav \
  --align-text "You can apparently promote on Sundays on /r/apple on Reddit." \
  --language English
```

Forced-align from a raw ASR output file:

```sh
./build/q3asr-cli \
  --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf \
  --audio testdata/long-audio.wav \
  --align-text-file testdata/long-audio-result.txt
```

If the file contains `language ...<asr_text>...`, the CLI automatically strips the prefix and reuses the embedded language when `--language` is not provided.

Aligner-specific CLI flags:

- `--aligner-model <path>`: local forced-aligner GGUF
- `--align-text <text>`: transcript text to align against the audio
- `--align-text-file <path>`: read align text from a file; raw `language ...<asr_text>...` content is accepted directly
- `--align-max-chunk-sec <sec>`: max audio seconds per forced-align chunk; defaults to `180`, `0` disables chunking
- `--korean-dict <path>`: optional Korean dictionary file for the aligner tokenizer

Audio input rules today:

- wav only
- `16 kHz` sample rate required
- PCM 16-bit wav supported
- IEEE-float wav supported (`format 3`, 32-bit or 64-bit)

If you need verbose upstream decoder logs:

```sh
Q3ASR_LLAMA_VERBOSE=1 ./build/q3asr-cli ...
```

## Use The Library

The public C API is defined in `include/q3asr.h`.

Minimal flow:

```c
q3asr_context_params ctx_params = q3asr_context_default_params();
ctx_params.text_model_path = ".../Qwen3-ASR-1.7B-text-Q8_0.gguf";
ctx_params.mmproj_model_path = ".../Qwen3-ASR-1.7B-mmproj.gguf";

q3asr_context * ctx = q3asr_context_create(&ctx_params);

q3asr_aligner_context_params align_ctx_params = q3asr_aligner_context_default_params();
align_ctx_params.aligner_model_path = ".../qwen3-forced-aligner-0.6b-f16.gguf";
q3asr_aligner_context * aligner = q3asr_aligner_context_create(&align_ctx_params);

q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();
tx_params.aligner_context = aligner;
tx_params.max_audio_chunk_seconds = 180.0f;
tx_params.audio_chunk_overlap_seconds = 5.0f;
q3asr_transcribe_result result = {0};

if (q3asr_transcribe_wav_file(ctx, "input.wav", &tx_params, &result)) {
    /* result.raw_text, result.language, result.text */
}

q3asr_transcribe_result_clear(&result);
q3asr_aligner_context_destroy(aligner);
q3asr_context_destroy(ctx);
```

For chunked long-audio UI, `q3asr_transcribe_params` also now exposes an optional `progress_callback` that reports `language`, `committed_text`, `partial_text`, and chunk counters.

Forced-align flow:

```c
q3asr_aligner_context_params align_ctx_params = q3asr_aligner_context_default_params();
align_ctx_params.aligner_model_path = ".../qwen3-forced-aligner-0.6b-f16.gguf";

q3asr_aligner_context * aligner = q3asr_aligner_context_create(&align_ctx_params);

q3asr_align_params align_run_params = q3asr_align_default_params();
align_run_params.max_chunk_seconds = 180.0f;
q3asr_alignment_result alignment = {0};
if (q3asr_align_wav_file_ex(
        aligner,
        "input.wav",
        "aligned text",
        "English",
        &align_run_params,
        &alignment)) {
    /* alignment.items[i].text / start_time / end_time */
}

q3asr_alignment_result_clear(&alignment);
q3asr_aligner_context_destroy(aligner);
```

## Run Tests

Build and run:

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The current tests are asset-gated:

- `q3asr-smoke`
  - Uses the default local Chinese sample.
  - Verifies that a real transcription runs and the detected language is `Chinese`.
- `q3asr-english-regression`
  - Registers only when `testdata/q3asr-input.wav` exists locally.
  - Verifies detected language `English` and the `/r/apple` substring.
- `q3asr-streaming-regression`
  - Verifies the raw-text callback fires multiple times on the English regression sample and that the last streamed raw text equals the final raw transcript.
- `q3asr-long-transcribe-regression`
  - Forces the English sample through the aligner-backed chunked transcription path.
  - Verifies the merged result stays continuous and preserves the `/r/apple` phrase.
- `q3asr-aligner-english`
  - Uses the local forced-aligner GGUF and `testdata/q3asr-input.wav`.
  - Verifies the current Python-parity English unitization: `10` items including `rapple`.
- `q3asr-aligner-chunked-english`
  - Forces the same English sample through the chunked aligner path with `--max-chunk-sec 2`.
  - Verifies the chunked path preserves the same `10` normalized units.
- `q3asr-aligner-chinese`
  - Uses the local forced-aligner GGUF and `testdata/asr_zh.wav`.
  - Verifies the current Python-parity Chinese character-level unitization: `13` items from `甚` through `况`.

## Reference Implementations

This repo is intentionally validated against sibling reference projects:

- `../qwen3-asr-llamacpp`
  - patched `llama.cpp` runtime
  - source of the streamlined design doc and most decoder-parity checks
- `../Qwen3-ASR`
  - official Python reference
  - useful for prompt/template confirmation and safetensor parity
- `../qwen3-asr.cpp`
  - older monolithic encoder implementation
  - useful for mel/encoder bring-up comparisons

See `AGENTS.md` for the preferred validation workflow and project-specific constraints for future changes.

## Notes And Known Issues

- `clip.audio.conv_chunksize` in the mmproj GGUF is not the encoder time-axis split. The active code correctly derives the time chunk from `n_window * 2`.
- The old naive long-audio transcription path is still gone. The new long-audio transcription mode exists only when a forced aligner is available to own the merge.
- The forced aligner now lives in this repo as a separate subsystem, and it now also drives the transcription-side long-audio stitcher through overlapped decode windows plus timestamp-owned middle spans.
- The stitcher now also uses decoder token logprobs inside a small overlap band around each chunk boundary to arbitrate which neighboring window owns that raw-text span.
- The default long-audio decode-window overlap is `5s`, and the current boundary-arbitration band is derived from that overlap as `min(overlap, max(0.75, overlap * 0.5))`, so the default boundary band is `2.5s`.
- The current forced-aligner tokenizer path is designed around Python parity for English/Chinese/Korean normalization, but Japanese is still a fallback heuristic rather than a port of Python's `nagisa` segmentation.
- The repo now includes a copied long fixture at `testdata/long-audio.wav` plus a raw transcript file at `testdata/long-audio-result.txt` for manual alignment experiments.
- The new `--align-text-file` CLI path plus default chunked alignment makes those experiments practical; the 15-minute fixture completed locally in about `148s` and produced `3002` monotonic aligned units.
- The current long-audio aligner assigns chunk ownership over normalized lexical units by duration. That is sufficient for timestamp experiments, but it is not yet the final punctuation-preserving merge policy.
- The end-to-end 15-minute transcription path completed locally in about `249s` with `--audio-chunk-sec 180 --max-tokens 1024`, produced one continuous `language English<asr_text>...` block, and the stitcher now arbitrates boundary bands with decoder confidence, but the result still drifted at the tail (`600.` vs. the reference `600 bucks.`).
- The Python vLLM streaming path is still the reference for future session behavior: it re-feeds accumulated audio and appends a rollback-trimmed text prefix on each step.
- Compatibility symlinks were left in the sibling reference repos after artifact migration, but this repo is now the canonical home for the GGUF models, test wavs, and the upstream `llama.cpp` checkout.
- Long-audio behavior, full streaming/session support, and resampling are still pending.

## Worklog

Ongoing implementation notes live in `docs/worklog.md`.

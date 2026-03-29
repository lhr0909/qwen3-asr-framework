# Q3ASR Worklog

## 2026-03-29

### Native Offline Clustering Bring-Up

- Extended `src/offline_diarizer.cpp` and `src/offline_diarizer.h` from an asset wrapper into a real native Community-1 clustering stage for precomputed features:
  - loads the official segmentation and embedding GGUFs
  - loads a converted Community-1 PLDA GGUF
  - runs the x-vector transform and PLDA projection in C++
  - runs centroid-linkage AHC initialization in C++
  - runs the VBx update loop in C++
  - applies constrained Hungarian assignment back onto the per-chunk local speakers
- Added two helper scripts:
  - `scripts/convert_community1_plda_to_gguf.py`
    - converts `plda/plda.npz` and `plda/xvec_transform.npz` into a GGUF the C++ runtime can load directly
  - `scripts/export_community1_offline_fixture.py`
    - exports a short Python-reference fixture with binarized segmentations, speaker embeddings, expected hard clusters, and expected centroids
- Expanded `tests/offline_diarizer_test.cpp` from asset validation into a real parity regression:
  - loads a short Community-1 fixture exported from the Python reference
  - runs the native C++ offline clustering stage
  - checks the predicted hard clusters and centroids against the Python reference output
- Updated `CMakeLists.txt` so `q3asr-offline-diarizer` now requires:
  - the official segmentation GGUF
  - the official embedding GGUF
  - the converted Community-1 PLDA GGUF
  - the exported short reference fixture
- Relaxed `src/diarization_gguf.cpp` so the common loader accepts both:
  - PyTorch-derived diarization GGUFs
  - NumPy-derived clustering GGUFs

### Native Offline Clustering Validation

- Generated the converted Community-1 PLDA GGUF locally:
  - `uvx --with numpy,scipy,pyyaml python scripts/convert_community1_plda_to_gguf.py --bundle-dir models/hf/diarization/pyannote-speaker-diarization-community-1 --output models/gguf/pyannote-speaker-diarization-community-1-plda-f32.gguf`
- Generated a short 20-second Python-reference fixture from `testdata/long-audio.wav`:
  - `/tmp/pyannote-audio-4/bin/python scripts/export_community1_offline_fixture.py --audio testdata/long-audio.wav --start-sec 0 --duration-sec 20 --num-speakers 2 --output tests/data/community1-offline-20s-long-audio.json`
  - observed: `num_chunks=11`, `num_frames=589`, `num_speakers=3`, `embedding_dim=256`
- Rebuilt successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Verified the focused diarization tests:
  - `./build/q3asr-offline-diarizer-test --community1-bundle-dir models/hf/diarization/pyannote-speaker-diarization-community-1 --segmentation-model models/gguf/pyannote-segmentation-3.0-pytorch-f32.gguf --embedding-model models/gguf/pyannote-wespeaker-voxceleb-resnet34-LM-pytorch-f32.gguf --clustering-model models/gguf/pyannote-speaker-diarization-community-1-plda-f32.gguf --reference-fixture testdata/diarization/community1-offline-20s-long-audio.json`
    - observed: `offline_diarizer_test passed`
  - `ctest --test-dir build --output-on-failure -R 'q3asr-(offline-diarizer|diarization-gguf|streaming-diarizer)'`
    - observed: `3/3` passing
- Re-ran the full test suite:
  - `ctest --test-dir build --output-on-failure`
  - observed: `12/12` passing

### Native Offline Clustering Remaining Work

- The C++ offline path now covers the Community-1 clustering half, not the raw-audio front half.
- The remaining native gaps are:
  - segmentation model forward from the official GGUF
  - speaker-embedding model forward from the official GGUF
  - reconstruction of final diarization/exclusive diarization directly from native segmentations
- The current parity test depends on a short tracked fixture under `tests/data/`.
- That fixture is intentionally short to keep `ctest` practical, so it validates the native clustering stage, not full-dataset quality.

### Native Offline Clustering Gotchas

- The new PLDA GGUF uses `diarization.serialization_format = "numpy"`, so the common GGUF loader needed to stop assuming every diarization artifact was PyTorch-derived.
- The offline parity test currently compares against the Python reference after a short fixture export step instead of running the full Python pipeline inside `ctest`.
- This keeps the regression fast and stable, but it also means the native C++ path is only proven against the exported intermediate features, not yet against raw waveform inference.

### Streaming / Offline Diarizer Split

- Renamed the `diart`-derived C++ online module to the clearer streaming name:
  - `src/streaming_diarizer.cpp`
  - `src/streaming_diarizer.h`
  - `tests/streaming_diarizer_test.cpp`
  - `q3asr-streaming-diarizer`
- Added a new C++ `offline_diarizer` surface:
  - `src/offline_diarizer.cpp`
  - `src/offline_diarizer.h`
  - `tests/offline_diarizer_test.cpp`
- Kept the scope honest:
  - `offline_diarizer` currently validates the Community-1 bundle config, PLDA assets, and official PyTorch-derived GGUFs
  - it does not claim to execute Community-1 from raw audio in C++ yet
- Cleaned the old diarization ONNX packaging path out of the active repo surface:
  - removed `scripts/convert_diarization_onnx_to_gguf.py`
  - switched the C++ GGUF loader test and CMake defaults to the official PyTorch-derived GGUFs only
  - removed the ONNX-oriented README workflow

### Streaming / Offline Diarizer Validation

- Reconfigured and rebuilt successfully after the rename and cleanup:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Verified the diarization-focused C++ tests:
  - `ctest --test-dir build --output-on-failure -R 'q3asr-(streaming-diarizer|diarization-gguf|offline-diarizer)'`
  - observed: `3/3` passing
- The new offline C++ asset wrapper validates:
  - Community-1 config parsing from `models/hf/diarization/pyannote-speaker-diarization-community-1/config.yaml`
  - presence of `plda/plda.npz` and `plda/xvec_transform.npz`
  - official segmentation GGUF metadata/tensors
  - official embedding GGUF metadata/tensors

### Streaming / Offline Diarizer Remaining Work

- Native offline Community-1 execution is still not implemented in C++.
- The real missing pieces are:
  - segmentation model forward from the official PyTorch-derived GGUF
  - speaker-embedding model forward from the official PyTorch-derived GGUF
  - PLDA scoring in C++
  - VBx decoding in C++
- Until those exist, Python remains the only end-to-end Community-1 execution reference.

### Community-1 Reference Diarization Pipelines

- Added a Python reference helper for pyannote diarization work:
  - `scripts/pyannote_diarization.py`
- Kept the scope honest about what is implemented today:
  - no claim that q3asr executes diarization models in C++ yet
  - no public C API or C++ CLI wiring yet
  - this is a Python-side reference layer to validate Community-1 behavior before porting the runtime pieces
- Added two practical modes based on `pyannote/speaker-diarization-community-1`:
  - `offline-community1`
    - runs the full offline pipeline
    - keeps Community-1's bundled VBx + PLDA clustering path intact
  - `streaming-community1`
    - uses overlapping rolling windows on audio loaded in memory
    - runs chunk-local Community-1 diarization on each window
    - remaps local chunk speakers online using Community-1 speaker embeddings
    - commits a stable center strip from each chunk to reduce edge churn
    - this is intentionally a lower-accuracy reference approximation, not exact offline VBx parity
- Documented the new reference workflow in `README.md`.

### Community-1 Reference Diarization Validation

- Verified the new script parses and exposes both modes:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py --help`
  - `python3 -m py_compile scripts/pyannote_diarization.py`
- Validated offline Community-1 on the two local two-speaker podcast cuts:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py offline-community1 --audio testdata/long-audio.wav --num-speakers 2 --output-json /tmp/q3asr-offline-community1-long-audio.json`
    - observed: `device=mps`, `speakers=2`, `segments=410`, `exclusive_segments=419`
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py offline-community1 --audio testdata/long-audio-2.wav --num-speakers 2 --output-json /tmp/q3asr-offline-community1-long-audio-2.json`
    - observed: `device=mps`, `speakers=2`, `segments=444`, `exclusive_segments=444`
- Smoke-tested the streaming approximation on a 60-second slice:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio /tmp/long-audio-60.wav --num-speakers 2 --window-sec 20 --step-sec 5 --output-json /tmp/q3asr-streaming-community1-60s.json --verbose`
    - observed: `chunks=9`, `global_speakers=2`, `segments=21`
- Validated the default streaming settings on both local long podcast cuts:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio testdata/long-audio.wav --num-speakers 2 --output-json /tmp/q3asr-streaming-community1-long-audio.json`
    - observed: `device=mps`, `chunks=180`, `global_speakers=2`, `segments=611`
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio testdata/long-audio-2.wav --num-speakers 2 --output-json /tmp/q3asr-streaming-community1-long-audio-2.json`
    - observed: `device=mps`, `chunks=197`, `global_speakers=2`, `segments=724`

### Community-1 Reference Diarization Remaining Work

- Mirror the validated Python-side structure into the real C++ runtime:
  - add Community-1-compatible segmentation execution
  - add Community-1-compatible speaker embedding execution
  - feed those outputs into the existing internal `StreamingDiarizer`
- Decide whether Community-1 VBx/PLDA should become:
  - an offline-only refinement stage
  - or a later micro-batch reconciliation path layered on top of the streaming approximation
- Add a stable comparison harness between:
  - offline Community-1
  - streaming Community-1 approximation
  - future C++ diarization output

### Community-1 Reference Diarization Gotchas

- Community-1 with modern PyTorch requires a small trusted-checkpoint compatibility shim:
  - newer `torch.load` defaults to `weights_only=True`
  - the local script forces `weights_only=False` when the caller leaves it unset so the official pyannote checkpoints can load
- The streaming mode is not a faithful online implementation of Community-1's VBx/PLDA clustering.
- It intentionally uses Community-1 at the chunk level, then performs simpler online speaker remapping between chunks.
- That makes it useful as a design and quality reference, but not as the final parity target.

## 2026-03-28

### Diart Port Bring-Up

- Studied the upstream `diart` package and kept the port scoped to the parts that are both portable and useful in this repo today:
  - `src/streaming_diarizer.cpp`
  - `src/streaming_diarizer.h`
- Ported the online diarization runtime pieces instead of vendoring the Python package or pretending the pyannote models were already portable:
  - overlap-aware speaker weighting with the `diart` Equation-2 style penalty
  - ggml-backed per-frame softmax for that weighting step
  - L2 speaker-embedding normalization
  - delayed overlap aggregation with `mean`, `hamming`, and `first` strategies
  - constrained online speaker clustering with Hungarian assignment, new-speaker thresholding, and centroid updates
  - a small stateful `StreamingDiarizer` wrapper that emits aggregated and binarized speaker activity per step
- Kept the new surface internal for now:
  - no public C API additions yet
  - no CLI wiring yet
  - no attempt to claim end-to-end diarization quality without ggml speaker models
- Added a synthetic regression target:
  - `tests/streaming_diarizer_test.cpp`
  - `q3asr-streaming-diarizer`
- Wired the module into the normal library build in `CMakeLists.txt`.

### Diart Port Validation

- Reconfigured and rebuilt successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Verified the new diarization regression directly:
  - `./build/q3asr-streaming-diarizer-test`
  - observed: `streaming_diarizer_test passed`
- Re-ran the full repo suite after adding the new module:
  - `ctest --test-dir build --output-on-failure`
  - observed: `10/10` passing
- The new regression covers the main algorithmic pieces we ported:
  - overlap penalty matches a hand-computed softmax example
  - embedding normalization produces unit-length vectors
  - delayed aggregation reproduces the `diart` frame-count behavior on the overlapping-buffer example
  - clustering preserves speaker identity across local-speaker reordering and opens a new global speaker when the distance threshold is exceeded

### Diart Port Remaining Work

- This is not a full `diart` replacement yet.
- The current port expects external segmentation scores and speaker embeddings as inputs.
- The actual neural models used by upstream `diart` are still missing from this repo in ggml form:
  - speaker segmentation
  - speaker embedding extraction
- If diarization is going to become a first-class feature here, the next practical step is to choose a concrete model path:
  - convert a small segmentation model to gguf/ggml and integrate it
  - convert a speaker embedding model to gguf/ggml and integrate it
  - only then add a public C API and CLI path

### Diart Port Gotchas

- Upstream `diart` is a Python pipeline around pyannote-style models, not a standalone diarization model bundle.
- The algorithm port was the realistic part to bring over in one pass; the neural model port is the real remaining cost.
- `diart`'s delayed aggregation depends on pyannote's fixed-duration loose crop semantics, which are slightly counterintuitive at the frame boundaries:
  - the crop count is effectively `round(duration / frame_step) + 1`
  - matching that behavior was necessary to reproduce the upstream overlapping-window example
- The new module currently uses ggml only for the overlap-penalty softmax math.
- That is intentional:
  - it keeps the port honest about what is already implemented
  - it avoids introducing fake "ggml diarization" claims before speaker models actually exist in ggml form

### Diarization Model Bring-Up

- Pulled local diarization model artifacts from Hugging Face with `uvx hf` instead of assuming the default gated pyannote repos were directly available:
  - `uvx hf download onnx-community/pyannote-segmentation-3.0`
  - `uvx hf download chengdongliang/wespeaker voxceleb_resnet34_LM.onnx`
- Confirmed the default `diart` repos are still gated for this account in CLI usage:
  - `pyannote/segmentation`
  - `pyannote/embedding`
  - both returned `Access denied. This repository requires approval.`
- Added a small ONNX-to-GGUF packaging script:
  - `scripts/convert_diarization_onnx_to_gguf.py`
  - it stores graph metadata plus initializer tensors in GGUF for C++/ggml-side inspection
  - this is intentionally a bring-up format conversion, not a claim that q3asr can already execute arbitrary ONNX graphs
- Added a C++/ggml-side loader and regression for these packaged diarization GGUFs:
  - `src/diarization_gguf.cpp`
  - `src/diarization_gguf.h`
  - `tests/diarization_gguf_test.cpp`
  - `q3asr-diarization-gguf`
- Kept the scope honest about runtime support:
  - the segmentation ONNX graph includes `Conv`, `LSTM`, `InstanceNormalization`, pooling, and shape-control ops
  - the embedding ONNX graph is a ResNet-style conv stack with reductions and a final projection
  - GGUF conversion is straightforward, but full ggml execution still needs dedicated graph/runtime implementation

### Official PyTorch Diarization Conversion

- Retried the original gated pyannote repos after access was granted and confirmed the official checkpoints are now downloadable:
  - `pyannote/segmentation-3.0`
  - `pyannote/wespeaker-voxceleb-resnet34-LM`
  - `pyannote/speaker-diarization-3.1`
- Confirmed an important reference detail from the official `speaker-diarization-3.1` pipeline config:
  - the end-to-end pipeline uses `pyannote/segmentation-3.0`
  - its embedding backend is `pyannote/wespeaker-voxceleb-resnet34-LM`, not the older `pyannote/embedding`
- Added a second converter path for the official PyTorch checkpoints:
  - `scripts/convert_diarization_pytorch_to_gguf.py`
  - it packages PyTorch state-dict tensors plus selected checkpoint/config metadata into GGUF
  - it keeps the serialization format explicit as `pytorch` instead of pretending these checkpoints are ONNX graphs
- Expanded the C++ GGUF inspection path to understand both packaging formats:
  - `src/diarization_gguf.cpp`
  - `src/diarization_gguf.h`
  - `tests/diarization_gguf_test.cpp`
  - `q3asr-diarization-gguf-pytorch`
- Kept the runtime claim narrow here as well:
  - this is still GGUF packaging and loader validation only
  - q3asr still does not execute the official pyannote PyTorch models in ggml yet

### Official PyTorch Diarization Validation

- Downloaded the official checkpoints locally:
  - `uvx hf download pyannote/segmentation-3.0`
  - `uvx hf download pyannote/wespeaker-voxceleb-resnet34-LM`
  - `uvx hf download pyannote/speaker-diarization-3.1`
- Inspected the checkpoint structure before conversion:
  - `pyannote/segmentation-3.0` is a PyTorch Lightning checkpoint with `54` state-dict tensors
  - `pyannote/wespeaker-voxceleb-resnet34-LM` is a PyTorch Lightning checkpoint with `218` state-dict tensors
- The new converter includes small compatibility shims for modern environments:
  - restores `torchaudio.set_audio_backend` / `get_audio_backend` expectations used by `pyannote.audio==3.1.1`
  - restores `np.NaN` for older pyannote codepaths under NumPy 2.x
- Planned validation path after conversion:
  - convert both official checkpoints to GGUF
  - load them through `q3asr-diarization-gguf-test`
  - confirm the tensor counts and key metadata match the official checkpoint structure

### Diarization Model Validation

- Verified Hugging Face CLI usage from the official docs and the local tool:
  - official docs: `uvx hf auth login`, `uvx hf download`, `hf download --local-dir` usage
- Confirmed account auth locally:
  - `uvx hf auth whoami`
- Downloaded accessible diarization models:
  - `onnx-community/pyannote-segmentation-3.0`
  - `chengdongliang/wespeaker` `voxceleb_resnet34_LM.onnx`
- Inspected the downloaded graphs:
  - segmentation model:
    - input `input_values` with shape `batch_size,num_channels,num_samples`
    - output `logits` with shape `batch_size,num_frames,7`
    - ops include `Conv=2`, `LSTM=4`, `InstanceNormalization=4`
  - embedding model:
    - input `feats` with shape `B,T,80`
    - output `embs` with shape `B,256`
    - ops include `Conv=36`, `Relu=33`, `Gemm=1`
- Ran the segmentation ONNX model on `testdata/long-audio.wav` through ONNX Runtime as a reference spot check:
  - on 60-second slices, the model emitted multiple speaker classes in later windows, e.g.:
    - `120s`: classes `{0, 2, 3}`
    - `300s`: classes `{0, 2, 3, 6}`
  - that is enough evidence that `testdata/long-audio.wav` is a reasonable local diarization fixture for continued bring-up

### Diarization Model Remaining Work

- q3asr still does not execute these diarization ONNX graphs in ggml.
- The next real runtime milestone is not more container work:
  - implement the segmentation graph in ggml or add a limited ONNX importer/runtime path
  - implement the embedding graph plus feature extraction (`80`-bin speaker features for the wespeaker model)
  - feed those outputs into the existing `StreamingDiarizer`
- If the user wants exact `diart` parity instead of accessible public models, the Hugging Face account still needs repository approval for the gated pyannote checkpoints before CLI download will work.

## 2026-03-22

### Thai Long-Audio Follow-Up

- Fixed the immediate Thai long-audio failure in the aligner normalization path:
  - `src/forced_aligner.cpp` no longer drops Thai code points as non-alignable text
  - Thai normalization now uses Apple `CFStringTokenizer` word segmentation on macOS builds
  - non-Apple builds keep a dependency-free Thai fallback that produces grapheme-like alignable units instead of an empty unit list
- Kept the long-audio stitcher from inserting synthetic spaces for space-less languages:
  - `src/q3asr.cpp` now treats Thai the same way as Chinese/Japanese when joining recovered fragments
- Added a lightweight normalization regression:
  - `tests/normalization_test.cpp`
  - `q3asr-normalization`
- Kept the change scoped to tokenizer/merge behavior:
  - no decoder prompt changes
  - no aligner graph/model changes

### Thai Long-Audio Validation

- Reconfigured and rebuilt successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Full suite passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `9/9` passing
- Verified the new normalization regression directly:
  - `./build/q3asr-normalization-test`
  - observed: `english=10`, `chinese=13`, `thai=31`
- Cut a local 30-second Thai clip for a direct aligner spot check:
  - `ffmpeg -y -i testdata/media_wnh75e4vhiyatfpo.wav -t 30 -acodec pcm_s16le /tmp/q3asr-thai-30s.wav`
- Verified that the forced aligner now accepts Thai transcript text on that clip:
  - `./build/q3asr-aligner-test --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio /tmp/q3asr-thai-30s.wav --text '<thai transcript>' --language Thai`
  - observed: `count=88` with monotonic timestamps instead of `Forced aligner input text produced no alignable units`
- Re-ran the real chunked streaming command on `testdata/media_wnh75e4vhiyatfpo.wav`:
  - `./build/q3asr-chunk-stream-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio ./testdata/media_wnh75e4vhiyatfpo.wav --audio-chunk-sec 30 --audio-overlap-sec 3 --max-tokens 1024`
  - observed: progressed through chunks `1/120` to `4/120` before manual interrupt, so the original chunk-1 zero-unit failure is gone

### Thai Long-Audio Remaining Work

- The tokenizer gap is fixed, but Thai forced-alignment quality still needs longer manual evaluation:
  - the official Qwen3 forced-aligner docs still do not list Thai among the model's supported aligner languages
  - this change proves the runtime no longer fails on Thai tokenization, not that Thai timestamp quality is reference-grade yet
- Non-Apple builds still use the local fallback tokenizer instead of a dictionary-backed segmenter.
- If Thai long-audio quality remains a priority on non-Apple hosts, evaluate a portable dictionary segmenter or a vendored library with real Thai word breaking rather than UAX-29-only token boundaries.
- Do not vendor ICU into `third_party/` for this issue right now:
  - Apple builds already have a low-friction system tokenizer path via `CFStringTokenizer`
  - if non-Apple Thai segmentation needs to improve, prefer an optional system ICU integration first
  - only pull ICU into the repo if identical cross-platform segmentation becomes a hard project requirement worth the build and data-packaging cost

### Thai Long-Audio Gotchas

- Direct ICU `BreakIterator` integration was not practical in this macOS workspace:
  - `libicucore` is present in the SDK
  - the SDK does not ship the `brkiter.h` headers needed for a normal C++ ICU integration
  - `CFStringTokenizer` was the lowest-friction system API that actually exposed Thai word segmentation here
- A standards-only splitter such as Rust `unicode-segmentation` is not enough for Thai words in this workload:
  - on the same Thai sample it split into many grapheme-like pieces, not the dictionary-style word boundaries that `Intl.Segmenter` / Apple tokenization returned
  - that makes it a poor fit for the aligner ownership surface even though it is easy to build

## 2026-03-18

### Context Follow-Up

- Added Python-style prompt context to the transcription API:
  - `q3asr_transcribe_params.context`
- Threaded that context into the decoder prompt builder in `src/decoder_llama.cpp`:
  - the system message now mirrors the Python reference shape
  - there is still a single audio user turn plus the existing `language ...<asr_text>` suffix behavior
- Extended the CLIs with:
  - `q3asr-cli --context <text>`
  - `q3asr-cli --context-file <path>`
  - `q3asr-chunk-stream-cli --context <text>`
  - `q3asr-chunk-stream-cli --context-file <path>`
- Extended the smoke harness with:
  - `--context <text>`
  - `--context-file <path>`
- Added a prompt-context regression:
  - `q3asr-context-regression`

### Context Validation

- Rebuilt successfully:
  - `cmake --build build -j`
- Full suite still passes after the prompt change:
  - `ctest --test-dir build --output-on-failure`
  - observed: `8/8` passing
- Manual CLI spot check with inline context:
  - `./build/q3asr-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --audio testdata/q3asr-input.wav --context "PSPDFKit Claudebot /r/apple" --show-raw`
  - observed: same expected `/r/apple` transcript
- Manual CLI spot check with context file:
  - `./build/q3asr-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --audio testdata/q3asr-input.wav --context-file /tmp/q3asr-context.txt --show-raw`
  - observed: same expected `/r/apple` transcript

### Context Remaining Work

- This matches the Python reference mechanism for hint injection, but it is still prompt conditioning, not decoder logit biasing.
- The public C API still takes a single context string because this repo only exposes one-audio transcription calls today.
- If multi-audio batching is added later, context should follow the Python semantics:
  - one context string per input audio
  - broadcast a scalar string across the batch

### Goal

Follow the official Python timestamping path closely enough to make long-audio forced alignment practical in this repo.

### What Changed

- Kept the new long-audio fixtures local in this repo:
  - `testdata/long-audio.wav`
  - `testdata/long-audio-result.txt`
- Kept the new raw-output-file CLI path:
  - `--align-text-file <path>`
- Added chunked forced-alignment runtime params in the public API:
  - `q3asr_align_params`
  - `q3asr_align_default_params()`
  - `q3asr_align_wav_file_ex()`
  - `q3asr_align_pcm_f32_ex()`
- Extended `q3asr-cli` with:
  - `--align-max-chunk-sec <sec>`
- Implemented a Python-style long-audio forced-align path in `src/forced_aligner.cpp`:
  - default `180s` max chunk size
  - `5s` low-energy boundary search window
  - `100ms` local energy window
  - padding of too-short chunks to the Python minimum input duration
  - per-chunk timestamp offset and concatenation back into one result
- Kept short-input behavior unchanged by routing only over-limit audio through the chunked path.
- Reused the aligner's normalized lexical units as the chunk-ownership surface:
  - the full transcript is normalized once
  - normalized units are assigned to audio chunks proportionally by cumulative duration
  - each chunk then aligns only its assigned normalized-unit span
- Added a real-model chunked alignment regression:
  - `q3asr-aligner-chunked-english`
- Kept the earlier `--align-text-file` parsing behavior:
  - if the file contains `language ...<asr_text>...`, the CLI strips the prefix and reuses the embedded language automatically
- Kept the small CMake fallback so stale cached test paths recover to repo-local defaults under `testdata/`.

### Validation

- Rebuilt and reconfigured successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Full test suite passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `6/6` passing
- Verified the raw-output-file path on the short English sample with a temporary file containing:
  - `language English<asr_text>You can apparently promote on Sundays on /r/apple on Reddit.`
- Verified the forced chunked path on the same English sample through:
  - `q3asr-aligner-chunked-english`
  - observed: same `10` normalized units, including `rapple`
- Ran a real long-audio forced-alignment command:
  - `./build/q3asr-cli --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/long-audio.wav --align-text-file testdata/long-audio-result.txt > /tmp/q3asr-long-align.tsv`
- Observed for the 15-minute fixture:
  - runtime: about `147.69s`
  - output lines: `3002`
  - monotonic timestamps across the whole file
  - last end time: `911.566`
- Spot-checked chunk-boundary neighborhoods around roughly `180s`, `360s`, `540s`, and `720s` and confirmed that the merged alignment stays ordered and text remains plausible through those boundaries.

### Remaining Work

- Replace duration-proportional normalized-unit assignment with a better ownership policy once raw display-text stitching matters.
- Build the future transcription overlap merger on top of these aligned units and absolute timestamps.
- Compare long-audio boundary quality more directly against the Python `return_time_stamps=True` path once a matching local chunked-ASR experiment is worth the runtime.
- Add a dedicated helper or script if repeated long-audio alignment runs against `testdata/long-audio.wav` become common.

### Gotchas

- This implementation follows the Python timestamping reference at the audio-chunk level, not at the transcript-chunk source level:
  - Python gets per-chunk text from chunked ASR first
  - this repo currently assigns chunk text spans over normalized aligner units by cumulative duration
- That means the path is good enough for large-file timestamp experiments, but it is not yet the final punctuation-preserving merge policy for future transcription stitching.
- A stale cached `Q3ASR_TEST_AUDIO_ENGLISH` path can still point at an old root-level `q3asr-input.wav`; the CMake fallback recovers to `testdata/q3asr-input.wav` automatically when that happens.

### Follow-Up Goal

Use the forced aligner inside the transcription pipeline itself so long audio can be decoded in bounded windows while still producing one continuous transcript.

### Follow-Up Changes

- Extended `q3asr_transcribe_params` with:
  - `aligner_context`
  - `max_audio_chunk_seconds`
  - `audio_chunk_overlap_seconds`
- Reused the loaded forced aligner from transcription calls instead of loading a second hidden runtime.
- Added a long-audio transcription stitcher in `src/q3asr.cpp`:
  - split the full audio into Python-style low-energy core chunks
  - expand each chunk into an overlapped decode window
  - transcribe each window with the normal decoder path
  - force-align the decoded text back onto the same window audio
  - keep only the aligner-owned middle span for the chunk core
  - map the owned normalized units back to raw text spans and append them into one merged transcript
- Kept the old one-shot transcription path unchanged for short audio and for requests without an aligner context.
- Extended `q3asr-cli` so transcription mode can now also take:
  - `--aligner-model`
  - `--audio-chunk-sec`
- Added a real-model chunked transcription regression:
  - `q3asr-long-transcribe-regression`

### Follow-Up Validation

- Rebuilt and reconfigured successfully:
  - `cmake -S . -B build`
  - `cmake --build build -j`
- Full test suite passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `7/7` passing
- Manual short-sample chunked transcription check:
  - `./build/q3asr-smoke-test --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/q3asr-input.wav --audio-chunk-sec 6 --expect-language English --expect-substring /r/apple --capture-stream --expect-stream-calls-at-least 2 --expect-stream-equals-raw`
  - observed: one continuous final raw string with the `/r/apple` phrase preserved
- Manual 15-minute end-to-end transcription run:
  - `./build/q3asr-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/long-audio.wav --audio-chunk-sec 180 --max-tokens 1024 --show-raw > /tmp/q3asr-long-transcribe.txt`
- Observed for the 15-minute fixture:
  - runtime: about `251.02s`
  - peak RSS: about `3.47 GB`
  - exactly one `language English<asr_text>` prefix in the final output
  - one continuous `text:` block instead of per-chunk resets
  - expected content like `StatSig` and `I downloaded Xcode` remained present

### Follow-Up Remaining Work

- Improve the raw-text span recovery so punctuation-heavy tails and edge tokens survive more reliably.
- Decide whether chunked long-audio transcription should auto-scale `max_tokens` instead of relying on the caller to raise it for dense speech.
- Compare boundary quality against a Python-side chunked ASR + forced-align experiment, not just against the captured long transcript.
- Revisit whether chunk overlap should become user-configurable on the public CLI/API once the merge policy settles.

### Follow-Up Gotchas

- The long-audio transcription merge is continuous, but it is still heuristic:
  - it stayed close to the known 15-minute reference transcript
  - it still drifted at the tail on this run (`600.` vs. the reference `600 bucks.`)
- The current callback behavior differs by mode:
  - one-shot decoding still streams token-by-token
  - chunked long-audio mode currently emits cumulative committed text after each merged chunk

### Boundary Confidence Follow-Up

- Threaded decoder token logprobs through the internal transcription path:
  - `src/decoder_llama.cpp` now records one byte span plus chosen-token logprob for each generated token
  - `src/q3asr.cpp` now trims those spans down to the parsed transcript text after removing any `language ...<asr_text>` prefix
- Reworked the long-audio stitcher in `src/q3asr.cpp`:
  - keep the existing low-energy core chunking and overlapped decode windows
  - reserve a small overlap band around each chunk boundary
  - extract that same absolute-time band from the left and right neighboring windows
  - score each candidate band by average decoder logprob over the recovered raw-text byte span
  - let the higher-confidence side own the boundary band before appending the stable middle spans
- Kept the public CLI/API surface unchanged:
  - no new flags were added
  - confidence is currently internal merge metadata, not a public result field yet

### Boundary Confidence Validation

- Rebuilt successfully:
  - `cmake --build build -j`
- Full suite still passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `7/7` passing
- Re-ran the 15-minute transcription spot check:
  - `./build/q3asr-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/long-audio.wav --audio-chunk-sec 180 --max-tokens 1024 --show-raw > /tmp/q3asr-long-transcribe-new.txt`
  - observed runtime: about `249s`
  - observed result: still exactly one continuous `language English<asr_text>` block
  - observed tail: still ends at `600.` instead of the reference `600 bucks.`

### Boundary Confidence Remaining Work

- The new boundary arbitration is a better mechanism than plain midpoint ownership, but it is not enough by itself to get reference-level long-audio parity.
- If long-audio quality remains the priority, the next likely improvements are:
  - expose per-token confidence publicly for debugging
  - compare left/right overlap candidates with a stronger span similarity check, not just mean logprob plus prefix heuristics
  - evaluate chunked output directly against a Python-side chunked-ASR experiment instead of only against the captured long transcript

### Temperature Follow-Up

- Exposed decoder temperature through the public transcription params and the CLI:
  - `q3asr_transcribe_params.temperature`
  - `decoder_transcribe_params.temperature`
  - `q3asr-cli --temp <value>`
- Kept the default at `0.0`, which preserves the previous behavior exactly:
  - greedy argmax decode
  - matches the current reference-style usage in this repo and the sibling patched `llama.cpp` command lines

### Chunked Streaming Follow-Up

- Added a richer progress callback to the public transcription params:
  - `q3asr_progress_callback`
  - `q3asr_transcribe_params.progress_callback`
  - `q3asr_transcribe_params.progress_callback_user_data`
- The callback reports:
  - `language`
  - `committed_text`
  - `partial_text`
  - `chunk_index`
  - `chunk_count`
- Chose the `committed + partial` model over an append-only replacement stream because the long-audio path only knows stable ownership after decode, align, and boundary arbitration:
  - `committed_text` is monotonic and safe to keep
  - `partial_text` is the current decode window preview and can change on the next chunk boundary
- Added a dedicated console-first CLI:
  - `q3asr-chunk-stream-cli`
  - it renders progress on stderr and prints the final transcript on stdout
- Kept the existing `q3asr-cli` behavior unchanged:
  - `--stream-raw` is still the raw append-only path
  - the new committed/partial stream is separate on purpose

### Chunked Streaming Validation

- Rebuilt successfully:
  - `cmake --build build -j`
- Full suite still passes:
  - `ctest --test-dir build --output-on-failure`
  - observed: `7/7` passing
- Extended `q3asr-long-transcribe-regression` to cover the new callback:
  - it now captures both raw streaming and committed/partial progress streaming
  - it checks that progress callbacks fire, that at least one non-empty partial was seen, and that the final committed text matches the final parsed transcript
- Manual short-sample console run:
  - `./build/q3asr-chunk-stream-cli --text-model models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf --mmproj-model models/gguf/Qwen3-ASR-1.7B-mmproj.gguf --aligner-model models/gguf/qwen3-forced-aligner-0.6b-f16.gguf --audio testdata/q3asr-input.wav --audio-chunk-sec 6`
  - observed: the console now shows a stable `committed` section plus a provisional `partial` section instead of trying to fake an append-only token stream through chunk boundaries

### Chunked Streaming Gotchas

- The provisional `partial_text` stream is still a heuristic preview:
  - it strips the leading `language ...` prefix before `<asr_text>` is complete
  - it also removes a simple suffix/prefix overlap against already committed text for readability
- This is a console usability layer, not a stable machine-merge protocol yet.

### Overlap Flag Follow-Up

- Exposed the long-audio decode-window overlap on the CLI/test surfaces:
  - `q3asr-cli --audio-overlap-sec <sec>`
  - `q3asr-chunk-stream-cli --audio-overlap-sec <sec>`
  - `q3asr-smoke-test --audio-overlap-sec <sec>`
- Kept the default unchanged:
  - overlap defaults to `5.0s` whenever chunking is enabled and the caller does not override it
- The current boundary-arbitration band is still derived from the overlap inside `src/q3asr.cpp`:
  - `min(overlap, max(0.75, overlap * 0.5))`
  - so the current default `5.0s` overlap yields a `2.5s` confidence-arbitration band

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

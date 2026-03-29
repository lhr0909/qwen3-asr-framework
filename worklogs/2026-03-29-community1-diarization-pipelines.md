# 2026-03-29 Community-1 Diarization Pipelines

## Context

- The repo already contains:
  - a C++ GGUF loader for diarization model packaging
  - a C++ port of the `diart` online clustering and delayed aggregation logic
- The repo does not yet execute diarization neural models in C++/ggml.
- Local validation on the two long podcast cuts showed `pyannote/speaker-diarization-community-1` is materially more self-consistent than `pyannote/speaker-diarization-3.1`.

## References

- `docs/worklog.md`
- `models/hf/diarization/pyannote-speaker-diarization-community-1/config.yaml`
- `models/hf/diarization/pyannote-speaker-diarization-3.1/config.yaml`
- https://github.com/juanmc2005/diart
- https://huggingface.co/pyannote/speaker-diarization-community-1

## Plan

- [x] Add a Python reference diarization script that supports Community-1 offline diarization.
- [x] Add a lower-latency streaming approximation that is diart-inspired:
  - rolling windows
  - overlapping chunks
  - online speaker remapping by centroid similarity
  - stable center-strip commit behavior
- [x] Document the reference workflows in the repo docs.
- [x] Validate both modes on the local long-audio fixtures and record the outcome.

## Notes

- Added `scripts/pyannote_diarization.py` with:
  - `offline-community1`
  - `streaming-community1`
- Kept the implementation Python-only so we can validate the model/pipeline behavior without pretending the C++ runtime already executes Community-1 models.
- The streaming mode uses chunk-local Community-1 diarization and online speaker remapping between chunks, rather than trying to force the full offline VBx path into a low-latency loop.

## Validation

- Script shape and syntax:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py --help`
  - `python3 -m py_compile scripts/pyannote_diarization.py`
- Offline Community-1:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py offline-community1 --audio testdata/long-audio.wav --num-speakers 2 --output-json /tmp/q3asr-offline-community1-long-audio.json`
    - observed: `speakers=2`, `segments=410`, `exclusive_segments=419`
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py offline-community1 --audio testdata/long-audio-2.wav --num-speakers 2 --output-json /tmp/q3asr-offline-community1-long-audio-2.json`
    - observed: `speakers=2`, `segments=444`, `exclusive_segments=444`
- Streaming approximation:
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio /tmp/long-audio-60.wav --num-speakers 2 --window-sec 20 --step-sec 5 --output-json /tmp/q3asr-streaming-community1-60s.json --verbose`
    - observed: `chunks=9`, `global_speakers=2`, `segments=21`
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio testdata/long-audio.wav --num-speakers 2 --output-json /tmp/q3asr-streaming-community1-long-audio.json`
    - observed: `chunks=180`, `global_speakers=2`, `segments=611`
  - `/tmp/pyannote-audio-4/bin/python scripts/pyannote_diarization.py streaming-community1 --audio testdata/long-audio-2.wav --num-speakers 2 --output-json /tmp/q3asr-streaming-community1-long-audio-2.json`
    - observed: `chunks=197`, `global_speakers=2`, `segments=724`

## Follow-up

- Use this script as the quality/reference layer while implementing the matching C++ runtime.
- Reuse the validated streaming structure:
  - rolling windows
  - center-strip commit
  - online speaker remapping
- Treat the offline Community-1 path as the refinement reference, especially because it includes VBx + PLDA.

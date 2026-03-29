#!/usr/bin/env python3
"""Reference pyannote diarization helpers for Community-1.

This script provides two practical modes while the C++ runtime does not yet
execute diarization models directly:

1. `offline-community1`
   Runs the full offline `pyannote/speaker-diarization-community-1` pipeline,
   including its bundled VBx/PLDA clustering stage.

2. `streaming-community1`
   Runs an approximate streaming setup inspired by `diart`:
   - overlapping rolling windows
   - chunk-local diarization on each window
   - online speaker remapping using chunk speaker embeddings
   - stable center-strip commit behavior to reduce edge churn

The streaming mode intentionally trades some quality for latency and simplicity.
It is a reference path for experimentation, not a claim of exact offline parity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_DIR = REPO_ROOT / "models/hf/diarization/pyannote-speaker-diarization-community-1"


def patch_torch_load() -> None:
    import torch

    real_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return real_torch_load(*args, **kwargs)

    torch.load = patched_torch_load


def import_runtime():
    try:
        import torch
        import torchaudio

        patch_torch_load()
        from pyannote.audio import Pipeline
    except ModuleNotFoundError as exc:  # pragma: no cover - user environment guard
        raise SystemExit(
            "Missing diarization runtime dependency. Install with something like:\n"
            "  uv venv --python 3.12 /tmp/pyannote-audio-4\n"
            "  uv pip install --python /tmp/pyannote-audio-4/bin/python pyannote.audio==4.0.0"
        ) from exc

    return torch, torchaudio, Pipeline


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


@dataclass
class GlobalSpeakerState:
    speaker: str
    centroid: np.ndarray
    updates: int = 1

    def as_json(self) -> dict:
        return {
            "speaker": self.speaker,
            "updates": self.updates,
            "centroid": self.centroid.tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser("offline-community1", help="Run the full offline Community-1 VBx pipeline")
    configure_common_args(offline)

    streaming = subparsers.add_parser(
        "streaming-community1",
        help="Run a diart-inspired rolling-window streaming approximation using Community-1",
    )
    configure_common_args(streaming)
    streaming.add_argument("--window-sec", type=float, default=20.0, help="Rolling window duration in seconds")
    streaming.add_argument("--step-sec", type=float, default=5.0, help="Chunk step size in seconds")
    streaming.add_argument(
        "--speaker-match-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for reusing an existing global speaker",
    )
    streaming.add_argument(
        "--speaker-update-alpha",
        type=float,
        default=0.3,
        help="Exponential update rate for global speaker centroids",
    )
    streaming.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of global speakers to keep in the streaming approximation",
    )
    streaming.add_argument("--verbose", action="store_true", help="Print per-window progress")

    return parser.parse_args()


def configure_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument(
        "--pipeline-dir",
        default=str(DEFAULT_PIPELINE_DIR),
        help="Local pyannote speaker-diarization-community-1 directory",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Execution device",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Force the number of speakers when the pipeline supports it",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON output path")
    parser.add_argument("--output-rttm", default=None, help="Optional RTTM output path")


def resolve_device(torch, device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested MPS device, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested CUDA device, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0.0:
        return vec
    return vec / norm


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.dot(normalize(lhs), normalize(rhs)))


def load_pipeline(Pipeline, pipeline_dir: Path, device) -> object:
    pipeline = Pipeline.from_pretrained(pipeline_dir)
    pipeline.to(device)
    return pipeline


def annotation_to_segments(annotation, absolute_offset: float = 0.0) -> list[Segment]:
    segments: list[Segment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            Segment(
                start=float(turn.start + absolute_offset),
                end=float(turn.end + absolute_offset),
                speaker=str(speaker),
            )
        )
    return segments


def clip_segments(segments: Iterable[Segment], clip_start: float, clip_end: float, label_map: dict[str, str]) -> list[Segment]:
    clipped: list[Segment] = []
    for segment in segments:
        start = max(segment.start, clip_start)
        end = min(segment.end, clip_end)
        if end - start <= 1.0e-6:
            continue
        clipped.append(Segment(start=start, end=end, speaker=label_map.get(segment.speaker, segment.speaker)))
    return clipped


def merge_adjacent_segments(segments: list[Segment], gap_epsilon: float = 1.0e-3) -> list[Segment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))
    merged: list[Segment] = [ordered[0]]
    for segment in ordered[1:]:
        prev = merged[-1]
        if segment.speaker == prev.speaker and segment.start <= prev.end + gap_epsilon:
            prev.end = max(prev.end, segment.end)
            continue
        merged.append(segment)
    return merged


def write_rttm(path: Path, uri: str, segments: list[Segment]) -> None:
    lines = []
    for segment in segments:
        duration = max(0.0, segment.end - segment.start)
        lines.append(
            "SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>".format(
                uri=uri,
                start=segment.start,
                duration=duration,
                speaker=segment.speaker,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def dump_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def audio_duration_seconds(waveform, sample_rate: int) -> float:
    return float(waveform.shape[-1]) / float(sample_rate)


def load_audio_for_streaming(torchaudio, path: Path):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise SystemExit(f"Expected loaded waveform to be 2D, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def make_chunk_starts(total_duration: float, window_sec: float, step_sec: float) -> list[float]:
    if total_duration <= window_sec:
        return [0.0]

    starts: list[float] = []
    current = 0.0
    last_start = max(0.0, total_duration - window_sec)
    epsilon = 1.0e-6
    while current < last_start - epsilon:
        starts.append(current)
        current += step_sec
    if not starts or abs(starts[-1] - last_start) > epsilon:
        starts.append(last_start)
    return starts


def emit_interval_for_chunk(
    chunk_index: int,
    chunk_starts: list[float],
    window_sec: float,
    step_sec: float,
    total_duration: float,
) -> tuple[float, float]:
    if len(chunk_starts) == 1:
        return 0.0, total_duration

    center_margin = max(0.0, (window_sec - step_sec) * 0.5)
    chunk_start = chunk_starts[chunk_index]
    center_start = chunk_start + center_margin
    center_end = min(total_duration, center_start + step_sec)

    if chunk_index == 0:
        return 0.0, center_end
    if chunk_index == len(chunk_starts) - 1:
        return center_start, total_duration
    return center_start, center_end


def assign_streaming_speakers(
    labels: list[str],
    embeddings: np.ndarray | None,
    global_speakers: list[GlobalSpeakerState],
    similarity_threshold: float,
    update_alpha: float,
    max_speakers: int,
) -> dict[str, str]:
    if embeddings is None or embeddings.shape[0] != len(labels):
        return {label: label for label in labels}

    normalized_embeddings = [normalize(np.asarray(embedding, dtype=np.float32)) for embedding in embeddings]

    assignments: dict[int, int] = {}
    used_global_indices: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for local_index, embedding in enumerate(normalized_embeddings):
        for global_index, state in enumerate(global_speakers):
            candidates.append((cosine_similarity(embedding, state.centroid), local_index, global_index))
    candidates.sort(reverse=True)

    for similarity, local_index, global_index in candidates:
        if similarity < similarity_threshold:
            continue
        if local_index in assignments or global_index in used_global_indices:
            continue
        assignments[local_index] = global_index
        used_global_indices.add(global_index)

    for local_index, embedding in enumerate(normalized_embeddings):
        if local_index in assignments:
            continue

        if len(global_speakers) < max_speakers:
            global_index = len(global_speakers)
            global_speakers.append(
                GlobalSpeakerState(
                    speaker=f"SPEAKER_{global_index:02d}",
                    centroid=embedding.copy(),
                )
            )
            assignments[local_index] = global_index
            used_global_indices.add(global_index)
            continue

        remaining = [index for index in range(len(global_speakers)) if index not in used_global_indices]
        if not remaining:
            remaining = list(range(len(global_speakers)))
        best_global = max(remaining, key=lambda index: cosine_similarity(embedding, global_speakers[index].centroid))
        assignments[local_index] = best_global
        used_global_indices.add(best_global)

    label_map: dict[str, str] = {}
    for local_index, label in enumerate(labels):
        global_index = assignments[local_index]
        state = global_speakers[global_index]
        label_map[label] = state.speaker
        state.centroid = normalize((1.0 - update_alpha) * state.centroid + update_alpha * normalized_embeddings[local_index])
        state.updates += 1

    return label_map


def run_offline(args: argparse.Namespace) -> dict:
    torch, _, Pipeline = import_runtime()
    device = resolve_device(torch, args.device)
    audio_path = Path(args.audio)
    pipeline_dir = Path(args.pipeline_dir)
    pipeline = load_pipeline(Pipeline, pipeline_dir, device)

    kwargs = {}
    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers
    output = pipeline(str(audio_path), **kwargs)

    speaker_segments = annotation_to_segments(output.speaker_diarization)
    exclusive_segments = annotation_to_segments(output.exclusive_speaker_diarization)
    labels = list(output.speaker_diarization.labels())
    speaker_embeddings = []
    if output.speaker_embeddings is not None:
        for speaker, embedding in zip(labels, np.asarray(output.speaker_embeddings, dtype=np.float32)):
            speaker_embeddings.append(
                {
                    "speaker": speaker,
                    "embedding": embedding.tolist(),
                    "norm": float(np.linalg.norm(embedding)),
                }
            )

    result = {
        "mode": "offline-community1",
        "pipeline": str(pipeline_dir),
        "audio": str(audio_path),
        "device": str(device),
        "num_speakers": args.num_speakers,
        "speaker_count": len(labels),
        "speaker_diarization": [asdict(segment) for segment in speaker_segments],
        "exclusive_speaker_diarization": [asdict(segment) for segment in exclusive_segments],
        "speaker_embeddings": speaker_embeddings,
    }

    dump_json(Path(args.output_json) if args.output_json else None, result)
    if args.output_rttm:
        write_rttm(Path(args.output_rttm), audio_path.stem, speaker_segments)

    print(
        "offline-community1 complete:",
        f"device={device}",
        f"speakers={len(labels)}",
        f"segments={len(speaker_segments)}",
        f"exclusive_segments={len(exclusive_segments)}",
    )
    return result


def run_streaming(args: argparse.Namespace) -> dict:
    torch, torchaudio, Pipeline = import_runtime()
    device = resolve_device(torch, args.device)
    audio_path = Path(args.audio)
    pipeline_dir = Path(args.pipeline_dir)
    pipeline = load_pipeline(Pipeline, pipeline_dir, device)

    waveform, sample_rate = load_audio_for_streaming(torchaudio, audio_path)
    total_duration = audio_duration_seconds(waveform, sample_rate)
    chunk_starts = make_chunk_starts(total_duration, args.window_sec, args.step_sec)
    max_speakers = args.max_speakers or args.num_speakers or 20

    global_speakers: list[GlobalSpeakerState] = []
    committed_segments: list[Segment] = []
    chunk_summaries: list[dict] = []

    for chunk_index, chunk_start in enumerate(chunk_starts):
        chunk_end = min(total_duration, chunk_start + args.window_sec)
        sample_start = int(round(chunk_start * sample_rate))
        sample_end = int(round(chunk_end * sample_rate))
        chunk_waveform = waveform[:, sample_start:sample_end]

        kwargs = {}
        if args.num_speakers is not None:
            kwargs["num_speakers"] = args.num_speakers

        output = pipeline(
            {
                "waveform": chunk_waveform,
                "sample_rate": sample_rate,
            },
            **kwargs,
        )

        local_labels = list(output.speaker_diarization.labels())
        label_map = assign_streaming_speakers(
            local_labels,
            None if output.speaker_embeddings is None else np.asarray(output.speaker_embeddings, dtype=np.float32),
            global_speakers,
            args.speaker_match_threshold,
            args.speaker_update_alpha,
            max_speakers,
        )

        emit_start, emit_end = emit_interval_for_chunk(
            chunk_index,
            chunk_starts,
            args.window_sec,
            args.step_sec,
            total_duration,
        )
        local_segments = annotation_to_segments(output.exclusive_speaker_diarization, absolute_offset=chunk_start)
        committed_segments.extend(clip_segments(local_segments, emit_start, emit_end, label_map))

        if args.verbose:
            print(
                "stream-window",
                f"{chunk_index + 1}/{len(chunk_starts)}",
                f"chunk={chunk_start:.2f}-{chunk_end:.2f}",
                f"emit={emit_start:.2f}-{emit_end:.2f}",
                f"local_labels={local_labels}",
                f"global_labels={sorted(set(label_map.values()))}",
            )

        chunk_summaries.append(
            {
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "emit_start": emit_start,
                "emit_end": emit_end,
                "local_labels": local_labels,
                "label_map": label_map,
                "speaker_embedding_count": 0 if output.speaker_embeddings is None else int(output.speaker_embeddings.shape[0]),
            }
        )

    committed_segments = merge_adjacent_segments(committed_segments)
    result = {
        "mode": "streaming-community1",
        "pipeline": str(pipeline_dir),
        "audio": str(audio_path),
        "device": str(device),
        "num_speakers": args.num_speakers,
        "window_seconds": args.window_sec,
        "step_seconds": args.step_sec,
        "speaker_match_threshold": args.speaker_match_threshold,
        "speaker_update_alpha": args.speaker_update_alpha,
        "max_speakers": max_speakers,
        "duration_seconds": total_duration,
        "chunk_count": len(chunk_starts),
        "segments": [asdict(segment) for segment in committed_segments],
        "global_speakers": [state.as_json() for state in global_speakers],
        "chunk_summaries": chunk_summaries,
    }

    dump_json(Path(args.output_json) if args.output_json else None, result)
    if args.output_rttm:
        write_rttm(Path(args.output_rttm), audio_path.stem, committed_segments)

    print(
        "streaming-community1 complete:",
        f"device={device}",
        f"chunks={len(chunk_starts)}",
        f"global_speakers={len(global_speakers)}",
        f"segments={len(committed_segments)}",
    )
    return result


def main() -> int:
    args = parse_args()
    if args.command == "streaming-community1":
        if args.window_sec <= 0.0:
            raise SystemExit("--window-sec must be positive")
        if args.step_sec <= 0.0:
            raise SystemExit("--step-sec must be positive")
        if args.step_sec > args.window_sec:
            raise SystemExit("--step-sec must be less than or equal to --window-sec")
        if args.max_speakers is not None and args.max_speakers <= 0:
            raise SystemExit("--max-speakers must be positive")
    if args.command == "offline-community1":
        run_offline(args)
        return 0
    if args.command == "streaming-community1":
        run_streaming(args)
        return 0
    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

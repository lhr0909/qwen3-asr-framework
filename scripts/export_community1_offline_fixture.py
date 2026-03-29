#!/usr/bin/env python3
"""Export a short Community-1 offline clustering fixture for C++ regression tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_DIR = REPO_ROOT / "models/hf/diarization/pyannote-speaker-diarization-community-1"


def patch_torch_load(torch) -> None:
    real_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return real_torch_load(*args, **kwargs)

    torch.load = patched_torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output", required=True, help="Fixture JSON path")
    parser.add_argument("--pipeline-dir", default=str(DEFAULT_PIPELINE_DIR))
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--num-speakers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.audio.utils.signal import binarize

    patch_torch_load(torch)

    pipeline = Pipeline.from_pretrained(Path(args.pipeline_dir))
    pipeline.to(torch.device("cpu"))

    waveform, sample_rate = torchaudio.load(args.audio)
    if waveform.ndim != 2:
        raise SystemExit(f"Expected a 2D waveform tensor, got shape {tuple(waveform.shape)}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    sample_start = int(round(args.start_sec * sample_rate))
    sample_end = int(round((args.start_sec + args.duration_sec) * sample_rate))
    waveform = waveform[:, sample_start:sample_end]

    file = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": f"{Path(args.audio).name}:{args.start_sec:.3f}-{args.start_sec + args.duration_sec:.3f}",
    }

    segmentations = pipeline.get_segmentations(file)
    if pipeline._segmentation.model.specifications.powerset:
        binary_segmentations = segmentations
    else:
        binary_segmentations = binarize(
            segmentations,
            onset=pipeline.segmentation.threshold,
            initial_state=False,
        )

    embeddings = pipeline.get_embeddings(
        file,
        binary_segmentations,
        exclude_overlap=pipeline.embedding_exclude_overlap,
    )
    hard_clusters, soft_clusters, centroids = pipeline.clustering(
        embeddings=embeddings,
        segmentations=binary_segmentations,
        num_clusters=args.num_speakers,
        min_clusters=args.num_speakers,
        max_clusters=args.num_speakers,
        file=file,
        frames=pipeline._segmentation.model.receptive_field,
    )

    payload = {
        "audio": str(args.audio),
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "num_chunks": int(binary_segmentations.data.shape[0]),
        "num_frames": int(binary_segmentations.data.shape[1]),
        "num_speakers": int(binary_segmentations.data.shape[2]),
        "embedding_dim": int(embeddings.shape[2]),
        "num_clusters": int(args.num_speakers),
        "binary_segmentations": np.asarray(binary_segmentations.data, dtype=np.float32).reshape(-1).tolist(),
        "embeddings": np.asarray(embeddings, dtype=np.float32).reshape(-1).tolist(),
        "expected_hard_clusters": np.asarray(hard_clusters, dtype=np.int32).reshape(-1).tolist(),
        "expected_centroids": np.asarray(centroids, dtype=np.float32).reshape(-1).tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    print(f"num_chunks={payload['num_chunks']}")
    print(f"num_frames={payload['num_frames']}")
    print(f"num_speakers={payload['num_speakers']}")
    print(f"embedding_dim={payload['embedding_dim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

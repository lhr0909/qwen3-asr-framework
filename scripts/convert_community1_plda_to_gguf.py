#!/usr/bin/env python3
"""Convert Community-1 PLDA assets into a GGUF container for C++ runtime use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp/gguf-py"))

try:
    from gguf import GGUFWriter  # type: ignore  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name == "yaml":
        raise SystemExit(
            "PyYAML is required by llama.cpp's local gguf-py package. "
            "Install it or run via `uvx --with numpy,scipy,pyyaml python ...`."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, help="Local pyannote/speaker-diarization-community-1 directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--source-repo", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--name", default="pyannote-speaker-diarization-community-1-plda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = np.load(bundle_dir / "plda" / "xvec_transform.npz")
    plda = np.load(bundle_dir / "plda" / "plda.npz")

    mean1 = np.asarray(transform["mean1"], dtype=np.float32)
    mean2 = np.asarray(transform["mean2"], dtype=np.float32)
    lda = np.asarray(transform["lda"], dtype=np.float32)

    plda_mu = np.asarray(plda["mu"], dtype=np.float32)
    plda_tr = np.asarray(plda["tr"], dtype=np.float64)
    plda_psi = np.asarray(plda["psi"], dtype=np.float64)

    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    eigenvalues, eigenvectors = eigh(B, W)
    processed_psi = np.asarray(eigenvalues[::-1], dtype=np.float32)
    processed_transform = np.asarray(eigenvectors.T[::-1], dtype=np.float32)

    writer = GGUFWriter(str(output_path), "q3asr-diarization")
    writer.add_string("general.name", args.name)
    writer.add_string("diarization.kind", "speaker-clustering")
    writer.add_string("diarization.serialization_format", "numpy")
    writer.add_string("diarization.source_repo", args.source_repo)
    writer.add_string("diarization.source_file", "plda/xvec_transform.npz + plda/plda.npz")
    writer.add_uint32("diarization.tensor_count", 6)

    writer.add_tensor("xvec.mean1", mean1)
    writer.add_tensor("xvec.mean2", mean2)
    writer.add_tensor("xvec.lda", lda)
    writer.add_tensor("plda.mu", plda_mu)
    writer.add_tensor("plda.transform", processed_transform)
    writer.add_tensor("plda.psi", processed_psi)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(output_path)
    print(f"xvec_input_dim={mean1.shape[0]}")
    print(f"xvec_output_dim={mean2.shape[0]}")
    print(f"plda_dim={processed_psi.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

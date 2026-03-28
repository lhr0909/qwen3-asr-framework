#!/usr/bin/env python3
"""Convert a diarization-oriented PyTorch checkpoint into a GGUF container.

This is a packaging step for bring-up and inspection. It preserves the PyTorch
state dict tensors and selected checkpoint/config metadata in GGUF so the C++
side can load and inspect the model through ggml/GGUF utilities. It does not
make the model directly executable by q3asr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp/gguf-py"))

try:
    from gguf import GGUFWriter  # type: ignore  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - user-environment guard
    if exc.name == "yaml":
        raise SystemExit(
            "PyYAML is required by llama.cpp's local gguf-py package. "
            "Install it or run via `uvx --with torch,numpy,pyyaml,torchaudio,pyannote.audio==3.1.1 python ...`."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Input checkpoint directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--kind", required=True, choices=("speaker-segmentation", "speaker-embedding"))
    parser.add_argument("--name", default=None, help="Logical model name to store in GGUF metadata")
    parser.add_argument("--source-repo", default="", help="Hugging Face repo id or other source identifier")
    parser.add_argument("--source-file", default="pytorch_model.bin", help="Original source filename inside the source repo")
    parser.add_argument("--checkpoint", default="pytorch_model.bin", help="Checkpoint filename inside the input directory")
    parser.add_argument("--config-json", default=None, help="Optional sidecar config.json path")
    parser.add_argument("--preprocessor-json", default=None, help="Optional sidecar preprocessor_config.json path")
    parser.add_argument("--config-yaml", default=None, help="Optional sidecar config.yaml path")
    parser.add_argument("--hparams-yaml", default=None, help="Optional sidecar hparams.yaml path")
    parser.add_argument("--hydra-yaml", default=None, help="Optional sidecar hydra.yaml path")
    parser.add_argument("--overrides-yaml", default=None, help="Optional sidecar overrides.yaml path")
    return parser.parse_args()


def discover_sidecar(path: Path, explicit: str | None, filename: str) -> str:
    if explicit:
        return Path(explicit).read_text(encoding="utf-8")

    candidate = path / filename
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return ""


def install_runtime_compatibility_shims() -> None:
    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]

    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda *args, **kwargs: "soundfile"  # type: ignore[attr-defined]


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    install_runtime_compatibility_shims()

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Expected a mapping checkpoint at {checkpoint_path}, got {type(checkpoint).__name__}")
    return checkpoint


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise SystemExit("Checkpoint does not contain a usable state dict")
    return state_dict


def json_string(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def tensor_to_numpy(name: str, value: Any) -> np.ndarray:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"State dict entry {name!r} is not a tensor: {type(value).__name__}")

    if value.dtype == torch.bfloat16:
        raise TypeError(f"State dict entry {name!r} uses unsupported dtype bfloat16")

    array = value.detach().cpu().contiguous().numpy()
    if array.dtype not in (
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ):
        raise TypeError(f"State dict entry {name!r} uses unsupported dtype {array.dtype}")
    return array


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    checkpoint_path = input_dir / args.checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)

    pyannote_metadata = checkpoint.get("pyannote.audio", {})
    if not isinstance(pyannote_metadata, dict):
        pyannote_metadata = {}
    versions = pyannote_metadata.get("versions", {})
    if not isinstance(versions, dict):
        versions = {}
    architecture = pyannote_metadata.get("architecture", {})
    if not isinstance(architecture, dict):
        architecture = {}

    writer = GGUFWriter(str(output_path), "q3asr-diarization")
    writer.add_string("general.name", args.name or input_dir.name)

    writer.add_string("diarization.kind", args.kind)
    writer.add_string("diarization.serialization_format", "pytorch")
    writer.add_string("diarization.source_repo", args.source_repo)
    writer.add_string("diarization.source_file", args.source_file or checkpoint_path.name)
    writer.add_uint32("diarization.tensor_count", len(state_dict))

    writer.add_uint32("diarization.pytorch.top_level_key_count", len(checkpoint))
    writer.add_string(
        "diarization.pytorch.lightning_version",
        str(checkpoint.get("pytorch-lightning_version", "")),
    )
    writer.add_string(
        "diarization.pytorch.hparams_name",
        str(checkpoint.get("hparams_name", "")),
    )
    writer.add_string(
        "diarization.pytorch.hyper_parameters_json",
        json_string(checkpoint.get("hyper_parameters", {})),
    )
    writer.add_string(
        "diarization.pytorch.pyannote_audio_metadata_json",
        json_string(pyannote_metadata),
    )
    writer.add_string(
        "diarization.pytorch.model_module",
        str(architecture.get("module", "")),
    )
    writer.add_string(
        "diarization.pytorch.model_class",
        str(architecture.get("class", "")),
    )
    writer.add_string(
        "diarization.pytorch.versions.torch",
        str(versions.get("torch", "")),
    )
    writer.add_string(
        "diarization.pytorch.versions.pyannote_audio",
        str(versions.get("pyannote.audio", "")),
    )
    writer.add_string(
        "diarization.pytorch.specifications_repr",
        repr(pyannote_metadata.get("specifications", "")),
    )

    writer.add_string("diarization.config_json", discover_sidecar(input_dir, args.config_json, "config.json"))
    writer.add_string(
        "diarization.preprocessor_json",
        discover_sidecar(input_dir, args.preprocessor_json, "preprocessor_config.json"),
    )
    writer.add_string("diarization.config_yaml", discover_sidecar(input_dir, args.config_yaml, "config.yaml"))
    writer.add_string("diarization.hparams_yaml", discover_sidecar(input_dir, args.hparams_yaml, "hparams.yaml"))
    writer.add_string("diarization.hydra_yaml", discover_sidecar(input_dir, args.hydra_yaml, "hydra.yaml"))
    writer.add_string(
        "diarization.overrides_yaml",
        discover_sidecar(input_dir, args.overrides_yaml, "overrides.yaml"),
    )

    for name, value in state_dict.items():
        writer.add_tensor(name, tensor_to_numpy(name, value))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(output_path)
    print(f"tensor_count={len(state_dict)}")
    print(f"top_level_keys={len(checkpoint)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

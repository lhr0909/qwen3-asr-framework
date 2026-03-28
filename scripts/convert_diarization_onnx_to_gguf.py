#!/usr/bin/env python3
"""Convert a diarization-oriented ONNX checkpoint into a GGUF container.

This is a packaging step for bring-up and inspection. It preserves the ONNX
graph metadata and initializer tensors in GGUF so the C++ side can load and
inspect the model through ggml/GGUF utilities. It does not make the graph
directly executable by q3asr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp/gguf-py"))

try:
    from gguf import GGUFWriter  # type: ignore  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - user-environment guard
    if exc.name == "yaml":
        raise SystemExit(
            "PyYAML is required by llama.cpp's local gguf-py package. "
            "Install it or run via `uvx --with onnx,numpy,pyyaml python ...`."
        ) from exc
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--kind", required=True, choices=("speaker-segmentation", "speaker-embedding"))
    parser.add_argument("--name", default=None, help="Logical model name to store in GGUF metadata")
    parser.add_argument("--source-repo", default="", help="Hugging Face repo id or other source identifier")
    parser.add_argument("--source-file", default="", help="Original source filename inside the source repo")
    parser.add_argument("--config-json", default=None, help="Optional sidecar config.json path")
    parser.add_argument("--preprocessor-json", default=None, help="Optional sidecar preprocessor_config.json path")
    return parser.parse_args()


def discover_sidecar(path: Path, explicit: str | None, filename: str) -> str:
    if explicit:
        return Path(explicit).read_text(encoding="utf-8")

    candidates = [
        path.parent / filename,
        path.parent.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return ""


def shape_to_string(value_info: onnx.ValueInfoProto) -> str:
    dims: list[str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return ",".join(dims)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(input_path)

    writer = GGUFWriter(str(output_path), "q3asr-diarization")
    writer.add_string("general.name", args.name or input_path.stem)

    writer.add_string("diarization.kind", args.kind)
    writer.add_string("diarization.source_repo", args.source_repo)
    writer.add_string("diarization.source_file", args.source_file or input_path.name)
    writer.add_uint32("diarization.onnx.ir_version", int(model.ir_version))
    writer.add_uint32("diarization.onnx.initializer_count", len(model.graph.initializer))
    writer.add_uint32("diarization.onnx.node_count", len(model.graph.node))

    writer.add_uint32("diarization.onnx.opset_count", len(model.opset_import))
    for idx, opset in enumerate(model.opset_import):
        writer.add_string(f"diarization.onnx.opset.{idx}.domain", opset.domain or "main")
        writer.add_uint32(f"diarization.onnx.opset.{idx}.version", int(opset.version))

    writer.add_uint32("diarization.onnx.input_count", len(model.graph.input))
    for idx, value_info in enumerate(model.graph.input):
        writer.add_string(f"diarization.onnx.input.{idx}.name", value_info.name)
        writer.add_string(f"diarization.onnx.input.{idx}.shape", shape_to_string(value_info))

    writer.add_uint32("diarization.onnx.output_count", len(model.graph.output))
    for idx, value_info in enumerate(model.graph.output):
        writer.add_string(f"diarization.onnx.output.{idx}.name", value_info.name)
        writer.add_string(f"diarization.onnx.output.{idx}.shape", shape_to_string(value_info))

    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    writer.add_uint32("diarization.onnx.op_count", len(op_counts))
    for idx, (name, count) in enumerate(sorted(op_counts.items())):
        writer.add_string(f"diarization.onnx.op.{idx}.name", name)
        writer.add_uint32(f"diarization.onnx.op.{idx}.count", count)

    config_json = discover_sidecar(input_path, args.config_json, "config.json")
    preprocessor_json = discover_sidecar(input_path, args.preprocessor_json, "preprocessor_config.json")
    writer.add_string("diarization.config_json", config_json)
    writer.add_string("diarization.preprocessor_json", preprocessor_json)

    for initializer in model.graph.initializer:
        tensor = np.asarray(numpy_helper.to_array(initializer))
        writer.add_tensor(initializer.name, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(output_path)
    print(f"initializers={len(model.graph.initializer)}")
    print(f"inputs={len(model.graph.input)} outputs={len(model.graph.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

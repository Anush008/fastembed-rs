"""
quantize_pixie.py — ONNX quantization for XLM-RoBERTa-family embedding models.

Produces three self-contained ONNX variants from a float32 source model:
  model_quantized.onnx   — INT8 dynamic (all weights including word embeddings)
  model_int4.onnx        — INT4 MatMul (MatMulNBits) + INT8 word embedding
  model_int4_full.onnx   — INT4 MatMul + INT4 word embedding (opset 21, smallest)

Usage:
    python quantize_pixie.py \\
        --input  onnx/model.onnx \\
        --outdir onnx/ \\
        [--block-size 32]

    # Or via environment variables:
    PIXIE_INPUT=onnx/model.onnx PIXIE_OUTDIR=onnx/ python quantize_pixie.py

The input model is expected to reside in the same directory as its companion
data file (model.onnx_data) when using the default HuggingFace layout.
"""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> onnx.ModelProto:
    """Load an ONNX model, handling both inline and external initializers."""
    model = onnx.load(str(path), load_external_data=False)
    data_file = path.with_suffix(".onnx_data")
    if data_file.exists():
        onnx.load_external_data_for_model(model, str(path.parent))
    return model


def _save_temp(model: onnx.ModelProto, path: Path) -> None:
    """Save a model to disk, inlining all tensors (needed before quantization)."""
    onnx.save(model, str(path))


def _find_gather_input_name(model: onnx.ModelProto) -> str | None:
    """Return the initializer name fed into the first Gather (word embedding) node."""
    for node in model.graph.node:
        if node.op_type == "Gather":
            return node.input[0]  # initializer with embedding weight
    return None


# ── INT8 dynamic quantization ─────────────────────────────────────────────────

def make_int8(src: Path, dst: Path) -> None:
    """
    INT8 dynamic quantization — all weight tensors (MatMul + Gather).

    Uses onnxruntime quantize_dynamic with QInt8.  The word embedding Gather
    is included, bringing the ~977 MB FP32 embedding table down to ~244 MB.
    """
    print(f"  INT8: {src.name} → {dst.name}")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"  INT8 done  ({dst.stat().st_size / 1024**2:.0f} MB)")


# ── INT4 MatMulNBits quantization ─────────────────────────────────────────────

def _apply_matmul_nbits(src_model: onnx.ModelProto, block_size: int) -> onnx.ModelProto:
    """Apply MatMulNBits (INT4) to all MatMul weight tensors."""
    import tempfile, copy
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_in = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_out = Path(f.name)
    try:
        _save_temp(src_model, tmp_in)
        q = MatMulNBitsQuantizer(
            str(tmp_in),
            block_size=block_size,
            is_symmetric=True,
            nodes_to_exclude=[],
        )
        q.process()
        q.model.save_model_to_file(str(tmp_out), use_external_data_format=False)
        return onnx.load(str(tmp_out))
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def make_int4_int8_emb(src: Path, dst: Path, block_size: int = 32) -> None:
    """
    Two-pass: INT4 MatMul (MatMulNBits) + INT8 word embedding.

    Pass 1 — MatMulNBitsQuantizer packs transformer MatMul weights to 4-bit.
    Pass 2 — quantize_dynamic(op_types=["Gather"], QInt8) quantizes the
             word embedding table (250,002 × 1024) from FP32 to INT8.
    """
    import tempfile
    print(f"  INT4+INT8 emb: {src.name} → {dst.name}")
    model = _load(src)

    # Pass 1: INT4 MatMul
    print("    Pass 1: MatMulNBits INT4 ...")
    matmul_model = _apply_matmul_nbits(model, block_size=block_size)

    # Pass 2: INT8 Gather (word embedding table only)
    print("    Pass 2: INT8 Gather ...")
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = Path(f.name)
    try:
        _save_temp(matmul_model, tmp)
        quantize_dynamic(
            str(tmp), str(dst),
            op_types_to_quantize=["Gather"],
            weight_type=QuantType.QInt8,
        )
    finally:
        tmp.unlink(missing_ok=True)
    print(f"  INT4+INT8 emb done  ({dst.stat().st_size / 1024**2:.0f} MB)")


# ── INT4 full (word embeddings packed as INT4 nibbles) ────────────────────────

def _pack_int4_rows(weight: np.ndarray) -> tuple[bytes, np.ndarray]:
    """
    Pack a 2-D float32 tensor as per-row symmetric INT4.

    Each row r is quantized with scale = max(|row_r|) / 7.
    Values are clamped to [-7, 7] and packed as two INT4 nibbles per byte
    (little-endian nibble order: low nibble = even index, high nibble = odd).

    Returns:
        packed_bytes  — raw bytes (vocab_size × ceil(dim/2))
        scales        — float32 scale per row (vocab_size,)
    """
    vocab, dim = weight.shape
    abs_max = np.abs(weight).max(axis=1, keepdims=True).clip(min=1e-9)
    scales = (abs_max / 7.0).squeeze(1).astype(np.float32)
    quantized = np.round(weight / abs_max * 7.0).clip(-7, 7).astype(np.int8)

    # Pack pairs of INT4 values into bytes
    # Treat negative as unsigned 4-bit: -7..7 → offset doesn't apply for symmetric
    # Use unsigned nibbles with zero_point=0 (symmetric)
    u4 = (quantized % 16).astype(np.uint8)  # map negatives: e.g. -1 → 15
    padded = u4 if dim % 2 == 0 else np.pad(u4, ((0, 0), (0, 1)))
    packed = padded[:, 0::2] | (padded[:, 1::2] << 4)
    return packed.tobytes(), scales


def make_int4_full(src: Path, dst: Path, block_size: int = 32) -> None:
    """
    INT4 full: INT4 MatMul (MatMulNBits) + INT4 word embedding (DequantizeLinear).

    The word embedding Gather is replaced by:
        INT4_packed_tensor → DequantizeLinear(axis=0, scale=per_row) → FP32 lookup
    Requires ONNX opset 21 for the INT4 DequantizeLinear kernel in OnnxRuntime.

    Build from the FP32 source (not from model_int4.onnx which already has an
    INT8 DequantizeLinear node on the Gather output, causing a type conflict).
    """
    import tempfile
    print(f"  INT4 full: {src.name} → {dst.name}")
    model = _load(src)

    # Step 1: INT4 MatMul
    print("    Step 1: MatMulNBits INT4 ...")
    matmul_model = _apply_matmul_nbits(model, block_size=block_size)

    # Step 2: Migrate to opset 21 (required for INT4 DequantizeLinear)
    print("    Step 2: Opset 14 → 21 ...")
    matmul_model = onnx.version_converter.convert_version(matmul_model, 21)

    # Step 3: Find and replace the Gather (word embedding) node
    print("    Step 3: INT4-pack word embedding table ...")
    graph = matmul_model.graph

    # Locate embedding initializer name
    embed_init_name = _find_gather_input_name(matmul_model)
    if embed_init_name is None:
        raise RuntimeError("Could not find Gather (word embedding) node in graph.")

    # Extract current FP32 embedding tensor
    embed_init = next(
        (init for init in graph.initializer if init.name == embed_init_name), None
    )
    if embed_init is None:
        raise RuntimeError(f"Initializer '{embed_init_name}' not found.")

    weight_fp32 = np.array(
        onnx.numpy_helper.to_array(embed_init), dtype=np.float32
    )
    packed_bytes, scales = _pack_int4_rows(weight_fp32)

    # Replace the FP32 initializer with packed INT4
    graph.initializer.remove(embed_init)
    int4_name = embed_init_name + "_int4"
    scales_name = embed_init_name + "_scales"

    # INT4 tensor stored as raw bytes in ONNX (UINT4 = elem_type 17)
    int4_tensor = onnx.TensorProto()
    int4_tensor.name = int4_name
    int4_tensor.data_type = 17  # UINT4
    int4_tensor.dims.extend(list(weight_fp32.shape))
    int4_tensor.raw_data = packed_bytes
    graph.initializer.append(int4_tensor)

    # Per-row scale tensor (float32)
    scales_tensor = onnx.numpy_helper.from_array(scales, name=scales_name)
    graph.initializer.append(scales_tensor)

    # Insert DequantizeLinear(axis=0) between INT4 weights and the Gather node
    dql_out_name = embed_init_name + "_dq"
    dql_node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=[int4_name, scales_name],
        outputs=[dql_out_name],
        axis=0,
    )

    # Reroute: Gather now reads from dql_out instead of original initializer
    for node in graph.node:
        if node.op_type == "Gather" and node.input[0] == embed_init_name:
            node.input[0] = dql_out_name

    # Insert DequantizeLinear before the Gather node
    gather_idx = next(
        i for i, n in enumerate(graph.node)
        if n.op_type == "Gather" and n.input[0] == dql_out_name
    )
    graph.node.insert(gather_idx, dql_node)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = Path(f.name)
    try:
        onnx.save(matmul_model, str(tmp))
        onnx.checker.check_model(str(tmp))
        import shutil
        shutil.copy(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"  INT4 full done  ({dst.stat().st_size / 1024**2:.0f} MB)")


# ── entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  default=os.environ.get("PIXIE_INPUT"),
                   help="Path to the FP32 source model.onnx (may have companion .onnx_data)")
    p.add_argument("--outdir", default=os.environ.get("PIXIE_OUTDIR", "."),
                   help="Output directory for quantized models (default: cwd)")
    p.add_argument("--block-size", type=int, default=32,
                   help="Block size for MatMulNBits INT4 (default: 32)")
    p.add_argument("--variants", nargs="+",
                   choices=["int8", "int4", "int4_full", "all"],
                   default=["all"],
                   help="Which variants to produce (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input:
        raise SystemExit("Error: --input or PIXIE_INPUT env var required.")

    src = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    variants = set(args.variants)
    if "all" in variants:
        variants = {"int8", "int4", "int4_full"}

    print(f"Source : {src}")
    print(f"Out dir: {outdir}")
    print(f"Targets: {', '.join(sorted(variants))}")
    print()

    if "int8" in variants:
        make_int8(src, outdir / "model_quantized.onnx")

    if "int4" in variants:
        make_int4_int8_emb(src, outdir / "model_int4.onnx",
                           block_size=args.block_size)

    if "int4_full" in variants:
        make_int4_full(src, outdir / "model_int4_full.onnx",
                       block_size=args.block_size)

    print("\nAll done.")


if __name__ == "__main__":
    main()

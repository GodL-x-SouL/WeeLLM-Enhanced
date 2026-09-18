"""
comfy_quant.py -- Pure-PyTorch dequantization for ComfyUI-style quantized
safetensors (INT8 tensorwise/ConvRot + W4A8 ConvRot).

Why this exists
--------------
ComfyUI/Comfy-Kitchen quantized checkpoints (e.g. Winnougan LTX-2.5 W4A8,
official LTX-2.5 INT8-ConvRot) store some layers as packed integers plus
scale tensors, mixed in ONE file with full-precision (BF16/F32) layers for
the sensitive parts (norms, projections, scale_shift_table, ...).

WeeLLM streams plain floating tensors block-by-block, so this module
decodes each quantized group back to a floating tensor at load time using
only stock PyTorch ops -- no comfy-kitchen install, no custom CUDA kernels,
no driver floors. The math is ported from comfy-kit's eager (pure-PyTorch)
backend, so ``dequantize(quantize(w)) ~= w`` exactly as in ComfyUI's own
emulated/dequant fallback path.

Speed note (honest): on GPUs where a vendor INT8 kernel is native (T4 for
INT8), a kernel path would win per-matmul. But WeeLLM reloads every
block on every denoising step, and the Comfy W4A8 path is *emulated*
(dequant-then-compute) on a T4 anyway (needs SM>=8.0 for native). So this
module matches the best speed available on a T4 while working everywhere,
including CPU-only machines. A kernel fast-path can plug in later behind
the same seeker interface.

On-disk layouts supported
--------------------------
W4A8 (``asym_w4a8_int8``), per quantized ``<prefix>`` (a Linear layer)::

    <prefix>.weight            int8  [N, K/2]   (two int4 codes per byte)
    <prefix>.weight_s_rel      fp8   [N, K/16]  (per-group scales, group_size=16)
    <prefix>.weight_s_channel  fp32  [N]        (per-output-channel scales)
    <prefix>.weight_codebook   fp32  [16]       (optional Lloyd-Max levels)

INT8 (``int8_tensorwise``), per quantized ``<prefix>``::

    <prefix>.weight            int8  [N, K]
    <prefix>.weight_scale      fp32  scalar | [N] | [N, 1]   (tensorwise or per-row)
    <prefix>.comfy_quant       u8    JSON blob, optional; ``convrot:true``
                                means the stored weight was Hadamard-rotated
                                and must be un-rotated after scaling.

Anything else in the same file (BF16/F32 norms, biases, tables, ...) is a
plain tensor and passes through untouched -- that is the "mixed file"
support: detection is per-tensor, via companion-key presence.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch

logger = logging.getLogger("weellm")

__all__ = [
    "build_hadamard",
    "rotate_weight",
    "dequantize_int8",
    "dequantize_w4a8",
    "dequantize_int4",
    "classify_quant_groups",
    "is_quantized_key",
    "strip_comfy_prefix",
]

# Comfy single-file checkpoints wrap the diffusion transformer in this
# prefix (diffusers repos instead use transformer/... + connectors/...).
# WeeLLM streamers match diffusers-namespaced keys, so the seeker strips it.
_COMFY_DIFFUSION_PREFIX = "model.diffusion_model."


def strip_comfy_prefix(key: str) -> str:
    """``model.diffusion_model.X`` -> ``X`` (LTX single-file convention)."""
    if key.startswith(_COMFY_DIFFUSION_PREFIX):
        return key[len(_COMFY_DIFFUSION_PREFIX):]
    return key

# ---------------------------------------------------------------------------
# Hadamard (ConvRot) helpers -- ported from comfy_kitchen.tensor.int8_utils.
# The rotation matrix H is orthogonal and symmetric, so applying the same
# grouped multiply a second time undoes the rotation.
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: Dict[tuple, torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized Hadamard matrix of ``size`` (must be a power of 4, e.g. 256)."""
    key = (int(size), str(device), dtype)
    hit = _HADAMARD_CACHE.get(key)
    if hit is not None:
        return hit
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h, cur = h4, 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Un-rotate a ``[out, in]`` matrix: ``W = W_rot @ H^T`` per group.

    ``weight`` may be any floating dtype; math runs in float32 for accuracy.
    """
    if weight.dim() != 2:
        raise ValueError(f"rotate_weight needs a 2D matrix, got {weight.dim()}D")
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group {group_size}")
    h = build_hadamard(group_size, device=weight.device, dtype=torch.float32)
    w = weight.to(torch.float32).reshape(out_f, in_f // group_size, group_size)
    return torch.matmul(w, h.T.to(device=w.device)).reshape(out_f, in_f)


# ---------------------------------------------------------------------------
# INT8
# ---------------------------------------------------------------------------


def dequantize_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``q.float() * scale`` (+ Hadamard un-rotation for ConvRot weights)."""
    out = q.to(torch.float32) * scale.to(torch.float32)
    if convrot:
        out = rotate_weight(out, convrot_groupsize)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


# ---------------------------------------------------------------------------
# W4A8 -- ported from comfy_kitchen.backends.eager.w4a8_int8
# (dequantize_w4a8_int8_weight + _dequant_int4_grouped_to_int8).
# ---------------------------------------------------------------------------


def dequantize_w4a8(
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decode one W4A8 group to a floating ``[N, K]`` weight matrix.

    Steps: nibble-unpack -> codebook (or -8) -> per-group fp8 scale ->
    per-channel scale (+ optional correction) -> round to output dtype ->
    Hadamard un-rotation in the output dtype (same order as ComfyUI, so
    results are bit-identical to its eager backend).
    """
    if qdata.dim() != 2:
        raise ValueError(f"W4A8 qdata must be 2D, got {qdata.dim()}D")
    n, k_half = qdata.shape
    k = k_half * 2
    if k % group_size != 0 or k % convrot_groupsize != 0:
        raise ValueError(
            f"W4A8 K={k} must be divisible by group_size={group_size} and "
            f"convrot_groupsize={convrot_groupsize}"
        )
    groups = k // group_size
    if tuple(s_rel.shape) != (n, groups):
        raise ValueError(f"s_rel must be {(n, groups)}, got {tuple(s_rel.shape)}")
    if tuple(s_channel.shape) != (n,):
        raise ValueError(f"s_channel must be {(n,)}, got {tuple(s_channel.shape)}")
    if codebook is not None and tuple(codebook.shape) != (16,):
        raise ValueError(f"codebook must be (16,), got {tuple(codebook.shape)}")

    # 1. two int4 codes per stored byte: even index = low nibble.
    packed = qdata.to(torch.int32) & 0xFF
    codes = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF

    # 2. levels -> values.
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[codes]
    else:
        values = codes.float() - 8.0

    # 3. per-group relative scale -> back on the INT8 grid Comfy's GEMM uses.
    int8w = (
        (values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1))
        .view(n, k)
        .round()
        .clamp_(-127, 127)
        .to(torch.int8)
    )

    # 4. per-channel scale (+ optional asymmetric correction), still rotated basis.
    out = int8w * s_channel.float().view(n, 1)
    if correction is not None:
        out = out + correction.float().t()

    # 5. un-rotate to the original basis. Order matters for bit-parity with
    # ComfyUI: round to the output dtype FIRST, then rotate in that dtype
    # (their H is built in the weight dtype, not float32).
    out_dtype = dtype if dtype is not None else torch.bfloat16
    out = out.to(out_dtype)
    h = build_hadamard(convrot_groupsize, device=out.device, dtype=out.dtype)
    g = k // convrot_groupsize
    out = torch.matmul(
        out.view(n, g, convrot_groupsize), h.T.to(device=out.device, dtype=out.dtype)
    ).view(n, k)
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# INT4 (ConvRot W4A4) -- ported from comfy_kitchen.backends.eager.convrot_w4a4
# (dequantize_convrot_w4a4_weight + svdquant nibble codec).
#
# On disk, per quantized <prefix>:  <prefix>.weight (int8, [N, K/2], two
# SIGNED int4 codes per byte, low nibble = even column) + <prefix>.weight_scale
# (fp32, [N] per-row). Same companion names as INT8 -- the two are told apart
# by the comfy_quant blob / _quantization_metadata ``format`` field
# (``convrot_w4a4`` vs ``int8_tensorwise``); without any marker the group is
# treated as plain INT8.
# ---------------------------------------------------------------------------


def _unpack_int4_signed(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Row-major nibble unpack with signed interpretation ([-8, 7])."""
    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def dequantize_int4(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = 64,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decode one INT4-ConvRot group to a floating ``[N, K]`` weight matrix.

    Steps: signed-nibble unpack -> per-row scale -> Hadamard un-rotation in
    float32 (same order as ComfyUI) -> cast to ``dtype``.
    """
    if qdata.dim() != 2:
        raise ValueError(f"INT4 qdata must be 2D, got {qdata.dim()}D")
    n, k_half = qdata.shape
    k = k_half * 2
    if k % convrot_groupsize != 0 or k % quant_group_size != 0:
        raise ValueError(
            f"INT4 K={k} must be divisible by convrot_groupsize={convrot_groupsize} "
            f"and quant_group_size={quant_group_size}"
        )
    if tuple(scale.shape) not in ((n,), (n, 1)):
        raise ValueError(f"INT4 scale must be {(n,)} or {(n, 1)}, got {tuple(scale.shape)}")
    w = _unpack_int4_signed(qdata, k).to(torch.float32)
    w = w * scale.to(torch.float32).reshape(n, 1)
    out = rotate_weight(w, convrot_groupsize)
    out_dtype = dtype if dtype is not None else torch.bfloat16
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# File-level classification: which logical ".weight" keys are quantized?
# ---------------------------------------------------------------------------

_W4A8_SCALE_SUFFIX = ".weight_s_rel"
_INT8_SCALE_SUFFIX = ".weight_scale"


def classify_quant_groups(header: Dict[str, dict]) -> Dict[str, dict]:
    """Scan a safetensors header ``{key: {dtype, shape}}`` for quant groups.

    Returns ``{logical_weight_key: spec}`` where spec has ``kind`` of
    ``"w4a8"`` / ``"int8"`` plus companion key names. Plain tensors are
    absent from the result. Companion keys themselves (scales, codebooks)
    are NOT logical keys -- the seeker pulls them automatically.
    """
    groups: Dict[str, dict] = {}
    keys = set(header)
    for key, meta in header.items():
        if key == "__metadata__" or not key.endswith(".weight"):
            continue
        prefix = key[: -len(".weight")]
        if meta.get("dtype") != "I8":
            continue  # quantized weights are stored as int8 bytes
        rel = prefix + _W4A8_SCALE_SUFFIX
        if rel in keys:
            spec: dict = {
                "kind": "w4a8",
                "weight": key,
                "s_rel": rel,
                "s_channel": prefix + ".weight_s_channel",
                "codebook": prefix + ".weight_codebook",
                "group_size": 16,
                "convrot_groupsize": 256,
            }
            # group_size is confirmed by the s_rel shape: [N, K/16].
            shape = header[rel].get("shape", [])
            wshape = meta.get("shape", [])
            if len(shape) == 2 and len(wshape) == 2 and wshape[1] * 2 > 0:
                k_logical = wshape[1] * 2
                if shape[1] > 0 and k_logical % shape[1] == 0:
                    spec["group_size"] = k_logical // shape[1]
            groups[key] = spec
            continue
        sc = prefix + _INT8_SCALE_SUFFIX
        if sc in keys:
            groups[key] = {
                "kind": "int8",
                "weight": key,
                "scale": sc,
                "convrot": False,  # flipped below if a comfy_quant blob says so
                "convrot_groupsize": 256,
            }
    return groups


def is_quantized_key(key: str, header: Dict[str, dict]) -> bool:
    """True if ``key`` is a scale/codebook companion (not a model tensor)."""
    return key.endswith(
        (".weight_s_rel", ".weight_s_channel", ".weight_codebook", ".weight_scale")
    ) and not key.endswith(".weight")

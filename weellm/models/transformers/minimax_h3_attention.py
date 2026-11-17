"""
minimax_h3_attention.py -- safe attention for MiniMax-H3's packed sequence.

MiniMax-H3 packs text + target audio + target video rows into ONE
self-attention document.  At the studio's 960x544 / 124-frame defaults that
document is ~19,290 rows.  The math SDPA backend materialises the full
``[heads, rows, rows]`` score matrix in fp32::

    56 * 19290 * 19290 * 4 bytes = 77.6 GiB

which is an instant OOM on a 15 GB T4.  PyTorch only reaches the
memory-efficient cutlass kernel when the dtype is supported by it: on sm75
(T4) that means fp16 -- bf16 has no memory-efficient kernel before sm80, so
SDPA silently falls back to math.

This module does three things, all from WeeLLM's own code (no edits to the
diffusers install):

  1. reports which SDPA kernels are usable for the running GPU + dtype;
  2. forces the SDPA kernel flags into a sane state;
  3. wraps the MiniMax-H3 module's ``dispatch_attention_fn`` so that, when
     only the math backend is usable, queries are processed in slices whose
     score matrix fits a bounded VRAM budget.  The slice loop is exact
     (softmax is row-wise, so splitting queries does not change the result)
     and shrinks itself further if a slice still OOMs.

The wrapper is installed once per process and is a no-op on machines where
flash / memory-efficient attention is available.
"""

from __future__ import annotations

import gc
import logging
from typing import Tuple

import torch

logger = logging.getLogger("weellm")

# A budget at or above this means "never chunk" (flash / mem-efficient
# kernels never materialise the score matrix, so there is nothing to bound).
_NO_CHUNK = 1 << 61

_CFG = {
    "installed": False,
    # ``None`` = auto: measure free VRAM at attention time and use a fraction
    # of it.  An integer is an absolute score-matrix budget in bytes.
    "budget_bytes": None,
    "chunk_logged": set(),
    "head_dim": None,
}


def _capability(device=None) -> Tuple[int, int]:
    try:
        return tuple(torch.cuda.get_device_capability(device))
    except Exception:
        return (0, 0)


def mem_efficient_available(dtype: torch.dtype, device=None) -> bool:
    """True when PyTorch's cutlass memory-efficient SDPA kernel can run this dtype.

    The kernel supports fp16 from sm70 upwards; bf16 is rejected below sm80
    (T4 is sm75), which is exactly why bf16 on a T4 falls back to the 77 GiB
    math pass.
    """
    if not torch.cuda.is_available():
        return False
    major, _minor = _capability(device)
    if dtype == torch.float16:
        return major >= 7
    if dtype == torch.bfloat16:
        return major >= 8
    return False


def flash_available(dtype: torch.dtype, device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = _capability(device)
    return major >= 8 and dtype in (torch.float16, torch.bfloat16)


def _slice_mask(mask, start: int, end: int, seq_q: int):
    """Slice a query dimension out of an attention mask (or leave it alone)."""
    if mask is None or not torch.is_tensor(mask):
        return mask
    try:
        if mask.dim() == 2 and mask.shape[0] == seq_q:
            return mask[start:end]
        if mask.dim() == 3 and mask.shape[1] == seq_q:
            return mask[:, start:end]
        if mask.dim() == 4 and mask.shape[2] == seq_q:
            return mask[:, :, start:end]
        if mask.dim() >= 2 and mask.shape[-2] == seq_q:
            return mask[..., start:end, :]
    except Exception:
        pass
    return mask


def _attention_budget(device) -> int:
    """How large a score matrix we are willing to materialise right now.

    Measured at attention time, not at call time: the text encoder and the
    streamed blocks have a very different VRAM footprint by then.
    """
    try:
        free_bytes = torch.cuda.mem_get_info(device)[0]
    except Exception:
        free_bytes = 0
    return int(max(384 << 20, min(3 << 30, max(0, free_bytes) * 0.15)))


def _chunked_dispatch(orig, query, key, value, args, kwargs, chunk: int, row_bytes: int):
    seq_q = query.shape[1]
    attn_mask = kwargs.get("attn_mask")
    out_parts = []
    start = 0
    while start < seq_q:
        end = min(seq_q, start + chunk)
        kw = dict(kwargs)
        if "attn_mask" in kwargs:
            kw["attn_mask"] = _slice_mask(attn_mask, start, end, seq_q)
        try:
            out_parts.append(orig(query[:, start:end], key, value, *args, **kw))
        except torch.cuda.OutOfMemoryError:
            # Very tight VRAM: halve the slice and retry the same rows.  The
            # reduced budget sticks for later blocks too.
            if chunk <= 16:
                raise
            chunk = max(16, chunk // 2)
            _CFG["budget_bytes"] = min(
                _CFG["budget_bytes"] if _CFG["budget_bytes"] is not None else _NO_CHUNK,
                chunk * row_bytes,
            )
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.warning(
                "[MiniMax-H3] Attention slice OOM — retrying with %d-row slices (%.2f GiB budget).",
                chunk, (_CFG["budget_bytes"] / 2**30),
            )
            continue
        start = end
    return torch.cat(out_parts, dim=1) if len(out_parts) > 1 else out_parts[0]


def _dispatch(orig, query, key, value, *args, **kwargs):
    """Drop-in replacement for diffusers' ``dispatch_attention_fn``."""
    if query.dim() != 4:
        return orig(query, key, value, *args, **kwargs)

    try:
        _batch, seq_q, heads, _head_dim = query.shape
        seq_kv = key.shape[1]
        # The math backend upcasts to fp32 before materialising the scores.
        row_bytes = max(1, int(seq_kv) * int(heads) * 4)

        budget = _CFG["budget_bytes"]
        if budget is None:
            budget = _attention_budget(query.device)
        if budget >= _NO_CHUNK:
            try:
                return orig(query, key, value, *args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                # The fast kernel rejected this shape/dtype and SDPA fell back
                # to math.  Recover here instead of letting the whole step die.
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                    free_bytes = torch.cuda.mem_get_info(query.device)[0]
                except Exception:
                    free_bytes = 0
                budget = int(max(384 << 20, min(3 << 30, max(0, free_bytes) * 0.20)))
                _CFG["budget_bytes"] = budget
                logger.warning(
                    "[MiniMax-H3] Attention OOM on the fast kernel — switching to a "
                    "%.2f GiB bounded score budget for the rest of this run.",
                    budget / 2**30,
                )

        if seq_q * row_bytes <= budget:
            return orig(query, key, value, *args, **kwargs)

        chunk = max(1, min(int(seq_q), int(budget) // row_bytes))
        if chunk > 32:
            # Quantise to a power of two so tiny free-VRAM fluctuations do not
            # produce a new chunk size (and a new log line) on every block.
            chunk = 1 << (chunk.bit_length() - 1)
        marker = (int(seq_q), int(heads), int(chunk))
        if marker not in _CFG["chunk_logged"]:
            _CFG["chunk_logged"].add(marker)
            logger.warning(
                "[MiniMax-H3] Chunked attention engaged: %d rows × %d heads in %d-row "
                "slices (%.2f GiB score budget) — math backend fallback, so steps will be slower.",
                seq_q, heads, chunk, budget / 2**30,
            )
        return _chunked_dispatch(orig, query, key, value, args, kwargs, chunk, row_bytes)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        # Never let the safety wrapper break a normal attention call.
        return orig(query, key, value, *args, **kwargs)


def _install_wrapper() -> bool:
    if _CFG["installed"]:
        return True
    try:
        import diffusers.models.transformers.transformer_minimax_h3 as mod
    except Exception:
        return False
    orig = getattr(mod, "dispatch_attention_fn", None)
    if orig is None:
        return False
    if getattr(orig, "_weellm_chunked", False):
        _CFG["installed"] = True
        return True

    def dispatch(query, key, value, *args, **kwargs):
        return _dispatch(orig, query, key, value, *args, **kwargs)

    dispatch._weellm_chunked = True
    dispatch._weellm_original = orig
    mod.dispatch_attention_fn = dispatch
    _CFG["installed"] = True
    logger.debug("[MiniMax-H3] Installed chunk-safe attention wrapper.")
    return True


def _parse_patch_size(patch_size) -> Tuple[int, int, int]:
    if isinstance(patch_size, (list, tuple)):
        nums = [int(p) for p in patch_size if isinstance(p, (int, float))]
        if len(nums) >= 3:
            return max(1, nums[-3]), max(1, nums[-2]), max(1, nums[-1])
        if len(nums) == 2:
            return 1, max(1, nums[0]), max(1, nums[1])
        if nums:
            return 1, max(1, nums[0]), max(1, nums[0])
    try:
        p = max(1, int(patch_size))
        return 1, p, p
    except Exception:
        return 1, 2, 2


def estimate_packed_rows(
    num_frames: int,
    height: int,
    width: int,
    patch_size=(1, 2, 2),
    vae_scale: int = 16,
) -> Tuple[int, int, int]:
    """Return ``(video_rows, audio_rows, latent_frames)`` for a MiniMax-H3 clip.

    Frames are aligned to the video VAE's ``17k + 5`` grid, which yields
    ``5k + 2`` latent frames.  Audio is 40 latents/second at 24 fps, carried
    as two channel-major blocks.
    """
    n = int(num_frames)
    n = max(5, ((n - 5) // 17) * 17 + 5)
    latent_frames = (n - 5) // 17 * 5 + 2

    pt, ph, pw = _parse_patch_size(patch_size)
    scale = max(1, int(vae_scale))
    lat_h = max(1, int(height) // scale)
    lat_w = max(1, int(width) // scale)
    video_rows = (latent_frames // pt) * (lat_h // ph) * (lat_w // pw)
    audio_rows = 2 * int(round(n / 24.0 * 40.0))
    return video_rows, audio_rows, latent_frames


def configure_for_pipeline(
    pipe,
    prompt: str = "",
    height: int = 544,
    width: int = 960,
    num_frames: int = 121,
) -> dict:
    """Pick a safe SDPA setup for one MiniMax-H3 call and log the preflight.

    Returns a dict so callers/tests can inspect the decision.
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        transformer = getattr(pipe, "transformer_ref", None)
    streamer = getattr(transformer, "_weellm_streamer", None)
    model = getattr(streamer, "model", transformer)

    device = getattr(streamer, "device", None) or getattr(transformer, "device", None) or "cuda"
    dtype = getattr(streamer, "dtype", None) or getattr(transformer, "dtype", None) or torch.bfloat16
    config = getattr(model, "config", None)

    patch_size = getattr(config, "patch_size", (1, 2, 2))
    heads = int(getattr(config, "num_attention_heads", 56) or 56)
    vae_scale = int(getattr(config, "vae_scale_factor", 16) or 16)

    cap = _capability(device)
    eff = mem_efficient_available(dtype, device)
    flash = flash_available(dtype, device)

    # Put PyTorch's SDPA kernel flags in a known state.  Math stays enabled
    # as a safety net: odd shapes/head dims can be rejected by the fast
    # kernels and PyTorch should fall back instead of raising.
    try:
        torch.backends.cuda.enable_flash_sdp(bool(flash))
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    if eff or flash:
        backend = "flash" if flash else "memory-efficient"
        budget = _NO_CHUNK
        budget_note = ""
    else:
        backend = "chunked-math"
        # None = auto: the real budget is measured at attention time, after
        # the text encoder has been evicted and blocks are resident.  Sizing
        # it here (before the encoder runs) would badly overestimate it.
        budget = None
        budget_note = " (auto: 15% of free VRAM at attention time)"

    _CFG["budget_bytes"] = budget
    _CFG["head_dim"] = getattr(config, "attention_head_dim", None)
    installed = _install_wrapper()

    video_rows, audio_rows, latent_frames = estimate_packed_rows(
        num_frames, height, width, patch_size, vae_scale
    )
    # The text rows are prompt dependent and tiny next to video; estimate
    # from the words rather than pretending to know the tokenizer output.
    text_rows = max(8, len((prompt or "").split()) * 2 + 8)
    total_rows = video_rows + audio_rows + text_rows
    matrix_gib = total_rows * total_rows * heads * 4 / 2**30

    try:
        free_gib = torch.cuda.mem_get_info(device)[0] / 2**30
    except Exception:
        free_gib = -1.0
    try:
        model_matrix_gib = total_rows * total_rows * heads * dtype.itemsize / 2**30
    except Exception:
        model_matrix_gib = 0.0

    logger.info(
        "[MiniMax-H3] preflight: %d frames · %dx%d → %s video + %s audio + ~%d text "
        "= %s rows · %d heads · a full fp32 score matrix would be %.1f GiB "
        "(model dtype %s) · free VRAM %.1f GiB · backend=%s%s",
        num_frames, width, height, f"{video_rows:,}", f"{audio_rows:,}", text_rows,
        f"{total_rows:,}", heads, matrix_gib, str(dtype).replace("torch.", ""),
        free_gib, backend, budget_note,
    )

    if not (eff or flash):
        if dtype == torch.bfloat16 and cap[0] < 8:
            logger.warning(
                "[MiniMax-H3] bf16 has no memory-efficient attention kernel on sm%d%d — "
                "switch the studio Dtype to float16.  Until then this run is chunked and slow.",
                cap[0], cap[1],
            )
        logger.warning(
            "[MiniMax-H3] Attention will run in bounded slices, not one %.1f GiB score matrix. "
            "Generation is memory-safe but slower than normal.",
            matrix_gib,
        )
    if not installed:
        logger.warning(
            "[MiniMax-H3] Could not install the chunk-safe dispatch wrapper "
            "(diffusers layout changed?) — relying on the SDPA kernel selection."
        )

    return {
        "backend": backend,
        "dtype": dtype,
        "capability": cap,
        "budget_bytes": budget,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "text_rows": text_rows,
        "total_rows": total_rows,
        "heads": heads,
        "matrix_gib": matrix_gib,
        "model_matrix_gib": model_matrix_gib,
        "free_gib": free_gib,
        "installed": installed,
    }

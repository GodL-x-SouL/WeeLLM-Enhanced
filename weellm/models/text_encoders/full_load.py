"""
full_load.py -- Resident (full-load) encode phase for text encoder streamers.

WeeLLM's default TE path streams every layer from disk during encoding, which
is needlessly slow on hosts where the encoder weights fit in VRAM + RAM (e.g.
Colab T4: 15 GB VRAM / 12.5 GB RAM). Before the encode forward, TEFullLoad
materializes the streamed layers once, in execution order:

  VRAM tier  -- placed on the GPU up to the free-VRAM budget. Plain checkpoints
                stay as compute-dtype parameters (hooks become no-ops); staged
                (GGUF / Comfy-quant) checkpoints keep their raw quantized bytes
                on the GPU and dequantize each layer on-device during forward.
  RAM tier   -- held in host RAM (raw stored bytes for staged seekers,
                compute-dtype tensors otherwise). Hooks H2D / dequantize from
                RAM instead of reading disk.
  disk tier  -- anything that fits neither budget keeps the legacy lazy path.

The pipeline releases all residency (release()) on the first diffusion
forward, so the transformer still streams layer-by-layer with the full budget.
Set WEELLM_TE_FULL_LOAD=0 to force the legacy streaming behaviour.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from weellm.io.memory import evict_module
from weellm.io.utils import clean_memory

logger = logging.getLogger("weellm")

# Headroom for TE activations / allocator slack during the encode forward.
_VRAM_SAFETY_BYTES = 1024**3
# Headroom for the OS, CUDA host allocations and transient read buffers.
_RAM_SAFETY_BYTES = 1024**3

_FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def free_vram_bytes(device: str) -> int:
    """Free VRAM for TE residency, honouring the user's vram_budget if set."""
    if not torch.cuda.is_available() or str(device).startswith("cpu"):
        return 0
    from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer

    budget = BaseTransformerStreamer._global_vram_budget_gb
    if budget is not None:
        return max(0, int(budget * 1024**3) - torch.cuda.memory_allocated(device))
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def free_ram_bytes() -> int:
    """Free host RAM for TE residency, honouring the user's ram_budget if set."""
    try:
        import psutil
    except ImportError:
        return 0
    from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer

    budget = BaseTransformerStreamer._global_ram_budget_gb
    if budget is not None:
        rss = psutil.Process(os.getpid()).memory_info().rss
        return max(0, int(budget * 1024**3) - rss)
    return int(psutil.virtual_memory().available)


def logical_bytes(seeker, keys: List[str], dtype: torch.dtype) -> int:
    """Compute-dtype footprint of *keys* (what VRAM holds after dequant/cast)."""
    if not getattr(seeker, "supports_staged_prefetch", False):
        from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

        if isinstance(seeker, SafetensorsBase):
            import numpy as np

            total = 0
            for key in keys:
                src = seeker.weight_map.get(key)
                if src is None:
                    continue
                header, _ = seeker._read_header(seeker.model_dir / src)
                meta = header.get(key)
                if not meta:
                    continue
                shape = meta.get("shape", [])
                dstr = meta.get("dtype", "F32")
                count = int(np.prod(shape)) if shape else 1
                if dstr in _FLOAT_DTYPES:
                    total += count * dtype.itemsize
                else:
                    total += count * np.dtype(DTYPE_MAP[dstr]).itemsize
            return total
    # GGUF / Comfy-quant seekers already report the dequantized bf16 footprint.
    return int(seeker.get_block_bytes(keys))


def raw_bytes(seeker, keys: List[str], dtype: torch.dtype) -> int:
    """Stored-byte footprint (what RAM holds for staged seekers)."""
    try:
        return int(seeker.get_block_bytes(keys, raw=True))
    except TypeError:
        return logical_bytes(seeker, keys, dtype)


def _map_tensors(payload, fn):
    """Apply *fn* to every tensor nested in a staged payload, preserving structure."""
    if torch.is_tensor(payload):
        return fn(payload)
    if isinstance(payload, tuple):
        return tuple(_map_tensors(p, fn) for p in payload)
    if isinstance(payload, list):
        return [_map_tensors(p, fn) for p in payload]
    if isinstance(payload, dict):
        return {k: _map_tensors(v, fn) for k, v in payload.items()}
    return payload


def has_meta_residents(model: nn.Module, keys: List[str]) -> bool:
    """True if any resident parameter (or buffer) was evicted back to meta."""
    for key in keys:
        obj = model
        try:
            for part in key.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if torch.is_tensor(obj) and obj.device.type == "meta":
            return True
    return any(b.device.type == "meta" for b in model.buffers())


class TEFullLoad:
    """Budget-aware full materialization of one text encoder's streamed blocks.

    ``blocks`` is an ordered list of ``(block_id, weight_map_keys, module)``.
    ``place`` writes a compute-dtype GPU state dict into the model (including
    any architecture-specific key mapping). ``reload_residents`` re-loads the
    non-streamed weights if a previous diffusion pass evicted them.
    """

    def __init__(
        self,
        owner,
        seeker,
        device: str,
        dtype: torch.dtype,
        blocks: List[Tuple[object, List[str], nn.Module]],
        place: Callable[[Dict[str, torch.Tensor]], None],
        reload_residents: Optional[Callable[[], None]] = None,
        cancel_prefetch: Optional[Callable[[], None]] = None,
    ):
        self._owner = owner
        self._seeker = seeker
        self._device = device
        self._dtype = dtype
        self._blocks = [b for b in blocks if b[1]]
        self._place = place
        self._reload_residents = reload_residents
        self._cancel_prefetch = cancel_prefetch
        self._staged = bool(getattr(seeker, "supports_staged_prefetch", False))

        self.active = False
        self.disabled = os.environ.get("WEELLM_TE_FULL_LOAD", "1") == "0"
        self._resident_ids: set = set()   # plain VRAM tier: parameters live on GPU
        self._store: Dict[object, object] = {}  # staged payloads / CPU state dicts
        self._tiers: Dict[object, str] = {}

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def ensure_materialized(self) -> None:
        if self.active or self.disabled or not self._blocks:
            return
        try:
            self._materialize()
            self.active = True
        except Exception as exc:
            logger.warning(
                "[TE Full-Load] Materialization failed (%s) — falling back to layer streaming.", exc
            )
            self._cleanup()
            self.disabled = True

    def _materialize(self) -> None:
        t0 = time.time()
        if self._cancel_prefetch is not None:
            self._cancel_prefetch()
        if self._reload_residents is not None:
            self._reload_residents()

        infos = []
        for bid, keys, module in self._blocks:
            logical = logical_bytes(self._seeker, keys, self._dtype)
            stored = raw_bytes(self._seeker, keys, self._dtype) if self._staged else logical
            infos.append((bid, keys, module, logical, stored))

        vram_free = free_vram_bytes(self._device)
        ram_free = free_ram_bytes()
        max_logical = max((i[3] for i in infos), default=0)
        if self._staged:
            # Raw bytes live on GPU; reserve workspace to dequantize one layer
            # plus activation headroom on top of the resident raw bytes.
            vram_cap = max(0, vram_free - max_logical - _VRAM_SAFETY_BYTES)
        else:
            vram_cap = max(0, vram_free - _VRAM_SAFETY_BYTES)
        ram_cap = max(0, ram_free - _RAM_SAFETY_BYTES)

        used_vram = used_ram = 0
        vram_full = ram_full = False
        for bid, keys, module, logical, stored in infos:
            cost = stored if self._staged else logical
            if not vram_full and used_vram + cost <= vram_cap:
                tier = "vram"
            elif not ram_full and used_ram + cost <= ram_cap:
                tier = "ram"
            else:
                tier = "disk"

            if tier == "vram":
                try:
                    self._load_vram(bid, keys, module)
                    used_vram += cost
                    self._tiers[bid] = "vram"
                    continue
                except torch.cuda.OutOfMemoryError:
                    logger.warning("[TE Full-Load] VRAM exhausted at %s — spilling remaining layers.", bid)
                    vram_full = True
                    self._safe_empty_cache()
                    tier = "ram" if not ram_full and used_ram + cost <= ram_cap else "disk"
            if tier == "ram":
                try:
                    self._load_ram(bid, keys)
                    used_ram += cost
                    self._tiers[bid] = "ram"
                    continue
                except (torch.cuda.OutOfMemoryError, MemoryError):
                    logger.warning("[TE Full-Load] RAM exhausted at %s — streaming remaining layers.", bid)
                    ram_full = True
                    tier = "disk"
            self._tiers[bid] = "disk"

        # A later encode on a reused pipeline must re-arm the pipeline's
        # TE-eviction gate so residency is released before diffusion again.
        tr_module = getattr(self._owner, "_weellm_tr_module", None)
        if tr_module is not None:
            tr_module._weellm_te_evicted = False

        counts = {t: sum(1 for v in self._tiers.values() if v == t) for t in ("vram", "ram", "disk")}
        logger.info(
            "[TE Full-Load] %d layers ready in %.1fs: %d in VRAM (%.1f GB), %d in RAM (%.1f GB), %d disk-streamed.",
            len(infos), time.time() - t0,
            counts["vram"], used_vram / 1e9, counts["ram"], used_ram / 1e9, counts["disk"],
        )

    def _load_vram(self, bid, keys: List[str], module: nn.Module) -> None:
        if self._staged:
            # Keep raw quantized bytes on the GPU; forward dequantizes per layer.
            payload = self._seeker.get_staged(keys, use_cache=False)
            self._store[bid] = _map_tensors(payload, lambda t: t.to(self._device, non_blocking=True))
            if torch.cuda.is_available():
                torch.cuda.synchronize(self._device)
        else:
            sd = self._get_tensors(keys, device=self._device)
            self._place(sd)
            del sd
            self._resident_ids.add(bid)

    def _load_ram(self, bid, keys: List[str]) -> None:
        if self._staged:
            # use_cache=True clones mmap bytes into anonymous RAM on small hosts.
            self._store[bid] = self._seeker.get_staged(keys, use_cache=True)
        else:
            self._store[bid] = self._get_tensors(keys, device="cpu")

    def _get_tensors(self, keys: List[str], device: str) -> Dict[str, torch.Tensor]:
        try:
            return self._seeker.get_tensors(keys, device=device, dtype=self._dtype, use_cache=False)
        except TypeError:
            return self._seeker.get_tensors(keys, device=device, dtype=self._dtype)

    # ------------------------------------------------------------------
    # Hook interface
    # ------------------------------------------------------------------

    def resident(self, bid) -> bool:
        """True when the block's parameters already live in VRAM (skip load/evict)."""
        return bid in self._resident_ids

    def fetch(self, bid) -> Optional[Dict[str, torch.Tensor]]:
        """GPU state dict for RAM/VRAM-staged blocks; None → legacy disk path."""
        payload = self._store.get(bid)
        if payload is None:
            return None
        if self._staged:
            return self._seeker.resolve_staged(payload, self._device, self._dtype)
        sd = {k: v.to(self._device, non_blocking=True) for k, v in payload.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
        return sd

    # ------------------------------------------------------------------
    # Release (called before the diffusion transformer starts streaming)
    # ------------------------------------------------------------------

    def release(self) -> None:
        if not self.active and not self._store and not self._resident_ids:
            return
        for bid, keys, module in self._blocks:
            if bid in self._resident_ids:
                evict_module(module)
        self._cleanup()
        logger.info("[TE Full-Load] Released encoder residency — diffusion gets the full budget.")

    def _cleanup(self) -> None:
        self._store.clear()
        self._resident_ids.clear()
        self._tiers.clear()
        self.active = False
        self._safe_empty_cache()

    def _safe_empty_cache(self) -> None:
        clean_memory(self._device)
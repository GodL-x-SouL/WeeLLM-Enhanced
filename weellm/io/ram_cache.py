"""
ram_cache.py -- Bounded cache for raw (pre-dequant) tensor bytes.

The cache is shared by every seeker in the process. Upstream streamers read
the same blocks once per denoising step, so keeping the *stored* bytes
(quantized for GGUF / INT8 packs, bf16 for full checkpoints) in host RAM
turns repeated disk reads into RAM reads. Entries are keyed by
``(seeker_id, on_disk_key)``. Insertions never evict earlier entries: under
a cyclic block scan an LRU cache misses every time, while a fixed subset
that fits the budget hits on every step.

Only raw storage tensors are cached. Dequantized copies are intentionally
never cached at this layer: dequantizing is cheap on the GPU and caching the
expanded tensors would blow the RAM budget.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Dict, Optional

import torch

logger = logging.getLogger("weellm")

# Auto budget used on hosts below 16 GB total RAM (free Colab / Kaggle).
_SMALL_HOST_CACHE_BYTES = 2 * 1024**3
_SMALL_HOST_RESERVE_BYTES = 6 * 1024**3


class RawTensorCache:
    """Thread-safe byte-budget cache of raw CPU tensors (non-evicting puts)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._bytes = 0
        self._budget = 0
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._budget > 0

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def used_bytes(self) -> int:
        return self._bytes

    def configure(self, budget_bytes: int) -> None:
        """Set the cache budget in bytes (0 disables the cache and frees it)."""
        budget_bytes = max(0, int(budget_bytes))
        with self._lock:
            self._budget = budget_bytes
            self._evict_locked(0)
            if budget_bytes == 0:
                self._entries.clear()
                self._bytes = 0

    def get(self, namespace, key) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        cache_key = (namespace, key)
        with self._lock:
            tensor = self._entries.get(cache_key)
            if tensor is None:
                self._misses += 1
                return None
            self._entries.move_to_end(cache_key)
            self._hits += 1
            return tensor

    def put(self, namespace, key, tensor: torch.Tensor) -> None:
        """Insert *tensor* unless the budget is exhausted.

        Deliberately non-evicting: transformer blocks are re-read in a fixed
        cyclic order every denoising step, and classic LRU has a 0% hit rate
        on cyclic scans when the working set exceeds the cache. Keeping the
        first blocks that fit gives a stable subset that hits every step.
        """
        if tensor is None or not self.enabled:
            return
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self._budget:
            return
        cache_key = (namespace, key)
        with self._lock:
            old = self._entries.pop(cache_key, None)
            if old is not None:
                self._bytes -= old.numel() * old.element_size()
            if self._bytes + nbytes > self._budget:
                if old is not None:
                    self._entries[cache_key] = old
                    self._bytes += old.numel() * old.element_size()
                return
            self._entries[cache_key] = tensor
            self._bytes += nbytes

    def _evict_locked(self, incoming: int) -> None:
        while self._entries and self._bytes + incoming > self._budget:
            _, tensor = self._entries.popitem(last=False)
            self._bytes -= tensor.numel() * tensor.element_size()
        if self._bytes < 0:
            self._bytes = 0

    def clear(self, namespace=None) -> None:
        with self._lock:
            if namespace is None:
                self._entries.clear()
                self._bytes = 0
                return
            for cache_key in [k for k in self._entries if k[0] == namespace]:
                tensor = self._entries.pop(cache_key)
                self._bytes -= tensor.numel() * tensor.element_size()
            self._bytes = max(0, self._bytes)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "budget_gb": self._budget / 1e9,
                "used_gb": self._bytes / 1e9,
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
            }


_GLOBAL_CACHE = RawTensorCache()


def get_raw_cache() -> RawTensorCache:
    return _GLOBAL_CACHE


def configure_ram_cache(budget_bytes: int) -> None:
    _GLOBAL_CACHE.configure(budget_bytes)


def clear_ram_cache() -> None:
    _GLOBAL_CACHE.clear()


def auto_ram_cache_budget() -> int:
    """Conservative raw-cache budget for small hosts (Colab/Kaggle).

    On hosts with < 16 GB total RAM keep at most 2 GB and always leave 6 GB
    for the pipeline, CUDA host allocations and the OS. Larger hosts default
    to 0 (opt in with ``--ram_cache_gb``).
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        if vm.total < 16 * 1024**3:
            budget = min(_SMALL_HOST_CACHE_BYTES, vm.available - _SMALL_HOST_RESERVE_BYTES)
            return max(0, int(budget))
        return 0
    except Exception:
        return 0

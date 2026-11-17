"""
disk_seek.py -- Disk-based safetensors tensor streamer.

Reads specific tensors directly from disk using file seeks, completely
avoiding memory-mapping and duplicate shard copies. Optimal for NVMe SSDs
and local hardware.

Design principle: this is a pure disk reader. All pipeline scheduling
decisions (when to read, what to cache, how deep to prefetch) belong in the
streamer layer above. This module just reads efficiently.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from weellm.io.ram_cache import get_raw_cache
from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

logger = logging.getLogger("weellm")

# Coalesced-read tuning: merge neighbouring tensors into one read when the
# gap is small, and never allocate a single buffer larger than this.
_COALESCE_GAP = 1 << 20
_COALESCE_MAX_SPAN = 256 << 20

# Source dtypes whose CPU-side cast is always implemented (float8 is not).
_CPU_CASTABLE = {"F64", "F32", "F16", "BF16"}


def _to_tensor(view, dtype_str: str, shape) -> torch.Tensor:
    """Turn a raw byte view into a tensor of the dtype named in the header."""
    arr = np.frombuffer(view, dtype=DTYPE_MAP[dtype_str])
    if shape:
        arr = arr.reshape(shape)
    t = torch.from_numpy(arr)
    if dtype_str == "BF16":
        t = t.view(torch.bfloat16)
    elif dtype_str == "F8_E4M3":
        t = t.view(torch.float8_e4m3fn)
    return t


class SafetensorsDiskSeeker(SafetensorsBase):
    """
    Reads specific tensors from Hugging Face safetensors files directly from
    disk using file seeks.

    Completely avoids memory-mapping and loading entire duplicate shards.
    Uses file.readinto() into a fresh per-tensor bytearray to avoid any
    need for .clone() and to keep peak RAM minimal.

    Best for: local machines with NVMe SSDs.
    """

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self._parse_index()
        self._cache_ns = id(self)

    def get_tensors(
        self,
        keys: List[str],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load the tensors named by *keys* from disk and return them as a dict.

        Parameters
        ----------
        keys:
            Tensor names to load (must be present in ``self.weight_map``).
        device:
            PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        dtype:
            If given, floating-point tensors are cast to this dtype after load.

        Returns
        -------
        Dict mapping tensor name -> ``torch.Tensor``.
        """
        # Group keys by source shard file to minimise file-open overhead.
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            by_src.setdefault(self.weight_map[key], []).append(key)

        result: Dict[str, torch.Tensor] = {}
        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)

            cache = get_raw_cache()
            with open(filepath, "rb") as f:
                # A bulk, uncached load (VAE pinning, text-encoder full load)
                # touches every tensor of a module in one go.  Reading each
                # tensor with its own seek turns that into hundreds of small
                # reads; coalescing neighbours into one read is several times
                # faster on HDD / network / Drive mounts.
                if not use_cache:
                    self._read_coalesced(f, filepath, header, data_base, src_keys, result)
                else:
                    for key in src_keys:
                        t = cache.get(self._cache_ns, key)
                        if t is None:
                            t = self._read_one(f, filepath, header, data_base, key)
                            cache.put(self._cache_ns, key, t)
                        result[key] = t

            # Cast before the host->device copy: the fp32 -> fp16 checkpoints
            # then move half the bytes over PCIe, and the GPU never holds a
            # throwaway fp32 staging copy of the block being pinned.
            for key, t in list(result.items()):
                dtype_str = header[key]["dtype"]
                if (
                    dtype is not None
                    and t.dtype != dtype
                    and t.is_floating_point()
                    and dtype_str in _CPU_CASTABLE
                ):
                    t = t.to(dtype=dtype)
                result[key] = t.to(device=device)

        return result

    def _read_one(self, f, filepath, header, data_base, key) -> torch.Tensor:
        meta      = header[key]
        dtype_str = meta["dtype"]
        shape     = meta["shape"]
        start, _  = meta["data_offsets"]

        np_dtype = DTYPE_MAP[dtype_str]
        count    = int(np.prod(shape)) if shape else 1
        nbytes   = count * np.dtype(np_dtype).itemsize

        # Fresh bytearray per tensor — no clone() needed, halves peak RAM.
        buf  = bytearray(nbytes)
        view = memoryview(buf)
        f.seek(data_base + start)
        n = f.readinto(view)
        if n != nbytes:
            raise ValueError(
                f"Short read for '{key}' in {filepath}: "
                f"expected {nbytes} B, got {n} B"
            )
        return _to_tensor(view, dtype_str, shape)

    def _read_coalesced(self, f, filepath, header, data_base, keys, result) -> None:
        """Read *keys* as a few large spans instead of one seek per tensor."""
        entries = []
        for key in keys:
            start, end = header[key]["data_offsets"]
            entries.append((start, end, key))
        entries.sort()

        group: List[tuple] = []
        group_start = group_end = 0

        def _flush() -> None:
            if not group:
                return
            span = group_end - group_start
            buf = bytearray(span)
            f.seek(data_base + group_start)
            n = f.readinto(memoryview(buf))
            if n != span:
                raise ValueError(
                    f"Short read in {filepath}: expected {span} B, got {n} B")
            mv = memoryview(buf)
            for start, end, key, dtype_str, shape in group:
                result[key] = _to_tensor(mv[start - group_start:end - group_start], dtype_str, shape)
            group.clear()

        for start, end, key in entries:
            meta = header[key]
            item = (start, end, key, meta["dtype"], meta["shape"])
            if group and (start - group_end > _COALESCE_GAP or end - group_start > _COALESCE_MAX_SPAN):
                _flush()
            if not group:
                group_start, group_end = start, end
            group.append(item)
            group_end = max(group_end, end)
        _flush()

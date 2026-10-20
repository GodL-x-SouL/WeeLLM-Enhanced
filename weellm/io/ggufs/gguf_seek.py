"""
gguf_seek.py -- GGUF tensor reader for WeeLLM layer-streaming.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from weellm.io.ggufs.keymaps import build_remap_fn
from weellm.io.ram_cache import get_raw_cache

logger = logging.getLogger("weellm")



class GGUFSeeker:
    # Enables quantized-bytes RAM staging + GPU dequant in the H2D worker.
    supports_staged_prefetch = True

    def __init__(self, gguf_path: Path) -> None:
        try:
            import gguf as _gguf_lib
        except ImportError:
            raise ImportError("gguf>=0.13.0 is required.")

        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf_path}")

        self._cache_ns = id(self)

        logger.info("[GGUFSeeker] Opening %s ...", self.gguf_path.name)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self._reader = _gguf_lib.GGUFReader(str(self.gguf_path))

        self._tensor_meta: Dict[str, Tuple] = {}
        raw_names = [t.name for t in self._reader.tensors]
        
        self.is_flux_format = any(k.startswith("double_blocks.") for k in raw_names)
        arch = self._get_arch()
        remap, self.keymap_cls = build_remap_fn(raw_names, arch)

        for tensor in self._reader.tensors:
            orig_shape = self._get_orig_shape(tensor.name)
            if orig_shape is None:
                orig_shape = tuple(int(v) for v in reversed(tensor.shape))

            diffusers_entries = remap(tensor.name)
            for diffusers_name, slice_info in diffusers_entries:
                self._tensor_meta[diffusers_name] = (
                    tensor.tensor_type,
                    orig_shape,
                    tensor.data,
                    slice_info,
                    tensor.name,  # keep original name for cache keying
                )

        self.weight_map: Dict[str, str] = {k: self.gguf_path.name for k in self._tensor_meta}
        logger.info("[GGUFSeeker] Loaded %d tensors (arch=%s)", len(self._tensor_meta), arch)

    def patch_model(self, model: torch.nn.Module, config, device: str, dtype: torch.dtype) -> None:
        """Allow the detected keymap to perform any GGUF-specific structural patches or tensor injections."""
        if getattr(self, "keymap_cls", None) and hasattr(self.keymap_cls, "patch_model_before_stream"):
            self.keymap_cls.patch_model_before_stream(model, config, device, dtype, seeker=self)

    def _get_arch(self) -> str:
        try:
            import gguf as _gguf_lib
            field = self._reader.get_field("general.architecture")
            if field is not None and field.types:
                if field.types[0] == _gguf_lib.GGUFValueType.STRING:
                    return str(field.parts[field.data[-1]], encoding="utf-8")
        except Exception:
            pass
        return "unknown"

    def _get_orig_shape(self, tensor_name: str):
        try:
            import gguf as _gguf_lib
            field_key = f"comfy.gguf.orig_shape.{tensor_name}"
            field = self._reader.get_field(field_key)
            if field is None:
                return None
            if (
                len(field.types) == 2
                and field.types[0] == _gguf_lib.GGUFValueType.ARRAY
                and field.types[1] == _gguf_lib.GGUFValueType.INT32
            ):
                return tuple(int(field.parts[part_idx][0]) for part_idx in field.data)
        except Exception:
            pass
        return None

    def get_tensors(
        self,
        keys: List[str],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        import gguf as _gguf_lib
        from weellm.io.ggufs.gguf_dequant import dequantize_tensor, TORCH_COMPATIBLE_QTYPES

        result: Dict[str, torch.Tensor] = {}
        target_dtype = dtype if dtype is not None else torch.bfloat16

        # Cache dequantized tensors so fused QKV keys share one decode pass
        _dequantized_cache: Dict[str, torch.Tensor] = {}

        for key in keys:
            if key not in self._tensor_meta:
                raise KeyError(f"Tensor '{key}' not found in GGUF file.")

            qtype, shape, raw_data_np, slice_info, orig_name = self._tensor_meta[key]

            if orig_name in _dequantized_cache:
                t = _dequantized_cache[orig_name]
            else:
                raw_torch = self._get_raw_tensor(orig_name, raw_data_np)

                if qtype in TORCH_COMPATIBLE_QTYPES:
                    if qtype == _gguf_lib.GGMLQuantizationType.F32:
                        t = raw_torch.view(torch.float32).reshape(shape)
                    elif qtype == _gguf_lib.GGMLQuantizationType.F16:
                        t = raw_torch.view(torch.float16).reshape(shape)
                    else:
                        t = raw_torch.reshape(shape)
                    if t.is_floating_point() and t.dtype != target_dtype:
                        t = t.to(target_dtype)
                else:
                    # Offload dequantization to GPU to accelerate it
                    temp_device = "cuda" if torch.cuda.is_available() else device
                    t = dequantize_tensor(
                        raw_torch.to(temp_device, non_blocking=True),
                        qtype,
                        shape,
                        dtype=target_dtype,
                    )
                    if device == "cpu" and temp_device != "cpu":
                        t = t.to("cpu")  # synchronous — prevents H2D race condition

                _dequantized_cache[orig_name] = t

            if slice_info is not None:
                split_idx, total_splits = slice_info
                chunk_size = t.shape[0] // total_splits
                t = t[split_idx * chunk_size : (split_idx + 1) * chunk_size, ...]
                t = t.clone()  # release reference to the full cached tensor

            if getattr(self, "keymap_cls", None) and hasattr(self.keymap_cls, "postprocess_tensor"):
                t = self.keymap_cls.postprocess_tensor(key, t, orig_name)

            result[key] = t.to(device=device)

        return result

    def _get_raw_tensor(self, orig_name: str, raw_data_np, use_cache: bool = True) -> torch.Tensor:
        """Return the raw quantized bytes as a CPU uint8 tensor.

        With the RAM cache enabled the bytes are copied once into anonymous
        memory (so the OS can reclaim the mmap pages) and reused across
        denoising steps. With the cache disabled this stays a zero-copy view
        into the GGUF mmap, matching the historical behaviour.
        """
        cache = get_raw_cache()
        if use_cache and cache.enabled:
            cached = cache.get(self._cache_ns, orig_name)
            if cached is not None:
                return cached

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(raw_data_np)

        if use_cache and cache.enabled:
            raw = raw.clone()
            cache.put(self._cache_ns, orig_name, raw)
        return raw

    # ------------------------------------------------------------------
    # Staged prefetch: raw bytes to CPU RAM, dequant on GPU in the H2D worker
    # ------------------------------------------------------------------

    def get_staged(self, keys: List[str], use_cache: bool = True) -> Dict[str, Tuple]:
        """Read raw quantized bytes only — no dequantization, no GPU work."""
        staged: Dict[str, Tuple] = {}
        for key in keys:
            if key not in self._tensor_meta:
                raise KeyError(f"Tensor '{key}' not found in GGUF file.")
            qtype, shape, raw_data_np, slice_info, orig_name = self._tensor_meta[key]
            staged[key] = (
                self._get_raw_tensor(orig_name, raw_data_np, use_cache=use_cache),
                qtype,
                shape,
                slice_info,
                orig_name,
            )
        return staged

    def resolve_staged(
        self,
        staged: Dict[str, Tuple],
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Upload raw bytes and dequantize on *device* (call inside H2D stream)."""
        import gguf as _gguf_lib
        from weellm.io.ggufs.gguf_dequant import dequantize_tensor, TORCH_COMPATIBLE_QTYPES

        target_dtype = dtype if dtype is not None else torch.bfloat16
        result: Dict[str, torch.Tensor] = {}
        dequantized: Dict[str, torch.Tensor] = {}

        for key, (raw_cpu, qtype, shape, slice_info, orig_name) in staged.items():
            t = dequantized.get(orig_name)
            if t is None:
                raw = raw_cpu.to(device, non_blocking=True)
                if qtype in TORCH_COMPATIBLE_QTYPES:
                    if qtype == _gguf_lib.GGMLQuantizationType.F32:
                        t = raw.view(torch.float32).reshape(shape)
                    elif qtype == _gguf_lib.GGMLQuantizationType.F16:
                        t = raw.view(torch.float16).reshape(shape)
                    else:
                        t = raw.reshape(shape)
                    if t.is_floating_point() and t.dtype != target_dtype:
                        t = t.to(target_dtype)
                else:
                    t = dequantize_tensor(raw, qtype, shape, dtype=target_dtype)
                dequantized[orig_name] = t

            if slice_info is not None:
                split_idx, total_splits = slice_info
                chunk_size = t.shape[0] // total_splits
                t = t[split_idx * chunk_size : (split_idx + 1) * chunk_size, ...].clone()

            if getattr(self, "keymap_cls", None) and hasattr(self.keymap_cls, "postprocess_tensor"):
                t = self.keymap_cls.postprocess_tensor(key, t, orig_name)

            result[key] = t
        return result

    def staged_block_bytes(self, keys: List[str]) -> int:
        return self.get_block_bytes(keys, raw=True)

    def get_block_bytes(self, keys: List[str], raw: bool = False) -> int:
        import math
        total = 0
        seen_raw = set()
        for key in keys:
            if key not in self._tensor_meta:
                continue
            qtype, shape, raw_data_np, slice_info, orig_name = self._tensor_meta[key]
            if raw:
                # Fused QKV keys share one on-disk tensor; count it once.
                if orig_name in seen_raw:
                    continue
                seen_raw.add(orig_name)
                total += int(raw_data_np.nbytes)
                continue
            n_elements = math.prod(shape) if shape else 1
            if slice_info is not None:
                n_elements = n_elements // slice_info[1]
            total += n_elements * 2
        return total

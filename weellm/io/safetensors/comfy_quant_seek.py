"""
comfy_quant_seek.py -- Disk streaming reader for ComfyUI-style quantized
safetensors (INT8 + W4A8 mixed with full-precision tensors).

Implements the same duck-typed interface as SafetensorsDiskSeeker /
GGUFSeeker (``weight_map``, ``get_tensors``, ``get_block_bytes``) so every
existing streamer works unchanged: quantized groups decode to plain
floating tensors at load time (see ``weellm.io.comfy_quant``), while the
full-precision layers in the same file stream exactly as before.

Key design point for mixed files: ``weight_map`` exposes ONLY logical
model keys (``<prefix>.weight`` for quantized layers, plus every plain
tensor). Scale/codebook companions (``weight_s_rel``, ``weight_s_channel``,
``weight_codebook``, ``weight_scale``, ``comfy_quant``) are pulled
automatically inside ``get_tensors`` and never appear in ``weight_map``,
so layer-key prefix matching, resident filters and ``place_tensors`` all
see a normal full-precision checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from weellm.io.comfy_quant import (
    classify_quant_groups,
    dequantize_int4,
    dequantize_int8,
    dequantize_w4a8,
    strip_comfy_prefix,
)
from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase

logger = logging.getLogger("weellm")


class ComfyQuantSeeker(SafetensorsBase):
    """Disk-based reader with on-the-fly INT8/W4A8 dequantization."""

    def __init__(self, model_dir: Union[str, Path]):
        super().__init__(model_dir)
        self._parse_index()
        # Normalize Comfy single-file prefixes (model.diffusion_model.*) to
        # the diffusers namespace streamers match on. _raw_map translates
        # logical keys back to on-disk names for reading.
        raw_map: Dict[str, str] = {}
        stripped: Dict[str, str] = {}
        for k, v in self.weight_map.items():
            sk = strip_comfy_prefix(k)
            stripped[sk] = v
            raw_map[sk] = k
        self.weight_map = stripped
        self._raw_map = raw_map
        # {logical_key: spec}; spec holds kind + companion key names + files.
        self.groups: Dict[str, dict] = {}
        # companion keys (scales/codebooks/blobs) hidden from weight_map.
        companions: set[str] = set()
        for shard in sorted(set(self.weight_map.values())):
            header, _ = self._read_header(self.model_dir / shard)
            self._apply_file_metadata(header)
            for logical_raw, spec in classify_quant_groups(header).items():
                logical = strip_comfy_prefix(logical_raw)
                spec = {kk: (strip_comfy_prefix(vv) if isinstance(vv, str) else vv) for kk, vv in spec.items()}
                spec = dict(spec)
                spec["file"] = self.weight_map.get(logical, shard)
                for ck in ("s_rel", "s_channel", "codebook", "scale"):
                    if spec.get(ck):
                        spec[ck + "_file"] = self.weight_map.get(spec[ck], spec["file"])
                        companions.add(spec[ck])
                # Optional comfy_quant JSON blob (INT8 convrot flags, W4A8 sizes).
                blob_key = logical[: -len(".weight")] + ".comfy_quant"
                if blob_key in self.weight_map:
                    companions.add(blob_key)
                    spec["blob"] = blob_key
                    spec["blob_file"] = self.weight_map.get(blob_key, spec["file"])
                    self._apply_blob(spec)
                self.groups[logical] = spec
        file_layers = getattr(self, "_file_layers", {})
        if file_layers:
            for logical, spec in self.groups.items():
                if spec.get("convrot") and spec["kind"] == "int8":
                    continue
                prefix = logical[: -len(".weight")]
                for layer, info in file_layers.items():
                    if not isinstance(info, dict):
                        continue
                    if prefix == layer or prefix.endswith("." + layer) or prefix.endswith("/" + layer):
                        fmt = str(info.get("format", "")).lower()
                        if fmt == "convrot_w4a4" and spec["kind"] == "int8":
                            spec["kind"] = "int4"
                        if spec["kind"] == "int4":
                            spec["convrot_groupsize"] = int(info.get("convrot_groupsize", 256))
                            spec["quant_group_size"] = int(info.get("quant_group_size", 64))
                        elif spec["kind"] == "int8" and info.get("convrot", False):
                            spec["convrot"] = True
                            spec["convrot_groupsize"] = int(info.get("convrot_groupsize", 256))
                        break
        if companions:
            self.weight_map = {k: v for k, v in self.weight_map.items() if k not in companions}
        n_plain = len(self.weight_map) - len(self.groups)
        n_i8cr = sum(
            1 for s in self.groups.values() if s["kind"] == "int8" and s.get("convrot")
        )
        n_i4 = sum(1 for s in self.groups.values() if s["kind"] == "int4")
        logger.info(
            "[ComfyQuantSeeker] %s: %d quantized (%d w4a8 / %d int8, of which %d convrot / %d int4) + %d plain tensors.",
            self.model_dir.name if isinstance(self.model_dir, Path) else self.model_dir,
            len(self.groups),
            sum(1 for s in self.groups.values() if s["kind"] == "w4a8"),
            sum(1 for s in self.groups.values() if s["kind"] == "int8"),
            n_i8cr,
            n_i4,
            n_plain,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_file_metadata(self, header: dict) -> None:
        """Remember ``_quantization_metadata`` (header ``__metadata__``).

        Some writers (e.g. official INT8-ConvRot packs) describe per-layer
        formats here instead of per-layer ``comfy_quant`` blobs. Stored for
        :meth:`_apply_blob`-style refinement after all groups are known.
        """
        try:
            meta = header.get("__metadata__", {}) or {}
            blob = meta.get("_quantization_metadata")
            if not blob:
                return
            info = json.loads(blob) if isinstance(blob, str) else blob
            layers = info.get("layers", {}) if isinstance(info, dict) else {}
            if layers:
                if not hasattr(self, "_file_layers"):
                    self._file_layers = {}
                self._file_layers.update(layers)
        except Exception as e:
            logger.debug("[ComfyQuantSeeker] unreadable _quantization_metadata: %s", e)

    def _apply_blob(self, spec: dict) -> None:
        """Refine a group spec from its ``comfy_quant`` JSON blob, if readable."""
        try:
            header, data_base = self._read_header(
                self.model_dir / spec.get("blob_file", spec["file"])
            )
            raw = self._read_raw(header, data_base, self.model_dir / spec.get("blob_file", spec["file"]), spec["blob"])
            blob = json.loads(raw.numpy().tobytes().decode("utf-8").rstrip("\x00"))
        except Exception as e:
            logger.debug("[ComfyQuantSeeker] unreadable comfy_quant blob: %s", e)
            return
        if not isinstance(blob, dict):
            return
        # Writers put tunables either top-level or under "params" (Comfy
        # prefers top-level, falls back to nested) -- merge the same way.
        nested = blob.get("params", {})
        if not isinstance(nested, dict):
            nested = {}
        conf = {**nested, **blob}
        fmt = str(conf.get("format", "")).lower()
        if fmt == "convrot_w4a4" and spec["kind"] == "int8":
            # Same .weight_scale companions as INT8 -- the blob decides.
            spec["kind"] = "int4"
        if spec["kind"] == "int8":
            if conf.get("convrot", False):
                spec["convrot"] = True
            spec["convrot_groupsize"] = int(conf.get("convrot_groupsize", 256))
        elif spec["kind"] == "w4a8":
            spec["group_size"] = int(conf.get("group_size", spec.get("group_size", 16)))
            spec["convrot_groupsize"] = int(
                conf.get("convrot_groupsize", spec.get("convrot_groupsize", 256))
            )
        elif spec["kind"] == "int4":
            spec["convrot_groupsize"] = int(
                conf.get("convrot_groupsize", spec.get("convrot_groupsize", 256))
            )
            spec["quant_group_size"] = int(
                conf.get("quant_group_size", spec.get("quant_group_size", 64))
            )

    def _raw(self, key: str) -> str:
        """Logical (diffusers-namespace) key -> on-disk safetensors key."""
        return self._raw_map.get(key, key)

    def _read_raw(self, header: dict, data_base: int, filepath: Path, key: str) -> torch.Tensor:
        """Read one raw tensor (no dtype cast) from an open header context."""
        meta = header[self._raw(key)]
        dtype_str, shape = meta["dtype"], meta["shape"]
        start, _ = meta["data_offsets"]
        np_dtype = DTYPE_MAP[dtype_str]
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * np.dtype(np_dtype).itemsize
        buf = bytearray(nbytes)
        view = memoryview(buf)
        with open(filepath, "rb") as f:
            f.seek(data_base + start)
            n = f.readinto(view)
        if n != nbytes:
            raise ValueError(f"Short read for '{key}': expected {nbytes} B, got {n} B")
        arr = np.frombuffer(view, dtype=np_dtype)
        if shape:
            arr = arr.reshape(shape)
        t = torch.from_numpy(arr)
        if dtype_str == "BF16":
            t = t.view(torch.bfloat16)
        elif dtype_str == "F8_E4M3":
            t = t.view(torch.float8_e4m3fn)
        return t

    def _load_piece(self, spec: dict, role: str, default_file: str) -> torch.Tensor:
        """Read one raw tensor of a quant group, following cross-shard files."""
        piece_key = spec[role]
        src = spec["file"] if role == "weight" else spec.get(role + "_file", default_file)
        header, data_base = self._read_header(self.model_dir / src)
        return self._read_raw(header, data_base, self.model_dir / src, piece_key)

    def _logical_shape(self, key: str) -> tuple:
        spec = self.groups[key]
        if spec["kind"] in ("w4a8", "int4"):
            header, _ = self._read_header(self.model_dir / spec["file"])
            stored = tuple(header[self._raw(spec["weight"])]["shape"])
            return (stored[0], stored[1] * 2)
        header, _ = self._read_header(self.model_dir / spec["file"])
        return tuple(header[self._raw(spec["weight"])]["shape"])

    # ------------------------------------------------------------------
    # Seeker interface
    # ------------------------------------------------------------------

    def get_tensors(
        self,
        keys: List[str],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        target = dtype if dtype is not None else torch.bfloat16
        by_src: Dict[str, List[str]] = {}
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Tensor '{key}' not found in model index.")
            by_src.setdefault(self.weight_map[key], []).append(key)

        result: Dict[str, torch.Tensor] = {}
        for src_file, src_keys in by_src.items():
            filepath = self.model_dir / src_file
            header, data_base = self._read_header(filepath)
            for key in src_keys:
                spec = self.groups.get(key)
                if spec is None:
                    t = self._read_raw(header, data_base, filepath, key)
                    t = t.to(device=device)
                    if t.is_floating_point() and t.dtype != target:
                        t = t.to(dtype=target)
                    result[key] = t
                    continue
                if spec["kind"] == "w4a8":
                    q = self._load_piece(spec, "weight", src_file)
                    s_rel = self._load_piece(spec, "s_rel", src_file)
                    if s_rel.dtype == torch.uint8:
                        # Some writers store fp8 bytes as U8 (Comfy does the
                        # same view() in its loader).
                        s_rel = s_rel.view(torch.float8_e4m3fn)
                    s_ch = self._load_piece(spec, "s_channel", src_file)
                    cb = None
                    if spec.get("codebook"):
                        try:
                            cb = self._load_piece(spec, "codebook", src_file)
                        except (KeyError, ValueError):
                            cb = None
                    t = dequantize_w4a8(
                        q,
                        s_rel.float() if s_rel.dtype == torch.float8_e4m3fn else s_rel,
                        s_ch,
                        codebook=cb,
                        group_size=spec.get("group_size", 16),
                        convrot_groupsize=spec.get("convrot_groupsize", 256),
                        dtype=target,
                    )
                elif spec["kind"] == "int4":
                    q = self._load_piece(spec, "weight", src_file)
                    sc = self._load_piece(spec, "scale", src_file)
                    t = dequantize_int4(
                        q,
                        sc,
                        convrot_groupsize=spec.get("convrot_groupsize", 256),
                        quant_group_size=spec.get("quant_group_size", 64),
                        dtype=target,
                    )
                else:  # int8
                    q = self._load_piece(spec, "weight", src_file)
                    sc = self._load_piece(spec, "scale", src_file)
                    t = dequantize_int8(
                        q,
                        sc,
                        convrot=spec.get("convrot", False),
                        convrot_groupsize=spec.get("convrot_groupsize", 256),
                        dtype=target,
                    )
                result[key] = t.to(device=device)
        return result

    def get_block_bytes(self, keys: List[str]) -> int:
        """Logical (dequantized, bf16-equivalent) footprint for budgeting."""
        total = 0
        for key in keys:
            if key in self.groups:
                shape = self._logical_shape(key)
                n = 1
                for d in shape:
                    n *= d
                total += n * 2
                continue
            if key not in self.weight_map:
                continue
            header, _ = self._read_header(self.model_dir / self.weight_map[key])
            if key not in header or key == "__metadata__":
                continue
            meta = header[key]
            shape = meta.get("shape", [])
            dstr = meta.get("dtype", "F32")
            n = int(np.prod(shape)) if shape else 1
            total += n * np.dtype(DTYPE_MAP[dstr]).itemsize
        return total

"""
ideogram4_transformer.py -- Hook-based layer-streaming for Ideogram 4 diffusion transformer.

Uses the Double-Stream buffer overlap to load the massive 34 FP8 blocks directly from the SSD
while maintaining under 4.0 GB VRAM footprint.

Note: Ideogram4 uses FP8 weight dequantization (weight * weight_scale). We override
`apply_state_dict` from BaseTransformerStreamer to perform the FP8 math on-the-fly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
from weellm.io.seeker import get_seeker
from weellm.io.utils import clean_memory, report_memory

logger = logging.getLogger("weellm")


class Ideogram4Transformer2DModelStreamer(BaseTransformerStreamer):
    """
    Hook-based streaming wrapper for Ideogram4Transformer2DModel.
    Loads 34 FP8-quantized blocks one at a time from SSD, dequantizes them
    (weight * weight_scale → bfloat16) before placing on GPU, and evicts
    immediately after the forward pass.
    """

    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        order = []
        blocks = getattr(self.model, "layers", getattr(self.model, "transformer_blocks", []))
        prefix = "layers" if hasattr(self.model, "layers") else "transformer_blocks"
        for i, block in enumerate(blocks):
            order.append((f"{prefix}.{i}", block))
        return order

    def _get_resident_keys(self) -> List[str]:
        expected_keys = set(self.model.state_dict().keys())
        return [
            k for k in self.seeker.weight_map
            if k in expected_keys and not k.startswith("layers.") and not k.startswith("transformer_blocks.")
        ]

    @staticmethod
    def _assert_split_attention_layout(model: nn.Module) -> None:
        """Fail fast when diffusers still exposes the legacy fused Ideogram4 attention.

        The Ideogram4 GGUF key-map (``weellm.io.ggufs.keymaps.transformers.ideogram4``)
        rewrites the checkpoint's fused ``attention.qkv`` / ``attention.o`` tensors into
        the canonical ``attention.to_q`` / ``to_k`` / ``to_v`` / ``to_out.0`` projections
        that current diffusers expects.  Against the legacy fused module the load would
        die deep inside accelerate with a cryptic
        ``AttributeError: 'Ideogram4Attention' object has no attribute 'o'``.
        """
        blocks = getattr(model, "layers", None) or getattr(model, "transformer_blocks", None)
        if not blocks:
            return
        attn = getattr(blocks[0], "attention", None)
        if attn is None:
            return
        if hasattr(attn, "to_q") and hasattr(attn, "to_out"):
            return
        raise RuntimeError(
            "The installed diffusers exposes the legacy fused Ideogram4 attention "
            f"(attributes: {sorted(a for a in dir(attn) if a in ('qkv', 'o', 'to_q', 'to_out'))}). "
            "WeeLLM maps the Ideogram4 GGUF checkpoint onto the canonical split layout "
            "(attention.to_q/to_k/to_v + attention.to_out.0). "
            "Upgrade diffusers: pip install -U git+https://github.com/huggingface/diffusers.git"
        )

    def apply_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Override to apply FP8 dequantization (weight * scale) before placing on GPU.

        The official FP8 checkpoint keeps the *original* Ideogram attention layout
        (a fused ``attention.qkv`` linear plus an ``attention.o`` linear), while
        diffusers' ``Ideogram4Attention`` uses the canonical split layout
        (``to_q`` / ``to_k`` / ``to_v`` + ``to_out.0``).  The fused QKV rows are
        contiguous ([q_all | k_all | v_all] -- the same layout the GGUF key-map
        assumes), and the FP8 scale is per output row, so it splits along with
        the weight.
        """
        processed_sd = {}
        scales = {k: v for k, v in state_dict.items() if k.endswith(".weight_scale")}

        # ---- 1. re-key the original Ideogram layout onto diffusers' split one
        for name, tensor in state_dict.items():
            if name.endswith(".weight_scale"):
                continue

            # Fused QKV -> to_q / to_k / to_v (+ the matching per-row scale)
            if name.endswith(".attention.qkv.weight"):
                prefix = name[: -len(".attention.qkv.weight")]
                third = tensor.shape[0] // 3
                scale = scales.get(name + "_scale")
                for i, part in enumerate(("to_q", "to_k", "to_v")):
                    processed_sd[prefix + ".attention." + part + ".weight"] = \
                        tensor[i * third:(i + 1) * third].contiguous()
                    if scale is not None:
                        processed_sd[prefix + ".attention." + part + ".weight_scale"] = \
                            scale[i * third:(i + 1) * third].contiguous()
                continue

            # Fused output projection -> to_out.0
            if name.endswith(".attention.o.weight"):
                prefix = name[: -len(".attention.o.weight")]
                processed_sd[prefix + ".attention.to_out.0.weight"] = tensor
                scale = scales.get(name + "_scale")
                if scale is not None:
                    processed_sd[prefix + ".attention.to_out.0.weight_scale"] = scale
                continue

            processed_sd[name] = tensor

        # ---- 2. FP8 dequantization: weight * weight_scale (per output row)
        final_sd = {}
        for name, tensor in processed_sd.items():
            if name.endswith(".weight_scale"):
                continue
            scale = processed_sd.get(name + "_scale")
            if scale is not None:
                scale = scale.to(device=tensor.device, dtype=torch.float32)
                if scale.dim() == 1:
                    if scale.numel() == tensor.shape[0]:
                        scale = scale.view(-1, 1)
                    elif len(tensor.shape) > 1 and scale.numel() == tensor.shape[1]:
                        scale = scale.view(1, -1)
                tensor = (tensor.to(torch.float32) * scale).to(self.dtype)
            final_sd[name] = tensor

        # ---- 3. place
        for name, tensor in final_sd.items():
            if tensor.is_floating_point():
                set_module_tensor_to_device(
                    self.model, name, self.device, value=tensor, dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(self.model, name, self.device, value=tensor)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        cache_to_ram: bool = False,
    ) -> "Ideogram4Transformer2DModelStreamer":
        import diffusers
        if not hasattr(diffusers, "Ideogram4Transformer2DModel"):
            raise ImportError(
                "Diffusers does not have Ideogram4Transformer2DModel. "
                "Please update diffusers: pip install -U diffusers"
            )

        model_dir = Path(model_dir)

        logger.info("Step 1/3 -- Initializing LiveSeeker on Ideogram4 Transformer weights ...")
        seeker = get_seeker(model_dir, cache_to_ram=cache_to_ram)
        logger.info("  Found %d tensors across HF shards.", len(seeker.weight_map))

        logger.info("Instantiating Ideogram4Transformer2DModel on meta device ...")
        from accelerate import init_empty_weights
        from weellm.io.utils import default_dtype
        config = diffusers.Ideogram4Transformer2DModel.load_config(str(model_dir))
        with default_dtype(dtype), init_empty_weights():
            model = diffusers.Ideogram4Transformer2DModel.from_config(config)
        model.eval()

        cls._assert_split_attention_layout(model)

        logger.info("Step 3/3 -- Loading resident transformer tensors to GPU ...")
        streamer = cls(model=model, seeker=seeker, device=device, dtype=dtype, prefetch=prefetch)
        
        for name, buf in list(model.named_buffers()):
            if buf.device.type != "meta":
                set_module_tensor_to_device(
                    model, name, device,
                    value=buf.to(device, dtype=dtype if buf.is_floating_point() else None),
                )

        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        streamer.apply_state_dict(resident_sd)
        del resident_sd
        clean_memory(device)
        report_memory("After resident load")

        logger.info(
            "Installed %d transformer blocks for streaming.",
            len(streamer._get_shard_order())
        )
        logger.info("Ideogram4Transformer2DModelStreamer ready. Mode: Live Seek from original shards")
        return streamer

"""
ideogram4.py -- GGUF key map for Ideogram 4 (Ideogram4Transformer2DModel).

Detected by: "layers.0.attention.qkv.weight" (fused QKV) together with
"layers.0.attention.o.weight" (fused output projection).

Why this map exists
-------------------
Ideogram 4 checkpoints ship the *original* Ideogram attention layout, where
every transformer block owns two fused linears:

    layers.N.attention.qkv.weight   -> [3 * hidden, hidden]
    layers.N.attention.o.weight     -> [hidden, hidden]

Diffusers' ``Ideogram4Attention`` uses the canonical split layout instead:

    layers.N.attention.to_q.weight      [hidden, hidden]
    layers.N.attention.to_k.weight      [hidden, hidden]
    layers.N.attention.to_v.weight      [hidden, hidden]
    layers.N.attention.to_out.0.weight  [hidden, hidden]

Without this map the GGUF names are passed through unchanged and the loader
tries to place "layers.N.attention.o.weight" into a module that has no "o"
attribute -- which surfaces as:

    AttributeError: 'Ideogram4Attention' object has no attribute 'o'

Layout of the fused QKV
-----------------------
The fused rows are contiguous ``[Q | K | V]`` -- exactly the layout the
original diffusers implementation consumed::

    qkv = attn.qkv(hidden_states).view(B, L, 3, num_heads, head_dim)
    query, key, value = qkv.unbind(dim=2)

so slicing the output dimension into three equal row blocks restores the
canonical to_q / to_k / to_v projections.

Every other tensor in the file (input_proj, llm_cond_*, t_embedding.*,
adaln_proj, embed_image_indicator, final_layer.*, attention_norm*/ffn_norm*,
feed_forward.w1/w2/w3, attention.norm_q/norm_k, adaln_modulation) already
matches the diffusers naming one-to-one and is passed through untouched.
"""

from typing import Any, Dict, List


class Ideogram4KeyMap:
    NAME = "ideogram4"

    # ── detection ────────────────────────────────────────────────────────────
    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        has_fused_qkv = any(k.startswith("layers.0.attention.qkv.") for k in gguf_keys)
        has_fused_o = any(k.startswith("layers.0.attention.o.") for k in gguf_keys)
        has_swiglu = any(k.startswith("layers.0.feed_forward.w1.") for k in gguf_keys)
        has_rope = "embed_image_indicator.weight" in gguf_keys
        return has_fused_qkv and has_fused_o and has_swiglu and has_rope

    # ── remapping ────────────────────────────────────────────────────────────
    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap: Dict[str, Any] = {}
        for name in gguf_keys:
            remap[name] = Ideogram4KeyMap._remap_one(name)
        return remap

    @staticmethod
    def _remap_one(name: str) -> List:
        # Fused QKV -> split to_q / to_k / to_v (contiguous row slices).
        if ".attention.qkv." in name:
            suffix = name.rsplit(".", 1)[-1]           # "weight" | "bias"
            base = name[: -len(f".qkv.{suffix}")]      # "...layers.N.attention"
            return [
                (f"{base}.to_q.{suffix}", (0, 3)),
                (f"{base}.to_k.{suffix}", (1, 3)),
                (f"{base}.to_v.{suffix}", (2, 3)),
            ]

        # Fused output projection -> the ModuleList entry diffusers expects.
        if ".attention.o." in name:
            return [(name.replace(".attention.o.", ".attention.to_out.0."), None)]

        # Alternative spellings seen in community conversions.
        if ".attention.out." in name:
            return [(name.replace(".attention.out.", ".attention.to_out.0."), None)]

        # Everything else is already diffusers-compatible.
        return [(name, None)]

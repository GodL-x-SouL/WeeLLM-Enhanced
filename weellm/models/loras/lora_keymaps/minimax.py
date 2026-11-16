from typing import Set, List, Tuple, Any

class MiniMaxH3LoRAKeyMap:
    NAME = "minimax_h3"

    # Prefixes used by the various MiniMax-H3 LoRA releases (ComfyUI exports,
    # PEFT exports, diffusers-native exports).  They all name the same blocks.
    _STRIP_PREFIXES = ("diffusion_model.", "base_model.model.", "transformer.")

    @classmethod
    def _normalise(cls, key: str) -> str:
        for p in cls._STRIP_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
        return key

    @classmethod
    def detect(cls, lora_keys: Set[str]) -> bool:
        for k in lora_keys:
            n = cls._normalise(k)
            if (n.startswith("token_refiner.blocks.") or n.startswith("blocks.")
                    or n.startswith("transformer_blocks.")
                    or ".mlp.fc1" in n or ".attn.qkv" in n):
                return True
        return False

    @classmethod
    def build_pairs(cls, lora_keys: Set[str], config=None) -> List[Tuple[str, str, List[Tuple[str, Any]]]]:
        pairs = []
        bases = set()
        for k in lora_keys:
            if ".lora_A." in k:
                bases.add(k.split(".lora_A.")[0])

        for base in bases:
            a_key = f"{base}.lora_A.weight"
            b_key = f"{base}.lora_B.weight"
            if a_key not in lora_keys or b_key not in lora_keys:
                continue

            # Original mapping logic from MiniMaxH3LoRALoader, but on the
            # prefix-stripped name so ComfyUI/PEFT/diffusers exports all work.
            name = cls._normalise(base)
            if name.startswith("token_refiner.blocks."):
                target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
            elif name.startswith("blocks."):
                target = name.replace("blocks.", "transformer_blocks.", 1)
            else:
                target = name
            target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear")

            targets = []
            if target.endswith(".attn.qkv_proj") or target.endswith(".attn.qkv"):
                prefix = target.rsplit(".", 1)[0] + "."
                # MiniMax stores the fused QKV PER-HEAD INTERLEAVED
                # ([h0: q k v, h1: q k v, ...]), and so is the LoRA delta.
                # The diffusers weights are contiguous thirds, so the delta has
                # to be de-interleaved first -- the head geometry comes from
                # the model config.
                num_heads = getattr(config, "num_attention_heads", None) if config is not None else None
                head_dim = getattr(config, "attention_head_dim", None) if config is not None else None
                if num_heads and head_dim:
                    for _i, _part in enumerate(("to_q", "to_k", "to_v")):
                        targets.append((
                            f"{prefix}{_part}.weight",
                            ("deinterleave_qkv", num_heads, head_dim, _i),
                        ))
                else:
                    # No config: fall back to contiguous thirds (correct only for
                    # checkpoints that store the fused QKV contiguously).
                    targets.append((f"{prefix}to_q.weight", (0, 3)))
                    targets.append((f"{prefix}to_k.weight", (1, 3)))
                    targets.append((f"{prefix}to_v.weight", (2, 3)))
            elif target.endswith(".mlp.fc1"):
                # We need to split into 2 pieces and reorder them: (value, gate) -> (gate, value)
                # This requires a special slice info or custom logic.
                # In GenericLazyLoRALoader, we can define slice_info = ("chunk_reorder", 2, [1, 0])
                targets.append((target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", ("chunk_reorder", 2, [1, 0])))
            elif target.endswith(".mlp.fc2"):
                targets.append((target.replace(".mlp.fc2", ".ff.net.2") + ".weight", None))
            elif target.endswith(".attn.out_proj"):
                targets.append((target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", None))
            else:
                targets.append((target + ".weight", None))

            pairs.append((a_key, b_key, targets))

        return pairs

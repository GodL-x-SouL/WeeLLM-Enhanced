"""GGUF key map for Flux.2 Klein transformers."""

from typing import Any, Dict, List

from weellm.io.ggufs.keymaps.flux import FluxKeyMap


class Flux2KeyMap(FluxKeyMap):
    NAME = "flux2-klein"

    @staticmethod
    def detect(gguf_keys: List[str], arch: str) -> bool:
        double_blocks = {
            int(parts[1])
            for key in gguf_keys
            if (parts := key.split("."))[:1] == ["double_blocks"]
            and len(parts) > 1
            and parts[1].isdigit()
        }
        single_blocks = {
            int(parts[1])
            for key in gguf_keys
            if (parts := key.split("."))[:1] == ["single_blocks"]
            and len(parts) > 1
            and parts[1].isdigit()
        }
        return (
            arch in ("flux2", "flux2-klein")
            or (
                double_blocks == set(range(8))
                and single_blocks == set(range(24))
                and "time_in.in_layer.weight" in gguf_keys
                and "double_stream_modulation_img.lin.weight" in gguf_keys
            )
        )

    @staticmethod
    def build_remap(gguf_keys: List[str]) -> Dict[str, Any]:
        remap = FluxKeyMap.build_remap(gguf_keys)
        for entries in remap.values():
            for index, (target, slice_info) in enumerate(entries):
                entries[index] = (target.replace("time_text_embed.", "time_guidance_embed."), slice_info)
        return remap

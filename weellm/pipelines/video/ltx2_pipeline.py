"""
weellm.pipelines.video.ltx2
===========================
Custom patched adapter for LTX-2.5 Video Pipeline.
Handles 8k+1 frame math and dummy audio injection.
"""

import math
import logging
import torch
from weellm.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeLTX2Pipeline(WeeVideoPipeline):
    
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from pathlib import Path
        import json
        import torch
        from weellm.weebasepipeline import WeeBasePipeline
        
        model_dir_path = Path(model_dir)
        
        # 1. Inject Dummy Components for LTX specific missing parts to satisfy diffusers __init__
        class DummyComponent:
            def __call__(self, *args, **kwargs): return None
            def predict_num_frames(self, *args, **kwargs): return 121
            def to(self, *args, **kwargs): return self
        kwargs.setdefault("duration_head", DummyComponent())
        kwargs.setdefault("prompt_enhancer", DummyComponent())
        
        # 2. Load extra LTX components if they exist
        index_path = model_dir_path / "model_index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            if "vocoder" in index and "vocoder" not in kwargs:
                try:
                    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
                    device = kwargs.get("device", "cuda")
                    dtype = kwargs.get("torch_dtype", torch.bfloat16)
                    vocoder = LTX2VocoderWithBWE.from_pretrained(
                        model_dir_path / "vocoder", torch_dtype=dtype
                    ).to(device)
                    kwargs["vocoder"] = vocoder
                except Exception as e:
                    logger.warning("Failed to load vocoder: %s", e)
                    kwargs.setdefault("vocoder", DummyComponent())
            else:
                kwargs.setdefault("vocoder", DummyComponent())
            
            if "connectors" in index and "connectors" not in kwargs:
                try:
                    from weellm.models.transformers.ltx2_connectors import LTX2ConnectorsStreamer
                    device = kwargs.get("device", "cuda")
                    dtype = kwargs.get("torch_dtype", torch.bfloat16)
                    cache_to_ram = kwargs.get("cache_to_ram", False)
                    
                    conn = LTX2ConnectorsStreamer.from_pretrained(
                        model_dir_path / "connectors", device=device, dtype=dtype, cache_to_ram=cache_to_ram
                    )
                    conn_model = getattr(conn, "model", getattr(conn, "_model", conn))
                    conn_model = WeeBasePipeline._patch_to(conn_model)
                    conn_model._weellm_streamer = conn
                    kwargs["connectors"] = conn_model
                except Exception as e:
                    logger.warning("Failed to load connectors in WeeLTX2Pipeline: %s", e)
                    
            if "audio_vae" in index and "audio_vae" not in kwargs:
                try:
                    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
                    device = kwargs.get("device", "cuda")
                    dtype = kwargs.get("torch_dtype", torch.bfloat16)
                    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                        model_dir_path / "audio_vae", torch_dtype=dtype
                    ).to(device)
                    kwargs["audio_vae"] = audio_vae
                except Exception as e:
                    logger.warning("Failed to load audio_vae: %s", e)
                    
        pipe = super().from_pretrained(model_dir, **kwargs)
        
        # Override the base pipeline's default VAE chunking because LTX-2.5 is sensitive to it
        if hasattr(pipe._pipeline.vae, "use_framewise_decoding"):
            pipe._pipeline.vae.use_framewise_decoding = False
            logger.warning(
                "\n[WARNING] LTX-2.5 VAE chunking (framewise decoding) has been disabled by default. "
                "Chunking causes severe temporal jitter/seams in LTX-2.5. "
                "Note: This processes all frames at once and requires significantly more memory, "
                "which may trigger system RAM swap on low-VRAM machines."
            )
            
        return pipe

    def __call__(self, prompt: str, **kwargs):
        # 1. Handle LTX-specific 8k+1 frame snapping
        _resolved_num_frames = kwargs.pop("num_frames", None)
        _duration = kwargs.pop("duration", None)
        
        if _resolved_num_frames is None and _duration is None:
            _duration = 5.0
            logger.info("  Auto-defaulting to %.1fs duration for LTX model", _duration)

        if _resolved_num_frames is None and _duration is not None:
            _fps = kwargs.get("fps", 24.0)
            _raw_frames = round(_duration * _fps)
            # LTX-2.5 requires frame count ≡ 1 (mod 8)
            _snapped = max(1, ((_raw_frames - 1 + 4) // 8) * 8 + 1)
            _resolved_num_frames = _snapped
            logger.info(
                "  Duration:  %.1fs @ %.0ffps → %d frames (snapped to 8k+1)",
                _duration, _fps, _resolved_num_frames,
            )
            
        kwargs["num_frames"] = _resolved_num_frames
        
        # 2. Inject Dummy Components for LTX specific missing audio parts
        _CALLABLE_COMPONENT_NAMES = ("vocoder", "duration_head", "prompt_enhancer")
        for _comp_name in _CALLABLE_COMPONENT_NAMES:
            if getattr(self._pipeline, _comp_name, None) is None:
                class DummyComponent:
                    def __call__(self, *args, **kwargs): return None
                    def predict_num_frames(self, *args, **kwargs): return 121
                setattr(self._pipeline, _comp_name, DummyComponent())
                logger.debug(
                    "[WeeLLM] Patched None '%s' with no-op dummy component (LTX adapter).",
                    _comp_name,
                )
                
        # 3. Delegate to the shared generic video generation loop in the base class
        return super().__call__(prompt=prompt, **kwargs)

    def _preprocess_latents_for_decode(self, latents, vae, kwargs):
        """
        LTX-2.5 specific latent preprocessing.
        LTX latents are pre-scaled, so we skip standard scaling/shifting.
        We also handle the optional latent upsampler here.
        """
        latent_upsampler = kwargs.get("latent_upsampler", None)
        if latent_upsampler:
            logger.info("Upsampling latents using %s", latent_upsampler)
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
            
            _dev = latents.device
            _dtype = vae.dtype if vae else latents.dtype
            
            _ups_model = LTX2LatentUpsamplerModel.from_pretrained(latent_upsampler).to(device=_dev, dtype=_dtype)
            with torch.no_grad():
                latents = _ups_model(latents)
            del _ups_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return latents

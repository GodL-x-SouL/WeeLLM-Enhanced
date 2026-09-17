"""
weellm.pipelines.video.ltx2
===========================
Custom patched adapter for LTX-2.5 Video Pipeline.
Handles 8k+1 frame math and dummy audio injection.
"""

import math
import logging
import torch
from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeLTX2Pipeline(WeeVideoPipeline):
    
    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        from pathlib import Path
        import json
        import torch
        from diffusers import LTX2Pipeline
        from weellm.pipelines.weebasepipeline import WeeBasePipeline

        # Force skip downloading/loading the unneeded massive components by injecting a dummy _path.
        # This tells WeeBasePipeline that we are overriding them, so it removes them from HF downloads.
        kwargs.setdefault("prompt_enhancer_path", "DUMMY_SKIP_DOWNLOAD")
        
        # If model_dir is a Hugging Face Repo ID, resolve it to a local path now
        # so that our manual vocoder/connectors loading can find the local files.
        from weellm.io.utils import resolve_model_path
        skip_components = {k[:-5] for k, v in kwargs.items() if k.endswith("_path") and v is not None}
        model_dir = str(resolve_model_path(str(model_dir), skip_components=skip_components or None))
        
        model_dir_path = Path(model_dir)
        device = kwargs.get("device", "cuda")
        dtype  = kwargs.get("torch_dtype", torch.bfloat16)
        
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
                    vocoder = LTX2VocoderWithBWE.from_pretrained(
                        model_dir_path / "vocoder", torch_dtype=dtype
                    ).to(device)
                    kwargs["vocoder"] = vocoder
                except Exception as e:
                    logger.error("Failed to load vocoder: %s", e)
                    raise e
            
            if "connectors" in index and "connectors" not in kwargs:
                try:
                    from weellm.models.transformers.ltx2_connectors import LTX2ConnectorsStreamer
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
                    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                        model_dir_path / "audio_vae", torch_dtype=dtype
                    ).to(device)
                    kwargs["audio_vae"] = audio_vae
                except Exception as e:
                    logger.warning("Failed to load audio_vae: %s", e)
                    
        _temporal = kwargs.pop("temporal_upscaler", None)
        if "temporal_upscaler" in index and _temporal is None:
            try:
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _temporal = LTX2LatentUpsamplerModel.from_pretrained(
                    model_dir_path / "temporal_upscaler", torch_dtype=dtype
                ).to(device)
            except Exception as e:
                logger.warning("Failed to load temporal_upscaler: %s", e)
                
        pipe = super().from_pretrained(model_dir, **kwargs)
        
        if _temporal is not None:
            if isinstance(_temporal, str):
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _temporal = LTX2LatentUpsamplerModel.from_pretrained(_temporal, torch_dtype=dtype).to(device)
            # Attach to the underlying diffusers pipeline so kwargs.get('temporal_upscaler') is not needed
            setattr(pipe._pipeline, "temporal_upscaler", _temporal)
        
        # Override the base pipeline's default VAE chunking because LTX-2.5 is sensitive to it
        if hasattr(pipe._pipeline.vae, "use_framewise_decoding"):
            pipe._pipeline.vae.use_framewise_decoding = False
            logger.warning(
                "\n[WARNING] LTX-2.5 VAE chunking (framewise decoding) has been disabled by default. "
                "Chunking causes severe temporal jitter/seams in LTX-2.5. "
                "Note: This processes all frames at once and requires significantly more memory, "
                "which may trigger system RAM swap on low-VRAM machines."
            )
        if hasattr(pipe._pipeline, "enable_vae_tiling"):
            pipe._pipeline.enable_vae_tiling()
            
        # Gemma 3/4 uses left-padding for prompts. Because it is a causal model, 
        # the padded tokens at the beginning of the sequence have nothing to attend to 
        # (their entire attention row is masked out). When PyTorch calculates Softmax 
        # on an entirely masked row (all -inf), it outputs NaN.
        # These NaNs ONLY exist on the padded tokens. The real tokens are perfectly healthy.
        # The downstream LTX transformer ignores these padded tokens anyway, but the NaNs 
        # trip up the WeeLLM cache corruption check. We simply zero them out.
        _orig_encode = pipe.encode_prompt
        def _safe_encode(*args, **kwargs):
            out = _orig_encode(*args, **kwargs)
            # out is (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)
            clean_out = []
            _target_dtype = getattr(pipe._pipeline, "dtype", dtype)
            for item in out:
                if isinstance(item, torch.Tensor) and torch.is_floating_point(item):
                    clean_item = torch.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0)
                    clean_out.append(clean_item.to(_target_dtype))
                else:
                    clean_out.append(item)
            return tuple(clean_out)
                    
        pipe.encode_prompt = _safe_encode
        
        # 4. Automatically handle Data Type casting for Connectors (Handles Cache Hits)
        if hasattr(pipe._pipeline, "connectors"):
            _orig_conn_forward = pipe._pipeline.connectors.forward
            def _safe_conn_forward(*args, **kwargs):
                _target_dtype = getattr(pipe._pipeline, "dtype", dtype)
                new_args = tuple(a.to(_target_dtype) if isinstance(a, torch.Tensor) and torch.is_floating_point(a) else a for a in args)
                new_kwargs = {k: (v.to(_target_dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v) for k, v in kwargs.items()}
                return _orig_conn_forward(*new_args, **new_kwargs)
            pipe._pipeline.connectors.forward = _safe_conn_forward

        return pipe

    def __call__(self, prompt: str, **kwargs):
        # 0. Handle Pipeline Routing for Image/Video modalities
        _first_frame = kwargs.pop("first_frame", None)
        _last_frame = kwargs.pop("last_frame", None)
        # For backwards compatibility with standard Diffusers 'image'
        _image_fallback = kwargs.pop("image", None)
        if _first_frame is None and _image_fallback is not None:
            _first_frame = _image_fallback
        
        # Remove video kwarg if it somehow gets passed to prevent crashes
        kwargs.pop("video", None)

        if _first_frame is not None:
            _current_class_name = self._pipeline.__class__.__name__
            _model_dir = getattr(self._pipeline, "model_dir", None)
            _safe_encode = getattr(self._pipeline, "encode_prompt", None)
            
            if _last_frame is not None:
                # Both frames provided: Route to InContext Pipeline (Interpolation)
                if "InContext" not in _current_class_name:
                    from diffusers import LTX2InContextPipeline
                    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
                    import inspect
                    logger.info("Routing to LTX2InContextPipeline (Video Interpolation)...")
                    valid_keys = set(inspect.signature(LTX2InContextPipeline.__init__).parameters.keys())
                    safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                    self._pipeline = LTX2InContextPipeline(**safe_components)
                    if _model_dir:
                        self._pipeline.model_dir = _model_dir
                
                # Build the condition objects for the underlying pipeline
                from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
                kwargs["conditions"] = [
                    LTX2VideoCondition(frames=_first_frame, index=0, strength=1.0),
                    LTX2VideoCondition(frames=_last_frame, index=-1, strength=1.0)
                ]
                
            else:
                # Only first frame provided: Route to ImageToVideo Pipeline
                if "ImageToVideo" not in _current_class_name:
                    from diffusers import LTX2ImageToVideoPipeline
                    import inspect
                    logger.info("Routing to LTX2ImageToVideoPipeline (Image-to-Video)...")
                    valid_keys = set(inspect.signature(LTX2ImageToVideoPipeline.__init__).parameters.keys())
                    safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                    self._pipeline = LTX2ImageToVideoPipeline(**safe_components)
                    if _model_dir:
                        self._pipeline.model_dir = _model_dir
                
                kwargs["image"] = _first_frame
                
            if _safe_encode is not None:
                self._pipeline.encode_prompt = _safe_encode
        else:
            # Neither provided: Route back to TextToVideo if needed
            _current_class_name = self._pipeline.__class__.__name__
            if _current_class_name != "LTX2Pipeline":
                from diffusers import LTX2Pipeline
                import inspect
                logger.info("Routing back to LTX2Pipeline (Text-to-Video)...")
                _model_dir = getattr(self._pipeline, "model_dir", None)
                _safe_encode = getattr(self._pipeline, "encode_prompt", None)
                valid_keys = set(inspect.signature(LTX2Pipeline.__init__).parameters.keys())
                safe_components = {k: v for k, v in self._pipeline.components.items() if k in valid_keys}
                self._pipeline = LTX2Pipeline(**safe_components)
                if _model_dir:
                    self._pipeline.model_dir = _model_dir
                if _safe_encode is not None:
                    self._pipeline.encode_prompt = _safe_encode

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
        
        # 2. Inject Dummy Components for LTX specific missing parts
        _CALLABLE_COMPONENT_NAMES = ("duration_head", "prompt_enhancer")
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
        We also handle the optional latent upsampler and temporal upscaler here.
        """
        temporal_upscaler = kwargs.get("temporal_upscaler", getattr(self._pipeline, "temporal_upscaler", None))
        if temporal_upscaler:
            logger.info("Upsampling latents temporally using %s", temporal_upscaler)
            
            _dev = latents.device
            _dtype = vae.dtype if vae else latents.dtype
            
            _ups_model = temporal_upscaler
            if isinstance(_ups_model, str):
                from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
                _ups_model = LTX2LatentUpsamplerModel.from_pretrained(_ups_model).to(device=_dev, dtype=_dtype)
            
            _ups_model = _ups_model.to(device=_dev, dtype=_dtype)
            with torch.no_grad():
                latents = _ups_model(latents)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
        """Intercept diffusers LoRA loading to preserve WeeLLM LiveSeeker hooks."""
        from weellm.models.loras.lora_streamer import GenericLazyLoRALoader
        logger.info(f"[WeeLLM] Intercepted load_lora_weights for {pretrained_model_name_or_path_or_dict}")
        
        lazy_loader = GenericLazyLoRALoader(pretrained_model_name_or_path_or_dict)
        _tr_model = getattr(self._pipeline, "transformer", None)
        if _tr_model is not None:
            lazy_loader.apply_to_module(_tr_model, "transformer")
            
        _conn_model = getattr(self._pipeline, "connectors", None)
        if _conn_model is not None:
            lazy_loader.apply_to_module(_conn_model, "connectors")

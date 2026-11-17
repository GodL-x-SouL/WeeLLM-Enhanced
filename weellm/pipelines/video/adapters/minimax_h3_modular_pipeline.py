"""
weellm.pipelines.video.minimax_h3_modular_pipeline
==================================================
Custom patched adapter for MiniMax-H3 Modular Pipeline.
Uses WeeBasePipeline's automatic interception to route standard
diffusers MiniMax components to WeeLLM layer-by-layer streamers.
"""

import logging
import torch
from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

class WeeMiniMaxPipeline(WeeVideoPipeline):
    
    def __call__(self, prompt: str, **kwargs):
        # MiniMaxH3Scheduler counts SIGMA GRID POINTS, not model evaluations:
        # set_timesteps(n) keeps the terminal 0 and drives n-1 forwards.  The
        # studio (and the 4-step turbo LoRA) mean evaluations, so requesting
        # "4 steps" used to run only 3.  Ask for one extra grid point.
        _steps = kwargs.get("num_inference_steps")
        if _steps is not None:
            try:
                kwargs["num_inference_steps"] = int(_steps) + 1
            except (TypeError, ValueError):
                pass
        # Pick a safe attention path BEFORE the packed sequence is built.
        # bf16 on pre-Ampere GPUs has no memory-efficient SDPA kernel, so the
        # math backend would materialise a [heads, rows, rows] fp32 matrix
        # (77.6 GiB at 960x544 / 124 frames).  The helper switches to a
        # bounded, query-chunked pass when that kernel is unavailable.
        try:
            from weellm.models.transformers.minimax_h3_attention import configure_for_pipeline
            configure_for_pipeline(
                self._pipeline,
                prompt=prompt,
                height=int(kwargs.get("height", 544) or 544),
                width=int(kwargs.get("width", 960) or 960),
                num_frames=int(kwargs.get("num_frames", 121) or 121),
            )
        except Exception as _attn_err:
            logger.warning("[MiniMax-H3] attention preflight failed (%s) — continuing.", _attn_err)

        # 1. Add empty dummy components for audio if missing
        # MiniMax ModularPipeline relies on `audio_vae` for t2va and fl2va,
        # but if we are doing video-only, we might need to bypass it.
        _CALLABLE_COMPONENT_NAMES = ("audio_vae", "audio_scheduler")
        for _comp_name in _CALLABLE_COMPONENT_NAMES:
            if getattr(self._pipeline, _comp_name, None) is None:
                if _comp_name == "audio_scheduler":
                    # Must instantiate a fresh scheduler from config so they don't share state
                    # and don't carry the WeeLLM patched closures from the video scheduler!
                    new_scheduler = self._pipeline.scheduler.__class__.from_config(self._pipeline.scheduler.config)
                    setattr(self._pipeline, _comp_name, new_scheduler)
                else:
                    _dev = getattr(self._pipeline, "device", "cpu")
                    class DummyComponent:
                        config = type('DummyConfig', (), {'latent_channels': 32, 'sampling_rate': 32000})()
                        def __call__(self, *args, **kwargs): return None
                        def encode(self, *args, **kwargs):
                            import torch
                            class Output:
                                latent_dist = type('Dist', (), {'sample': lambda: torch.zeros(1, 32, 1, device=_dev)})()
                            return Output()
                        def decode(self, *args, **kwargs):
                            # Modular MiniMax always decodes audio; a silent
                            # [2, samples] waveform is safer than AttributeError.
                            import torch
                            return (torch.zeros(1, 2, 32000, device=_dev),)
                    setattr(self._pipeline, _comp_name, DummyComponent())
                logger.debug(
                    "[WeeLLM] Patched None '%s' with dummy component.",
                    _comp_name,
                )

        # 2. MiniMax doesn't support guidance_scale since it's guidance-distilled
        kwargs.pop("guidance_scale", None)
        kwargs.pop("negative_prompt", None)
        kwargs.pop("image_guidance_scale", None)
        kwargs.pop("_video_cache", None)
        # The modular pipeline has no callback_on_step_end, but WeeLLM hooks its
        # denoise loop and drives the studio's progress bar from there.
        _step_cb = kwargs.pop("callback_on_step_end", None)
        if _step_cb is not None:
            kwargs["_weellm_step_callback"] = _step_cb
        
        # 3. Delegate to the shared generic video generation loop in the base class
        return super().__call__(prompt=prompt, **kwargs)

    # ── decode hooks ─────────────────────────────────────────────
    # MiniMax-H3 latents are normalized with per-channel `latents_mean` /
    # `latents_std` (NOT scaling_factor/shift_factor), and its VAE decodes
    # into ImageNet-normalized RGB.  The modular pipeline's own decode step
    # does both conversions; when the studio decodes manually (to control
    # VRAM) it has to reproduce them exactly or the video comes out washed
    # out / wrongly coloured.
    def _preprocess_latents_for_decode(self, latents, vae, kwargs):
        cfg = getattr(vae, "config", None)
        mean = getattr(cfg, "latents_mean", None)
        std = getattr(cfg, "latents_std", None)
        if mean is None or std is None:
            return super()._preprocess_latents_for_decode(latents, vae, kwargs)
        try:
            m = torch.tensor(mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            s = torch.tensor(std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
            if m.numel() != latents.shape[1]:
                return super()._preprocess_latents_for_decode(latents, vae, kwargs)
            return (latents.float() * s + m).to(latents.dtype)
        except Exception as e:
            logger.warning("[MiniMax-H3] latent denormalisation skipped (%s).", e)
            return super()._preprocess_latents_for_decode(latents, vae, kwargs)

    def _postprocess_decoded_video(self, video, vae, kwargs):
        try:
            if video.dim() == 5:
                under = getattr(self, "_pipeline", self)
                # The MiniMax processor is built with do_normalize=False and
                # the decode step reverts ImageNet normalization itself.  Only
                # do it here when the processor is not already doing it.
                cfg = getattr(getattr(under, "video_processor", None), "config", None)
                try:
                    do_norm = bool(cfg.get("do_normalize", False)) if cfg is not None else False
                except Exception:
                    do_norm = False
                if not do_norm:
                    mean = getattr(under, "pixel_mean", (0.485, 0.456, 0.406))
                    std = getattr(under, "pixel_std", (0.229, 0.224, 0.225))
                    m = torch.tensor(mean, device=video.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
                    s = torch.tensor(std, device=video.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
                    video = (video.float() * s + m).clamp(0, 1)
        except Exception as e:
            logger.warning("[MiniMax-H3] ImageNet denormalisation skipped (%s).", e)
        return super()._postprocess_decoded_video(video, vae, kwargs)

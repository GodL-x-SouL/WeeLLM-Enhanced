"""
weellm.pipelines.video.minimax_h3_modular_pipeline
==================================================
Custom patched adapter for MiniMax-H3 Modular Pipeline.
Uses WeeBasePipeline's automatic interception to route standard
diffusers MiniMax components to WeeLLM layer-by-layer streamers.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from weellm.pipelines.video.weevideopipeline import WeeVideoPipeline

logger = logging.getLogger("weellm")

# The audio VAE is mono and its decoder upsamples a latent 800x
# (5·5·2·2·2·2·2), which is all the silent fallback needs to size its output.
_AUDIO_UPSAMPLE = 800


def _pipeline_device(pipeline):
    """Best-effort real execution device (components may sit on meta)."""
    for name in ("transformer", "transformer_ref", "vae", "audio_vae", "text_encoder"):
        module = getattr(pipeline, name, None)
        device = getattr(module, "device", None)
        if device is not None and str(device) != "meta":
            return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _silent_audio_vae(device):
    """A complete stand-in for a missing audio VAE.

    `MiniMaxH3AudioDecodeStep` reads `config.latents_mean` / `latents_std`
    before decoding, so the dummy must expose both, and `decode` must return
    `(2, 1, samples)` — the step permutes it into `(1, 2, samples)`.
    """

    class _SilentAudioVAE:
        config = SimpleNamespace(
            latent_channels=32,
            sampling_rate=32000,
            clip_length=17,
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        )

        def __call__(self, *args, **kwargs):
            return None

        def encode(self, *args, **kwargs):
            class _Output:
                latent_dist = SimpleNamespace(
                    sample=lambda: torch.zeros(1, 32, 1, device=device)
                )

            return _Output()

        def decode(self, latents=None, *args, **kwargs):
            if torch.is_tensor(latents) and latents.dim() >= 1:
                channels = int(latents.shape[0]) if latents.dim() >= 3 else 2
                steps = int(latents.shape[-1])
                out_device = latents.device
            else:
                channels, steps, out_device = 2, 40, device
            waveform = torch.zeros(
                channels, 1, max(1, steps * _AUDIO_UPSAMPLE),
                device=out_device, dtype=torch.float32,
            )
            return (waveform,)

    return _SilentAudioVAE()


def _vae_dir_has_weights(vae_dir):
    """True only when the VAE can actually be opened.

    A folder with a config but half the shards (interrupted download) counts
    as missing, so the repair path re-fetches it instead of failing inside the
    loader with a tensor-not-found error.
    """
    try:
        vae_dir = Path(vae_dir)
        if not vae_dir.is_dir() or not (vae_dir / "config.json").is_file():
            return False
        index = vae_dir / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                shards = set(weight_map.values())
                if shards and not all((vae_dir / name).is_file() for name in shards):
                    return False
            except Exception:
                pass
        return any(vae_dir.glob("*.safetensors")) or any(vae_dir.glob("*.bin"))
    except Exception:
        return False


def _repo_id_from_cache_path(path):
    """`.../hub/models--MiniMaxAI--MiniMax-H3/snapshots/<sha>` -> repo id."""
    try:
        for part in Path(path).parts:
            if part.startswith("models--"):
                return part[len("models--"):].replace("--", "/")
    except Exception:
        pass
    return None


def _load_missing_video_vae(pipeline):
    """Reload a video VAE that failed to register during the build.

    The build catches per-component load errors and continues so one optional
    component cannot take a whole model down.  MiniMax-H3 is different: its
    decode block dereferences `components.vae.config` unconditionally, so a
    missed VAE only surfaces after the denoise loop — nine minutes later — as
    `'NoneType' object has no attribute 'config'`.  Recover it here, before
    any GPU time is spent, and fetch the folder if the download skipped it.
    """
    model_dir = getattr(pipeline, "model_dir", None)
    errors = getattr(pipeline, "_weellm_load_errors", None) or {}
    build_error = errors.get("vae")
    if not model_dir:
        logger.error(
            "[MiniMax-H3] Video VAE is missing and the pipeline has no model_dir to reload it from."
        )
        return None

    model_dir = Path(str(model_dir))
    vae_dir = model_dir / "vae"

    if not _vae_dir_has_weights(vae_dir):
        # The build accepts a local snapshot path; a direct repo id is also
        # possible when the studio is driven from Python.
        repo_id = None
        if not model_dir.is_dir() and "/" in str(model_dir) and not str(model_dir).startswith("/"):
            repo_id = str(model_dir)
        repo_id = repo_id or _repo_id_from_cache_path(model_dir)
        logger.warning(
            "[MiniMax-H3] Video VAE weights are missing under %s — fetching the "
            "`vae/` folder now (about 10 GB).  This has to happen before decode; "
            "starting it before denoise avoids losing a whole run at the end.",
            vae_dir,
        )
        if repo_id:
            try:
                from huggingface_hub import snapshot_download

                snap = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["vae/*", "model_index.json"],
                )
                model_dir = Path(snap)
                vae_dir = model_dir / "vae"
            except Exception as exc:
                logger.error("[MiniMax-H3] Could not download %s/vae: %s", repo_id, exc)

    if not _vae_dir_has_weights(vae_dir):
        logger.error(
            "[MiniMax-H3] Video VAE still missing at %s%s.  Re-run Cell 2 so the "
            "download can finish, then generate again.",
            vae_dir,
            f" (build error: {build_error})" if build_error else "",
        )
        return None

    try:
        from weellm.pipelines.weebasepipeline import WeeBasePipeline

        # Match what the build would have used: the video VAE decodes in the
        # transformer's dtype on MiniMax-H3 (fp16 on T4, bf16 on Ampere+),
        # never fp32 — 10 GB of blocks would not fit next to the residents.
        reload_dtype = torch.float32
        for _name in ("transformer", "transformer_ref"):
            _dtype = getattr(getattr(pipeline, _name, None), "dtype", None)
            if _dtype in (torch.float16, torch.bfloat16):
                reload_dtype = _dtype
                break

        logger.info("[MiniMax-H3] Reloading the video VAE from %s (%s) ...", vae_dir, reload_dtype)
        streamer = WeeBasePipeline._load_vae(
            model_dir,
            _pipeline_device(pipeline),
            reload_dtype,
            False,
            subfolder="vae",
        )
        model = getattr(streamer, "model", None)
        if model is None:
            raise RuntimeError("VAE streamer returned no model")
        # The build-time wrappers were installed on whatever `pipeline.vae`
        # was at that moment (None, so they do not exist).  Recreate the two
        # that matter for decode: free the transformer's streamed blocks and
        # cast latents to the VAE dtype.
        try:
            import gc
            import types

            inner = model.decode

            def _recovered_decode(self_obj, *args, **kwargs):
                for tr_name in ("transformer", "transformer_ref"):
                    tr = getattr(pipeline, tr_name, None)
                    tr_streamer = getattr(tr, "_weellm_streamer", None)
                    release = getattr(tr_streamer, "release_cached_blocks", None)
                    if callable(release):
                        try:
                            release(reason="VAE decode (recovered VAE)")
                        except Exception:
                            pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tgt = getattr(inner.__self__, "dtype", None)
                if tgt is not None:
                    args = tuple(
                        a.to(tgt) if torch.is_tensor(a) and a.is_floating_point() else a
                        for a in args
                    )
                    kwargs = {
                        k: (v.to(tgt) if torch.is_tensor(v) and v.is_floating_point() else v)
                        for k, v in kwargs.items()
                    }
                return inner(*args, **kwargs)

            model.decode = types.MethodType(_recovered_decode, model)
        except Exception as exc:
            logger.warning("[MiniMax-H3] Could not wrap recovered VAE decode: %s", exc)
        # Registering keeps `pipeline.components` in sync (the decode block
        # reads the attribute, but other diffusers helpers read the dict).
        try:
            pipeline.register_components(vae=model)
        except Exception:
            pipeline.vae = model
        logger.info(
            "[MiniMax-H3] Video VAE recovered (%s).",
            type(model).__name__,
        )
        return model
    except Exception as exc:
        logger.error("[MiniMax-H3] Video VAE reload failed: %s", exc)
        return None


def _ensure_decode_components(pipeline, quiet=False):
    """Guarantee everything MiniMax-H3's decode blocks touch is present."""
    changed = False

    if getattr(pipeline, "vae", None) is None:
        vae = _load_missing_video_vae(pipeline)
        if vae is not None:
            changed = True
        elif not quiet:
            vae_dir = Path(str(getattr(pipeline, "model_dir", ""))) / "vae"
            errors = getattr(pipeline, "_weellm_load_errors", None) or {}
            raise RuntimeError(
                "MiniMax-H3 cannot decode: the video VAE is missing. "
                f"Expected it in {vae_dir} (config.json + safetensors shards). "
                + (f"Build error: {errors.get('vae')}. " if errors.get("vae") else "")
                + "Re-run the download, then generate again."
            )

    # The audio VAE doubles as the audio-sampling-rate source.  A present but
    # incomplete component (old config without latents_mean/std) is just as
    # fatal as a missing one, so treat both as "install the silent fallback".
    audio_vae = getattr(pipeline, "audio_vae", None)
    audio_cfg = getattr(audio_vae, "config", None)
    try:
        audio_ok = (
            audio_cfg is not None
            and getattr(audio_cfg, "latents_mean", None) is not None
            and getattr(audio_cfg, "latents_std", None) is not None
            and hasattr(audio_vae, "decode")
        )
    except Exception:
        audio_ok = False
    if not audio_ok:
        device = _pipeline_device(pipeline)
        dummy = _silent_audio_vae(device)
        try:
            pipeline.register_components(audio_vae=dummy)
        except Exception:
            pipeline.audio_vae = dummy
        changed = True
        logger.warning(
            "[MiniMax-H3] Audio VAE missing/incomplete — using a silent stereo "
            "fallback so the video still decodes."
        )

    if getattr(pipeline, "video_processor", None) is None:
        try:
            from diffusers.video_processor import VideoProcessor

            processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
            try:
                pipeline.register_components(video_processor=processor)
            except Exception:
                pipeline.video_processor = processor
            changed = True
            logger.warning("[MiniMax-H3] video_processor was missing — built one.")
        except Exception as exc:
            logger.error("[MiniMax-H3] Could not build a video processor: %s", exc)

    if changed or getattr(pipeline, "_weellm_decode_audit_done", False) is not True:
        audit = {
            name: ("ok" if getattr(pipeline, name, None) is not None else "MISSING")
            for name in ("vae", "audio_vae", "video_processor", "transformer", "text_encoder")
            if hasattr(pipeline, name) or getattr(pipeline, name, None) is not None
        }
        try:
            pipeline._weellm_decode_audit_done = True
        except Exception:
            pass
        logger.info("[MiniMax-H3] decode components: %s", ", ".join(
            f"{k}={v}" for k, v in audit.items()
        ))

    return True


def _install_decode_guards():
    """Check the components again right before each decode block runs.

    A component can only disappear between build and decode through a bug or
    an offload path; checking inside the block turns a cryptic nine-minute
    failure into an immediate, actionable error (or a silent audio fallback).
    """
    try:
        from diffusers.modular_pipelines.minimax_h3 import decoders as _decoders
    except Exception:
        return

    _audio_cls = getattr(_decoders, "MiniMaxH3AudioDecodeStep", None)
    _classes = [
        cls for cls in (
            getattr(_decoders, "MiniMaxH3VideoDecodeStep", None),
            _audio_cls,
        ) if cls is not None
    ]

    for _cls in _classes:
        _orig = _cls.__call__
        if getattr(_orig, "_weellm_guarded", False):
            continue

        def _guarded(self, components, state, _orig=_orig, _is_audio=_cls is _audio_cls):
            _ensure_decode_components(components)
            if not _is_audio:
                return _orig(self, components, state)
            # A broken/slow audio VAE must not throw away a finished video.
            try:
                return _orig(self, components, state)
            except Exception as exc:
                if type(getattr(components, "audio_vae", None)).__name__ == "_SilentAudioVAE":
                    raise
                logger.warning(
                    "[MiniMax-H3] Audio decode failed (%s: %s) — retrying with silence; "
                    "the video is kept.", type(exc).__name__, exc,
                )
                dummy = _silent_audio_vae(_pipeline_device(components))
                try:
                    components.register_components(audio_vae=dummy)
                except Exception:
                    components.audio_vae = dummy
                return _orig(self, components, state)

        _guarded._weellm_guarded = True
        _guarded.__name__ = getattr(_orig, "__name__", "__call__")
        _guarded.__doc__ = getattr(_orig, "__doc__", None)
        _cls.__call__ = _guarded


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

        # 1. Make sure the modular decode blocks have every component they
        #    dereference.  Doing this BEFORE denoise turns a missing VAE into
        #    an immediate, actionable error instead of a crash after step N/N.
        _install_decode_guards()
        _ensure_decode_components(self._pipeline)

        # The audio scheduler is a separate scheduler instance; reusing the
        # video one would share its patched closures and timestep state.
        if getattr(self._pipeline, "audio_scheduler", None) is None:
            try:
                new_scheduler = self._pipeline.scheduler.__class__.from_config(
                    self._pipeline.scheduler.config
                )
                setattr(self._pipeline, "audio_scheduler", new_scheduler)
                logger.debug("[WeeLLM] Patched None 'audio_scheduler' with a fresh scheduler.")
            except Exception as exc:
                logger.warning("[MiniMax-H3] audio_scheduler fallback failed: %s", exc)

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

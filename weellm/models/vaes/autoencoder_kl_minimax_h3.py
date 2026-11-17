import json
import logging
import os
import time
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from weellm.io.seeker import get_seeker
from weellm.io.utils import default_dtype, clean_memory, report_memory
from .base_vae_streamer import BaseVAEStreamer

logger = logging.getLogger("weellm")

# VRAM kept free for decoder activations while the blocks are pinned.  The
# decoder works on 256x256 spatial tiles, so activations are small; 2 GiB is
# generous even at 960x544.
_PIN_RESERVE_BYTES = 2 << 30


class AutoencoderKLMiniMaxH3Streamer(BaseVAEStreamer):
    """
    Memory-efficient VAE wrapper for the 10GB MiniMax-H3 Video VAE.

    The decoder is a 36-block ViT with 2.42B parameters (4.5 GiB in fp16).  It
    used to be streamed from disk *per forward call*: at 960x544 / 124 frames
    the decoder runs 7 temporal clips x 15 spatial tiles = 105 passes, and
    every pass reloaded all 36 blocks -- 0.46 TB of disk reads (1.25 TB when
    ``tokens_chunk_size`` was reduced to 2).  That dominated the whole
    generation.

    Now, before decoding, as many decoder blocks as fit are pinned in VRAM in
    one pass.  On a T4 free VRAM after denoising is ~13 GiB, so all 36 blocks
    (4.5 GiB) are pinned and the decode runs at pure GPU speed.  Anything that
    does not fit still streams as before, so low-VRAM hosts keep working.
    Pinned blocks are released when the decode ends, so the next diffusion
    pass gets the full VRAM budget back.

    Set ``WEELLM_VAE_PIN=0`` to force streaming, or ``WEELLM_VAE_PIN_GB=<n>``
    to cap the pin budget explicitly.
    """

    def __init__(
        self,
        model: nn.Module,
        seeker,
        device: str,
        dtype: torch.dtype,
        config_extras: dict = None,
    ) -> None:
        super().__init__(model, seeker, device, dtype)
        # Real latents_mean/std/latent_channels from config.json
        self._config_extras = config_extras or {}
        self._call_counter: int = 0
        self._decode_pass_calls: int = 0
        self._decode_passes: int = 0
        self._n_blocks: int = 0
        self._pinned_prefixes: set = set()

        self._install_decoder_hooks()
        self._patch_encode()

    @property
    def decoder_streaming_prefixes(self) -> tuple:
        return ("decoder.transformer_blocks.",)

    def _install_decoder_hooks(self) -> None:
        # self.model is AutoencoderKLMiniMaxH3; decoder is a direct attribute
        decoder = getattr(self.model, "decoder", None)
        if decoder is None or not hasattr(decoder, "transformer_blocks"):
            logger.warning("[WeeLLM VAE] Decoder does not have expected transformer_blocks — skipping hooks.")
            return

        streaming_blocks = []
        for i, block in enumerate(decoder.transformer_blocks):
            streaming_blocks.append((f"decoder.transformer_blocks.{i}", block))

        self._n_blocks = len(streaming_blocks)
        for shard_prefix, block in streaming_blocks:
            block._vae_shard_prefix  = shard_prefix
            block._vae_loaded_keys   = []
            block.register_forward_pre_hook(self._block_pre_hook)
            block.register_forward_hook(self._block_post_hook)

        logger.info(
            "      -> [WeeLLM VAE] Installed streaming hooks on %d MiniMax VAE decoder blocks.",
            len(streaming_blocks)
        )

    # ── VRAM pinning ──────────────────────────────────────────────
    def _block_bytes(self, block: nn.Module) -> int:
        """Bytes this block needs on the GPU in the decode dtype."""
        try:
            itemsize = torch.tensor([], dtype=self.dtype).element_size()
        except Exception:
            itemsize = 2
        total = 0
        for p in block.parameters():
            total += p.numel() * (itemsize if p.is_floating_point() else p.element_size())
        for b in block.buffers():
            total += b.numel() * (itemsize if b.is_floating_point() else b.element_size())
        return total

    def _pin_decoder_blocks(self) -> int:
        """Load every decoder block that fits in free VRAM; return the count.

        One bulk load replaces the thousands of per-forward disk reads the
        streaming hooks would otherwise perform.
        """
        if os.environ.get("WEELLM_VAE_PIN", "1") == "0":
            logger.info("    [VAE Streamer] VRAM pinning disabled by WEELLM_VAE_PIN=0 — blocks will stream from disk.")
            return 0
        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            return 0
        if self._pinned_prefixes:
            self._release_pinned_blocks()

        decoder = getattr(self.model, "decoder", None)
        blocks = getattr(decoder, "transformer_blocks", None)
        if not blocks:
            return 0

        sizes = [self._block_bytes(b) for b in blocks]
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            return 0

        reserve = max(_PIN_RESERVE_BYTES, int(total_bytes * 0.10))
        budget = max(0, int((free_bytes - reserve) * 0.90))
        env_gb = os.environ.get("WEELLM_VAE_PIN_GB")
        if env_gb:
            try:
                budget = min(budget, int(float(env_gb) * 1e9))
            except Exception:
                pass

        n_pin, acc = 0, 0
        for size in sizes:
            if acc + size > budget:
                break
            acc += size
            n_pin += 1

        logger.info(
            "    [VAE Streamer] VRAM %.2f/%.2f GiB free before decode — pinning %d/%d decoder blocks (%.2f GiB).",
            free_bytes / 2**30, total_bytes / 2**30, n_pin, len(sizes), acc / 2**30,
        )
        if n_pin == 0:
            logger.warning(
                "    [VAE Streamer] Not enough free VRAM to pin a single block — "
                "falling back to per-forward disk streaming (slow)."
            )
            return 0

        for i in range(n_pin):
            prefix = f"decoder.transformer_blocks.{i}"
            try:
                keys = self._get_block_keys(prefix)
                try:
                    # Pinned blocks live in VRAM for the whole decode; caching
                    # their raw bytes in RAM would evict useful entries.
                    sd = self.seeker.get_tensors(
                        keys, device=self.device, dtype=self.dtype, use_cache=False)
                except TypeError:
                    sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
                for name, tensor in sd.items():
                    self._place_tensor(name, tensor, self.device, self.dtype)
                del sd
                self._pinned_prefixes.add(prefix)
            except torch.cuda.OutOfMemoryError:
                clean_memory(self.device)
                logger.warning(
                    "    [VAE Streamer] OOM while pinning block %d — keeping the %d that fit.",
                    i, len(self._pinned_prefixes),
                )
                break
            except Exception as e:
                logger.warning(
                    "    [VAE Streamer] Could not pin block %d (%s) — keeping the %d that fit.",
                    i, e, len(self._pinned_prefixes),
                )
                break
            if (i + 1) % 6 == 0 or (i + 1) == n_pin:
                logger.info("    [VAE Streamer] pinned decoder blocks %d/%d", i + 1, n_pin)

        if len(self._pinned_prefixes) == len(sizes):
            logger.info(
                "    [VAE Streamer] All %d decoder blocks resident in VRAM (%.2f GiB) — "
                "decode no longer touches the disk.",
                len(sizes), acc / 2**30,
            )
        else:
            logger.warning(
                "    [VAE Streamer] Only %d/%d decoder blocks pinned — the remaining %d still stream from disk.",
                len(self._pinned_prefixes), len(sizes), len(sizes) - len(self._pinned_prefixes),
            )
        return len(self._pinned_prefixes)

    def _release_pinned_blocks(self) -> None:
        if not self._pinned_prefixes:
            return
        for prefix in sorted(self._pinned_prefixes):
            try:
                self._evict_keys(self._get_block_keys(prefix))
            except Exception:
                pass
        logger.info("    [VAE Streamer] Released %d pinned decoder blocks.", len(self._pinned_prefixes))
        self._pinned_prefixes.clear()
        clean_memory(self.device)

    # ── per-block hooks (streaming fallback) ──────────────────────
    def _block_pre_hook(self, module: nn.Module, args):
        shard_prefix = module._vae_shard_prefix
        if shard_prefix in self._pinned_prefixes:
            return args

        keys = self._get_block_keys(shard_prefix)
        sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
        for name, tensor in sd.items():
            self._place_tensor(name, tensor, self.device, self.dtype)
        module._vae_loaded_keys = keys

        self._call_counter += 1
        return args

    def _block_post_hook(self, module: nn.Module, args, output):
        shard_prefix = getattr(module, "_vae_shard_prefix", None)
        if shard_prefix in self._pinned_prefixes:
            # Residency is managed for the whole decode; nothing to evict.
            return output

        # Evict block weights from GPU — CPU has no cache so this is final
        self._evict_keys(getattr(module, "_vae_loaded_keys", []))
        module._vae_loaded_keys = []

        # One "pass" = all 36 blocks once = one spatial tile / temporal clip.
        self._decode_pass_calls += 1
        if self._n_blocks and self._decode_pass_calls % self._n_blocks == 0:
            self._decode_passes += 1
            if self._decode_passes % 15 == 0:
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1e9
                    resv = torch.cuda.memory_reserved() / 1e9
                    logger.info(
                        "    [VAE Streamer] streamed pass %d · VRAM %.2f/%.2f GB",
                        self._decode_passes, used, resv,
                    )
                else:
                    logger.info("    [VAE Streamer] streamed pass %d", self._decode_passes)
        return output

    def _patch_encode(self) -> None:
        """Wrap model.encode() to lazy-load encoder weights on first call."""
        original_encode = self.model.encode

        def _lazy_encode(self_obj, *args, **kwargs):
            self._load_encoder()

            kwargs.pop("return_dict", None)

            # Cast first positional arg to correct dtype if it's a tensor
            if args and isinstance(args[0], torch.Tensor):
                args = (args[0].to(self.dtype),) + args[1:]

            # AutoencoderKLMiniMaxH3.encode returns AutoencoderKLOutput with .latent_dist
            result = original_encode(*args, return_dict=True, **kwargs)
            if hasattr(result, "latent_dist"):
                posterior = result.latent_dist
            else:
                from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
                posterior = DiagonalGaussianDistribution(result)

            self._evict_encoder()

            return (posterior, )

        self.model.encode = _lazy_encode.__get__(self.model, self.model.__class__)

    @property
    def spatial_compression_ratio(self):
        return 16

    @property
    def temporal_compression_ratio(self):
        return 1

    @property
    def config(self):
        # AutoencoderKLMiniMaxH3 already has latents_mean, latents_std, latent_channels
        # on its config — return it directly with the config_extras merged in.
        original_config = getattr(self.model, "config", {})

        class ConfigProxy:
            def __init__(self, c, extra):
                self._c = c
                self._extra = extra
            def __getattr__(self, name):
                if name in self._extra:
                    return self._extra[name]
                return getattr(self._c, name)

        return ConfigProxy(original_config, self._config_extras)

    # ── decode ────────────────────────────────────────────────────
    def _decode_plan(self, latents) -> dict:
        """Number of temporal clips / spatial tiles the diffusers decoder will run."""
        model = self.model
        cfg = getattr(model, "config", None)
        chunk = int(getattr(model, "tokens_chunk_size", 5) or 5)
        drop = int(getattr(cfg, "token_drop", 3) or 0)
        chunk = max(1, chunk)
        n_latent = int(latents.shape[2])
        num_tokens = n_latent + drop
        pad = (-num_tokens) % chunk
        clips = max(1, (num_tokens + pad) // chunk - (1 if drop > 0 else 0))

        h = int(latents.shape[-2]) * self.spatial_compression_ratio
        w = int(latents.shape[-1]) * self.spatial_compression_ratio
        tiles = 1
        if getattr(model, "use_tiling", False):
            try:
                hs, _, _ = model._split_tiles(
                    h, model.tile_sample_min_height, model.tile_sample_min_overlap_height)
                ws, _, _ = model._split_tiles(
                    w, model.tile_sample_min_width, model.tile_sample_min_overlap_width)
                tiles = max(1, len(hs) * len(ws))
            except Exception:
                tiles = 1
        return {
            "latent_frames": n_latent,
            "clips": clips,
            "tiles": tiles,
            "passes": clips * tiles,
            "height": h,
            "width": w,
        }

    def _decode_cache_file(self, latents) -> str:
        """Content-addressed cache path so a new prompt can never reuse an old video."""
        import hashlib
        cache_dir = os.path.join(os.getcwd(), ".weellm_cache")
        os.makedirs(cache_dir, exist_ok=True)
        t = latents.detach()
        step = max(1, t.numel() // 4096)
        sample = t.flatten()[::step].float().cpu().numpy().tobytes()
        digest = hashlib.sha1(sample).hexdigest()[:16]
        return os.path.join(cache_dir, "vae_decode_%s_%s.pt" % (tuple(t.shape), digest))

    def decode(self, latents, return_dict=True, **kwargs):
        """Decode latents through the streaming VAE decoder.

        AutoencoderKLMiniMaxH3.decode() routes through self.decoder which is
        MiniMaxH3VideoViTDecoder3d with 36 transformer_blocks.  On entry we pin
        as many blocks as fit in VRAM, run the decode, then release them so the
        next diffusion pass gets the memory back.
        """
        use_cache = os.environ.get("WEELLM_VAE_DECODE_CACHE", "0") == "1"
        cache_file = self._decode_cache_file(latents) if use_cache else None

        if use_cache and cache_file and os.path.exists(cache_file):
            logger.info("    [VAE Streamer] Loading cached decoded video from %s ...", cache_file)
            result = torch.load(cache_file, map_location=latents.device)
            logger.info("    [VAE Streamer] Cached decode loaded: %s", tuple(result.shape))
        else:
            if not use_cache:
                legacy = os.path.join(os.getcwd(), ".weellm_cache", "vae_decode_cache.pt")
                if os.path.exists(legacy):
                    logger.warning(
                        "    [VAE Streamer] Ignoring legacy %s — it is not keyed by prompt and would "
                        "return a previous video.  Set WEELLM_VAE_DECODE_CACHE=1 to opt in to the "
                        "content-addressed cache.",
                        legacy,
                    )

            n_pinned = self._pin_decoder_blocks()
            plan = self._decode_plan(latents)
            logger.info(
                "    [VAE Streamer] decode plan: %d latent frames -> %d temporal clips x %d spatial "
                "tiles = %d passes at %dx%d · %d/%d decoder blocks in VRAM",
                plan["latent_frames"], plan["clips"], plan["tiles"], plan["passes"],
                plan["width"], plan["height"], n_pinned, self._n_blocks,
            )

            report_memory("Before VAE decode")
            self._decode_pass_calls = 0
            self._decode_passes = 0
            t0 = time.time()
            try:
                with torch.no_grad():
                    out = self.model.decode(latents.to(self.dtype), return_dict=False)
                    # out is a tuple; out[0] is the decoded video tensor
                    result = out[0] if isinstance(out, (tuple, list)) else out
            finally:
                self._release_pinned_blocks()

            elapsed = time.time() - t0
            per_pass = elapsed / max(1, plan["passes"])
            report_memory("After VAE decode")
            logger.info(
                "    [VAE Streamer] decode done in %.1fs (%.2f s/pass): %s",
                elapsed, per_pass, tuple(result.shape),
            )

            if use_cache and cache_file:
                logger.info("    [VAE Streamer] Saving video decode cache to %s ...", cache_file)
                torch.save(result.cpu(), cache_file)
                result = result.to(latents.device)

        if return_dict:
            return result
        if isinstance(result, torch.Tensor):
            return (result,)
        if isinstance(result, (tuple, list)):
            return result
        if hasattr(result, "sample"):
            return (result.sample,)
        return (result,)

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    @classmethod
    def from_pretrained(
        cls,
        vae_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_to_ram: bool = False,
    ) -> "AutoencoderKLMiniMaxH3Streamer":
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

        vae_dir = Path(vae_dir)
        config_path = vae_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on MiniMax VAE weights ...")
        # The VAE weights are safetensors shards in the same dir as config.json.
        # Older versions stored a custom 'source_path' key; the standard diffusers
        # config does not have it — fall back to vae_dir itself.
        source_path = vae_dir / config["source_path"] if "source_path" in config else vae_dir
        seeker = get_seeker(str(source_path), cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating AutoencoderKLMiniMaxH3 on meta device ...")
        # Strip diffusers metadata keys before passing to constructor
        init_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
        with init_empty_weights():
            model = AutoencoderKLMiniMaxH3(**init_cfg)
        model.eval()

        # Extract latent normalisation stats for the config proxy
        config_extras = {
            "latent_channels": config.get("latent_channels", 24),
            "latents_mean":    config.get("latents_mean", [0.0] * 24),
            "latents_std":     config.get("latents_std",  [1.0] * 24),
            "clip_length":     config.get("clip_length",  17),
        }
        streamer = cls(model, seeker, device, dtype, config_extras=config_extras)

        logger.info("  Step 3/3 -- Loading VAE resident tensors to GPU ...")
        resident_keys = streamer._get_resident_keys()
        resident_sd   = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for name, tensor in resident_sd.items():
            streamer._place_tensor(name, tensor, device, dtype)
        del resident_sd

        # Non-persistent buffers (e.g. decoder.rope.inv_freq) are computed in
        # __init__ under init_empty_weights() and land on CPU — they are NOT in
        # the weight_map, so resident loading skips them.  Move any CPU buffers
        # in the decoder to the target device now so the RoPE forward doesn't
        # crash with a cuda:0 vs cpu device mismatch.
        _decoder = getattr(model, "decoder", None)
        if _decoder is not None:
            for buf_name, buf in list(_decoder.named_buffers()):
                if buf.device.type == "cpu":
                    buf.data = buf.data.to(device)
                    logger.info("    [VAE] Moved decoder buffer '%s' → %s", buf_name, device)

        clean_memory(device)
        report_memory("After VAE resident load")
        logger.info("  MiniMaxVAEStreamer ready (%d resident keys, decoder blocks pinned on demand).", len(resident_keys))
        return streamer

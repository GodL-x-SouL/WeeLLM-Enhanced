import hashlib
import json
import logging
import os
import shutil
import struct
import threading
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

# Blocks read per seeker call.  One call per block meant 36 file opens and 36
# seek patterns; consecutive blocks live next to each other in the shard, so
# batching them turns the pin into a few nearly-sequential reads.  Four blocks
# is 512 MiB of fp32 source in flight — small enough for an 8 GB host.
_PIN_BATCH_BLOCKS = 4

# Below this measured read rate the source is clearly slow storage (Drive,
# network mount) and the fp16 sidecar cache is worth building.
_PIN_CACHE_MIN_RATE = 80 << 20

_HEARTBEAT_SECONDS = 15.0


# ── checkpoint consistency / self-repair ─────────────────────────
def _read_safetensors_header(path):
    """Return ``(header, data_base, file_size)`` for a safetensors shard."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file is too small to be safetensors")
        (header_size,) = struct.unpack("<Q", raw)
        if header_size <= 0 or header_size > 64 << 20:
            raise ValueError(f"implausible header size {header_size}")
        blob = f.read(header_size)
        if len(blob) != header_size:
            raise ValueError("truncated safetensors header")
        header = json.loads(blob.decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + header_size, path.stat().st_size


def _vae_shard_problems(vae_dir: Path) -> dict:
    """Map shard filename -> problem for files that cannot supply the index.

    A key is usable if *any* shard has it: the seeker now resolves indices that
    point at the wrong shard.  Only keys that no shard contains, unreadable
    shards, and truncated shards are reported as problems.
    """
    vae_dir = Path(vae_dir)
    index = vae_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        # Single-file VAEs have no index to disagree with.
        return {} if any(vae_dir.glob("*.safetensors")) else {
            str(index.name): "missing index and no safetensors shards"
        }

    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except Exception as exc:
        return {"*": f"unreadable index: {type(exc).__name__}: {exc}"}

    by_shard = {}
    for key, src in weight_map.items():
        by_shard.setdefault(src, []).append(key)

    problems = {}
    headers = {}
    for src in by_shard:
        path = vae_dir / src
        if not path.is_file():
            problems[src] = "file is missing"
            continue
        try:
            header, data_base, size = _read_safetensors_header(path)
        except Exception as exc:
            problems[src] = f"unreadable header: {type(exc).__name__}: {exc}"
            continue
        headers[src] = header
        ends = [
            header[key]["data_offsets"][1]
            for key in by_shard[src]
            if key in header
        ]
        if ends and size < data_base + max(ends):
            problems[src] = (
                f"truncated ({size} B, needs at least {data_base + max(ends)} B)"
            )

    global_headers = list(headers.values())
    absent = {}
    for key, src in weight_map.items():
        if key in headers.get(src, {}):
            continue
        if any(key in header for header in global_headers):
            continue  # index drift; the seeker can find it in another shard
        absent.setdefault(src, []).append(key)

    for src, keys in absent.items():
        preview = ", ".join(repr(k) for k in keys[:3])
        problems[src] = (
            f"{len(keys)} indexed tensor(s) are in no shard at all "
            f"(e.g. {preview})"
        )
    return problems


def _repo_id_from_cache_path(path: Path):
    for part in Path(path).parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/", 1)
    return None


def _install_downloaded(downloaded: Path, dest: Path) -> None:
    """Point *dest* at *downloaded* without copying multi-GB shards."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".weellm-new")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    try:
        os.symlink(str(downloaded), tmp)
    except OSError:
        shutil.copy2(downloaded, tmp)
    os.replace(tmp, dest)


def _vae_missing_keys(vae_dir: Path) -> list:
    """Index keys that no shard header contains at all."""
    vae_dir = Path(vae_dir)
    index = vae_dir / "diffusion_pytorch_model.safetensors.index.json"
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except Exception:
        return []
    headers = {}
    for src in set(weight_map.values()):
        path = vae_dir / src
        if not path.is_file():
            continue
        try:
            header, _, _ = _read_safetensors_header(path)
        except Exception:
            continue
        headers[src] = header
    missing = []
    for key, src in weight_map.items():
        if key in headers.get(src, {}):
            continue
        if any(key in header for header in headers.values()):
            continue
        missing.append(key)
    return missing


def _repair_vae_shards(vae_dir: Path, subfolder: str = "vae") -> dict:
    """Re-download only the VAE shards that disagree with the index.

    Returns the remaining problems (empty when the folder is usable).
    """
    vae_dir = Path(vae_dir)
    problems = _vae_shard_problems(vae_dir)
    if not problems:
        return {}

    missing = _vae_missing_keys(vae_dir)
    structural = any("in no shard at all" not in why for why in problems.values())
    benign = all(
        key.lower().endswith(".bias") and "norm" in key.lower()
        for key in missing
    )
    if missing and not structural and benign:
        logger.warning(
            "  [VAE] Only neutral normalisation tensor(s) are absent (%s); "
            "initialising them instead of re-downloading a multi-GB shard.",
            ", ".join(missing[:4]),
        )
        return problems

    repo_id = _repo_id_from_cache_path(vae_dir)
    logger.warning(
        "  [VAE] Checkpoint/index mismatch under %s: %s",
        vae_dir, "; ".join(f"{k}: {v}" for k, v in problems.items()),
    )
    if not repo_id:
        logger.warning(
            "  [VAE] Cannot tell which Hub repo %s came from; the loader will "
            "use any shard that contains each tensor.", vae_dir,
        )
        return problems

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    for src in list(problems):
        if src == "*":
            continue
        filename = f"{subfolder}/{src}" if subfolder else src
        try:
            logger.warning("  [VAE] Re-downloading %s (bad local copy) ...", filename)
            got = hf_hub_download(
                repo_id=repo_id, filename=filename, force_download=True, token=token,
            )
            _install_downloaded(Path(got), vae_dir / src)
        except Exception as exc:
            logger.error("  [VAE] Re-download of %s failed: %s", filename, exc)

    remaining = _vae_shard_problems(vae_dir)
    if remaining:
        logger.warning(
            "  [VAE] Still inconsistent after repair: %s",
            "; ".join(f"{k}: {v}" for k, v in remaining.items()),
        )
    else:
        logger.info("  [VAE] Checkpoint repaired — index and shards agree again.")
    return remaining


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
    to cap the pin budget explicitly.  ``WEELLM_VAE_PIN_CACHE`` controls the
    fp16 sidecar cache (``auto`` by default: built only when the checkpoint
    reads slower than 80 MiB/s; ``1`` always, ``0`` never).
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
        self._decode_passes_total: int = 0
        self._n_blocks: int = 0
        self._pinned_prefixes: set = set()
        self._decode_t0: float = 0.0
        self._hb_stop = None
        self._hb_thread = None
        self._hb_label = "working"

        # The pipeline receives `streamer.model`, not the streamer itself, so
        # diffusers would call the raw decode() and skip the pin/plan/cache
        # path entirely (it kept streaming all 36 blocks per pass).  Bind the
        # streamer's decode onto the model.  The hooks below keep this object
        # alive, so this survives the streamer reference being dropped.
        self._original_model_decode = model.decode
        try:
            model.decode = self.decode
        except Exception as exc:
            logger.warning("[WeeLLM VAE] Could not bind streamer decode: %s", exc)

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

    def _fill_missing_resident(self, keys) -> None:
        """Initialise resident tensors that the checkpoint simply does not have.

        A mismatch between the index and the shard headers used to be fatal even
        for a tiny LayerNorm bias.  A zero bias / unit weight is exactly the
        neutral value for those layers, so use that instead of refusing to
        decode.  Any non-norm tensor is still a hard error: guessing a conv or
        attention weight would silently produce garbage video.
        """
        if not keys:
            return
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        filled, bad = [], []
        for key in keys:
            tensor = params.get(key)
            if tensor is None:
                tensor = buffers.get(key)
            if tensor is None or tensor.device.type != "meta":
                continue
            if "norm" not in key.lower():
                bad.append(key)
                continue
            value = 1.0 if key.endswith(".weight") else 0.0
            target_dtype = self.dtype if tensor.is_floating_point() else tensor.dtype
            data = torch.full(
                tuple(tensor.shape), value, device=self.device, dtype=target_dtype,
            )
            from accelerate.utils import set_module_tensor_to_device

            set_module_tensor_to_device(
                self.model, key, self.device, value=data, dtype=target_dtype,
            )
            filled.append(key)
        if bad:
            raise RuntimeError(
                "MiniMax VAE checkpoint is missing non-normalisation weights: "
                + ", ".join(repr(k) for k in bad[:6])
                + ". Re-download the `vae/` folder."
            )
        if filled:
            logger.warning(
                "    [VAE] %d tensor(s) absent from the checkpoint were initialised "
                "to neutral norm values: %s", len(filled), ", ".join(filled[:6]),
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

    # ── heartbeat ─────────────────────────────────────────────────
    # Loading 4.5 GiB and decoding 100+ passes can take minutes on slow
    # storage / a T4.  A quiet log for that long looks like a hang, so a
    # background thread reports progress even while the main thread blocks.
    def _start_heartbeat(self, label: str) -> None:
        self._stop_heartbeat()
        self._hb_label = label
        self._hb_stop = threading.Event()
        t0 = time.time()

        def _beat():
            while not self._hb_stop.wait(_HEARTBEAT_SECONDS):
                try:
                    if torch.cuda.is_available():
                        used = torch.cuda.memory_allocated() / 2**30
                        resv = torch.cuda.memory_reserved() / 2**30
                    else:
                        used = resv = 0.0
                    logger.info(
                        "    [VAE Streamer] still %s · %.0fs elapsed · pinned %d/%d blocks · "
                        "pass %d/%d · VRAM %.2f/%.2f GiB",
                        self._hb_label, time.time() - t0,
                        len(self._pinned_prefixes), self._n_blocks,
                        self._decode_passes, self._decode_passes_total or 0,
                        used, resv,
                    )
                except Exception:
                    pass

        self._hb_thread = threading.Thread(
            target=_beat, daemon=True, name="weellm-vae-heartbeat")
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        ev, th = self._hb_stop, self._hb_thread
        if ev is not None:
            ev.set()
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._hb_stop = None
        self._hb_thread = None

    # ── optional fp16 sidecar cache ───────────────────────────────
    def _fast_cache_dir(self) -> Path:
        root = os.environ.get("WEELLM_VAE_CACHE_DIR") or os.path.join(
            os.getcwd(), ".weellm_cache")
        return Path(root) / "vae_decoder_fp16"

    def _source_manifest(self):
        """Fingerprint the checkpoint so a stale cache is never reused.

        Always keyed on the whole decoder, not on the number of blocks that
        happen to fit this run: a smaller pin budget can still load a prefix
        of a cache built when all blocks fit.
        """
        try:
            model_dir = Path(str(getattr(self.seeker, "model_dir", "") or ""))
            shards = sorted({str(v) for v in getattr(self.seeker, "weight_map", {}).values()})
            files = []
            for src in shards:
                p = model_dir / src
                st = p.stat()
                files.append([src.replace("\\", "/"), int(st.st_size), int(st.st_mtime)])
            keys = []
            for i in range(self._n_blocks):
                keys.extend(self._get_block_keys(f"decoder.transformer_blocks.{i}"))
            return {
                "version": 1,
                "dtype": str(self.dtype),
                "n_blocks": int(self._n_blocks),
                "files": files,
                "keys_sha1": hashlib.sha1("\n".join(keys).encode()).hexdigest(),
            }
        except Exception:
            return None

    def _fast_cache_ready(self, n_blocks: int) -> bool:
        d = self._fast_cache_dir()
        manifest = d / "manifest.json"
        try:
            if not manifest.exists():
                return False
            stored = json.loads(manifest.read_text())
            current = self._source_manifest()
            if current is None or stored != current:
                return False
            return all((d / f"block_{i:02d}.pt").exists() for i in range(n_blocks))
        except Exception:
            return False

    def _load_fast_cache(self, n_blocks: int) -> int:
        d = self._fast_cache_dir()
        loaded = 0
        for i in range(n_blocks):
            sd = torch.load(d / f"block_{i:02d}.pt", map_location="cpu")
            for name, tensor in sd.items():
                self._place_tensor(name, tensor, self.device, self.dtype)
            del sd
            self._pinned_prefixes.add(f"decoder.transformer_blocks.{i}")
            loaded += 1
        return loaded

    def _block_module(self, prefix: str):
        mod = self.model
        for part in prefix.split("."):
            mod = getattr(mod, part)
        return mod

    def _save_block_cache(self, prefix: str) -> bool:
        try:
            d = self._fast_cache_dir()
            d.mkdir(parents=True, exist_ok=True)
            mod = self._block_module(prefix)
            sd = {}
            for name, p in mod.named_parameters():
                sd[f"{prefix}.{name}"] = p.detach().to("cpu")
            for name, b in mod.named_buffers():
                sd[f"{prefix}.{name}"] = b.detach().to("cpu")
            idx = int(prefix.rsplit(".", 1)[1])
            tmp = d / f"block_{idx:02d}.tmp"
            torch.save(sd, tmp)
            os.replace(tmp, d / f"block_{idx:02d}.pt")
            return True
        except Exception as e:
            logger.warning("    [VAE Streamer] Could not write fp16 decoder cache: %s", e)
            return False

    def _finalize_fast_cache(self, n_blocks: int) -> None:
        manifest = self._source_manifest()
        if manifest is None:
            return
        try:
            (self._fast_cache_dir() / "manifest.json").write_text(json.dumps(manifest))
            logger.info(
                "    [VAE Streamer] Saved fp16 decoder cache (%d blocks) to %s — "
                "the next run loads it instead of the slow source.",
                n_blocks, self._fast_cache_dir(),
            )
        except Exception as e:
            logger.warning("    [VAE Streamer] Could not finalize fp16 decoder cache: %s", e)

    # ── pinning ───────────────────────────────────────────────────
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
        # Account for blocks the allocator is still holding before deciding.
        clean_memory(self.device)
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
        src_dir = getattr(self.seeker, "model_dir", None)
        if src_dir:
            logger.info("    [VAE Streamer] decoder weights: %s", src_dir)
        if n_pin == 0:
            logger.warning(
                "    [VAE Streamer] Not enough free VRAM to pin a single block — "
                "falling back to per-forward disk streaming.  That is HOURS at this "
                "resolution: lower frames/resolution, or free VRAM first."
            )
            return 0

        all_sizes = sum(sizes[:n_pin])
        few = n_pin < len(sizes)

        # Fast path: a validated fp16 copy written by a previous run.
        if self._fast_cache_ready(n_pin):
            logger.info(
                "    [VAE Streamer] Found fp16 decoder cache at %s — loading from it "
                "instead of the slow checkpoint.", self._fast_cache_dir())
            t0 = time.time()
            try:
                loaded = self._load_fast_cache(n_pin)
                logger.info(
                    "    [VAE Streamer] Loaded %d/%d decoder blocks from cache in %.1fs "
                    "(%.0f MiB/s).",
                    loaded, n_pin, time.time() - t0, all_sizes / 2**20 / max(0.1, time.time() - t0),
                )
                return loaded
            except Exception as e:
                logger.warning(
                    "    [VAE Streamer] fp16 decoder cache is unreadable (%s) — "
                    "rebuilding from the checkpoint.", e)
                self._release_pinned_blocks()

        cache_mode = os.environ.get("WEELLM_VAE_PIN_CACHE", "auto").strip().lower()
        saving = cache_mode == "1"
        if cache_mode == "0":
            saving = False
        if few:
            saving = False

        t0 = time.time()
        read_bytes = 0
        all_saved = True
        try:
            for start in range(0, n_pin, _PIN_BATCH_BLOCKS):
                batch = list(range(start, min(start + _PIN_BATCH_BLOCKS, n_pin)))
                keys = []
                for i in batch:
                    keys.extend(self._get_block_keys(f"decoder.transformer_blocks.{i}"))
                try:
                    try:
                        # Pinned blocks live in VRAM for the whole decode; the
                        # raw fp32 bytes would not fit the RAM cache anyway.
                        sd = self.seeker.get_tensors(
                            keys, device=self.device, dtype=self.dtype, use_cache=False)
                    except TypeError:
                        sd = self.seeker.get_tensors(keys, device=self.device, dtype=self.dtype)
                    for name, tensor in sd.items():
                        self._place_tensor(name, tensor, self.device, self.dtype)
                    del sd
                except torch.cuda.OutOfMemoryError:
                    clean_memory(self.device)
                    logger.warning(
                        "    [VAE Streamer] OOM while pinning blocks %d-%d — keeping the %d that fit.",
                        batch[0], batch[-1], len(self._pinned_prefixes),
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "    [VAE Streamer] Could not pin blocks %d-%d (%s) — keeping the %d that fit.",
                        batch[0], batch[-1], e, len(self._pinned_prefixes),
                    )
                    break

                for i in batch:
                    self._pinned_prefixes.add(f"decoder.transformer_blocks.{i}")
                read_bytes += sum(sizes[i] for i in batch)
                elapsed = max(1e-6, time.time() - t0)
                rate = read_bytes / elapsed
                eta = max(0.0, (all_sizes - read_bytes) / max(1.0, rate))
                logger.info(
                    "    [VAE Streamer] pinned %d/%d blocks · %.2f GiB · %.1f MiB/s · "
                    "%.0fs elapsed · ETA %.0fs",
                    len(self._pinned_prefixes), n_pin, read_bytes / 2**30,
                    rate / 2**20, elapsed, eta,
                )

                if cache_mode == "auto" and not saving and rate < _PIN_CACHE_MIN_RATE:
                    saving = True
                    logger.info(
                        "    [VAE Streamer] source reads at %.1f MiB/s (slow storage) — "
                        "writing an fp16 cache to %s for the next run.  Moving the model "
                        "folder to fast local storage (e.g. /content) also fixes this.",
                        rate / 2**20, self._fast_cache_dir(),
                    )
                    # The rate is only measurable after the first batch, so the
                    # blocks already resident still need writing or the cache
                    # would be missing its first files for ever.
                    for i in range(start):
                        all_saved &= self._save_block_cache(
                            f"decoder.transformer_blocks.{i}")
                if saving:
                    for i in batch:
                        all_saved &= self._save_block_cache(
                            f"decoder.transformer_blocks.{i}")
        finally:
            if (saving and all_saved and len(self._pinned_prefixes) == n_pin
                    and not few):
                self._finalize_fast_cache(self._n_blocks)

        if len(self._pinned_prefixes) == len(sizes):
            logger.info(
                "    [VAE Streamer] All %d decoder blocks resident in VRAM (%.2f GiB) after %.0fs — "
                "decode no longer touches the disk.",
                len(sizes), acc / 2**30, time.time() - t0,
            )
        else:
            logger.warning(
                "    [VAE Streamer] Only %d/%d decoder blocks pinned — the remaining %d still "
                "stream from disk.  That can turn a 1 minute decode into hours.",
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
        try:
            clean_memory(self.device)
        except Exception as exc:
            logger.warning("    [VAE Streamer] Memory cleanup after release failed: %s", exc)

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
        if shard_prefix not in self._pinned_prefixes:
            # Streaming fallback: the block goes back to meta after this use.
            self._evict_keys(getattr(module, "_vae_loaded_keys", []))
            module._vae_loaded_keys = []

        # One "pass" = all 36 blocks once = one spatial tile / temporal clip.
        self._decode_pass_calls += 1
        if self._n_blocks and self._decode_pass_calls % self._n_blocks == 0:
            self._decode_passes += 1
            if self._decode_passes % 15 == 0:
                elapsed = max(0.0, time.time() - self._decode_t0) if self._decode_t0 else 0.0
                per = elapsed / max(1, self._decode_passes)
                try:
                    vram = " · VRAM %.2f/%.2f GiB" % (
                        torch.cuda.memory_allocated() / 2**30,
                        torch.cuda.memory_reserved() / 2**30,
                    ) if torch.cuda.is_available() else ""
                except Exception:
                    vram = ""
                logger.info(
                    "    [VAE Streamer] pass %d/%d · %.0fs elapsed · %.2f s/pass%s",
                    self._decode_passes, self._decode_passes_total or 0,
                    elapsed, per, vram,
                )
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

            # Plan first: the pin can take minutes on slow storage, and the
            # pass count tells the user exactly how much work is coming.
            plan = self._decode_plan(latents)
            self._decode_passes_total = plan["passes"]
            self._decode_pass_calls = 0
            self._decode_passes = 0
            logger.info(
                "    [VAE Streamer] decode plan: %d latent frames -> %d temporal clips x %d spatial "
                "tiles = %d decoder passes at %dx%d.",
                plan["latent_frames"], plan["clips"], plan["tiles"], plan["passes"],
                plan["width"], plan["height"],
            )

            self._start_heartbeat("loading decoder blocks")
            n_pinned = 0
            try:
                n_pinned = self._pin_decoder_blocks()
                self._hb_label = "decoding"
                logger.info(
                    "    [VAE Streamer] %d/%d decoder blocks in VRAM · starting %d passes.",
                    n_pinned, self._n_blocks, plan["passes"],
                )

                report_memory("Before VAE decode")
                self._decode_t0 = time.time()
                t0 = self._decode_t0

                def _run_decode():
                    with torch.no_grad():
                        out = self._original_model_decode(latents.to(self.dtype), return_dict=False)
                        # out is a tuple; out[0] is the decoded video tensor
                        return out[0] if isinstance(out, (tuple, list)) else out

                try:
                    result = _run_decode()
                    elapsed = time.time() - t0
                except torch.cuda.OutOfMemoryError:
                    # Pinning is an optimisation: if the tiles + pinned blocks
                    # do not fit, drop the pinned copy and stream block by
                    # block.  That is much slower but it is the path that
                    # worked before pinning existed, and losing a finished
                    # denoise over a decode OOM is worse than a slow decode.
                    logger.warning(
                        "    [VAE Streamer] OOM while decoding with %d pinned blocks — "
                        "releasing them and retrying in streaming mode (slower).",
                        len(self._pinned_prefixes),
                    )
                    self._release_pinned_blocks()
                    clean_memory(self.device)
                    report_memory("Before VAE decode retry")
                    self._hb_label = "decoding (streaming fallback)"
                    self._decode_t0 = time.time()
                    t0 = self._decode_t0
                    result = _run_decode()
                    elapsed = time.time() - t0
            finally:
                self._stop_heartbeat()
                # Flush every decoder kernel before the pinned buffers are
                # handed back to the allocator.  Without this, an async CUDA
                # fault surfaces as a mysterious death right after the last
                # "Released ..." log line, with no traceback.
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception as exc:
                        logger.warning(
                            "    [VAE Streamer] CUDA sync after decode failed: %s", exc)
                # Releasing is best-effort: a failure must not throw away a
                # finished decode.
                try:
                    self._release_pinned_blocks()
                except Exception as exc:
                    logger.warning(
                        "    [VAE Streamer] Could not release pinned blocks: %s", exc)

            per_pass = elapsed / max(1, plan["passes"])
            try:
                import os as _os

                import psutil

                _ram_note = " · RAM %.1f GB free (process %.1f GB)" % (
                    psutil.virtual_memory().available / 1e9,
                    psutil.Process(_os.getpid()).memory_info().rss / 1e9,
                )
            except Exception:
                _ram_note = ""
            report_memory("After VAE decode")
            logger.info(
                "    [VAE Streamer] decode done in %.1fs (%.2f s/pass): %s%s",
                elapsed, per_pass, tuple(result.shape), _ram_note,
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
        try:
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
        except ImportError:
            # The class moved around between diffusers dev builds; the public
            # export is stable, so use it when the module path is not.
            import diffusers as _diffusers

            AutoencoderKLMiniMaxH3 = getattr(_diffusers, "AutoencoderKLMiniMaxH3")

        vae_dir = Path(vae_dir)
        config_path = vae_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info("  Step 1/3 -- Initialising LiveSeeker on MiniMax VAE weights ...")
        # The VAE weights are safetensors shards in the same dir as config.json.
        # Older versions stored a custom 'source_path' key; the standard diffusers
        # config does not have it — fall back to vae_dir itself.
        source_path = vae_dir / config["source_path"] if "source_path" in config else vae_dir
        if Path(source_path).is_dir():
            # A stale index/shard mix used to crash the first resident load with
            # e.g. KeyError('decoder.norm_out.bias').  Check before touching GPU
            # and re-fetch only the shards that are actually wrong.
            try:
                _repair_vae_shards(Path(source_path), subfolder=vae_dir.name or "vae")
            except Exception as exc:
                logger.warning("  [VAE] Consistency repair skipped: %s", exc)
        seeker = get_seeker(str(source_path), cache_to_ram=cache_to_ram)
        logger.info("  Found %d VAE tensors across shards.", len(seeker.weight_map))

        logger.info("  Step 2/3 -- Instantiating AutoencoderKLMiniMaxH3 on meta device ...")
        # Strip diffusers metadata keys before passing to constructor
        init_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
        with init_empty_weights():
            try:
                model = AutoencoderKLMiniMaxH3(**init_cfg)
            except TypeError as exc:
                # Newer/older diffusers builds add or remove constructor
                # arguments.  Dropping only the keys the installed class does
                # not know keeps the checkpoint loading instead of leaving
                # `vae=None` for the decode block to trip over.
                import inspect

                params = inspect.signature(AutoencoderKLMiniMaxH3.__init__).parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    raise
                filtered = {k: v for k, v in init_cfg.items() if k in params}
                dropped = sorted(set(init_cfg) - set(filtered))
                logger.warning(
                    "  [VAE] %s rejected the config (%s); retrying without %s",
                    AutoencoderKLMiniMaxH3.__name__, exc, dropped,
                )
                model = AutoencoderKLMiniMaxH3(**filtered)
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
        try:
            resident_sd = seeker.get_tensors(
                resident_keys, device=device, dtype=dtype, allow_missing=True,
            )
        except TypeError:
            # Seekers other than the disk one do not accept allow_missing.
            resident_sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        for name, tensor in resident_sd.items():
            streamer._place_tensor(name, tensor, device, dtype)
        del resident_sd
        missing = list(getattr(seeker, "missing_keys", []) or [])
        if missing:
            streamer._fill_missing_resident(missing)

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

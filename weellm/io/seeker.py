import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("weellm")


def _extract_hub_repo_from_cache_path(path: Path):
    """
    If *path* is inside an HF Hub cache directory
    (e.g. .../models--org--repo/snapshots/<hash>/subfolder),
    return (repo_id, subfolder).  Returns (None, None) otherwise.
    """
    parts = path.parts
    for i, part in enumerate(parts):
        if part.startswith("models--"):
            # part looks like "models--HiDream-ai--HiDream-I1-Full"
            repo_id = part[len("models--"):].replace("--", "/", 1)
            # The subfolder is everything after .../snapshots/<hash>/
            try:
                snap_idx = parts.index("snapshots", i)
                subfolder_parts = parts[snap_idx + 2:]  # skip 'snapshots' and the hash
                subfolder = "/".join(subfolder_parts) if subfolder_parts else None
            except ValueError:
                subfolder = None
            return repo_id, subfolder
    return None, None


import threading
from contextlib import contextmanager

_override_local = threading.local()

@contextmanager
def override_weights_path(path: Union[str, Path, None], subfolder: str = None):
    """
    Temporarily overrides the path used by get_seeker() within the current thread.
    Useful for routing weights loading to a GGUF file without changing the directory
    path used for loading config.json.
    """
    old_path = getattr(_override_local, "weights_path", None)
    old_sub  = getattr(_override_local, "subfolder", None)
    
    _override_local.weights_path = path
    _override_local.subfolder = subfolder
    try:
        yield
    finally:
        _override_local.weights_path = old_path
        _override_local.subfolder = old_sub

def _looks_like_comfy_quant(path: Path) -> bool:
    """True if a safetensors file/dir contains Comfy-style quant companions.

    Detection is deliberately strict so plain BF16/FP8 checkpoints never
    route here: requires a ``*.weight_s_rel`` key (W4A8, unambiguous), or an
    INT8 ``*.weight`` with a sibling ``*.weight_scale``. Header JSON only --
    no tensor data is read.
    """
    import json as _json
    import struct as _struct

    def _header_of(fp: Path):
        with open(fp, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            (hsize,) = _struct.unpack("<Q", raw)
            return _json.loads(f.read(hsize).decode("utf-8"))

    try:
        if path.is_file() and path.suffix.lower() == ".safetensors":
            header = _header_of(path)
            keys = set(header) - {"__metadata__"}
            if any(k.endswith(".weight_s_rel") for k in keys):
                return True
            for k in keys:
                if k.endswith(".weight") and header.get(k, {}).get("dtype") == "I8":
                    if k[: -len(".weight")] + ".weight_scale" in keys:
                        return True
            return False
        if path.is_dir():
            for idx in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
                ip = path / idx
                if ip.exists():
                    wm = _json.loads(ip.read_text())["weight_map"]
                    # .weight_s_rel is W4A8-only (unambiguous). .weight_scale
                    # covers INT8/W4A4 packs; FP8 checkpoints also use that
                    # name but still load correctly through ComfyQuantSeeker's
                    # plain-tensor path, so routing them here is harmless.
                    if any(k.endswith(".weight_s_rel") for k in wm):
                        return True
                    if any(k.endswith(".weight_scale") for k in wm):
                        return True
                    return False
            shards = sorted(path.glob("*.safetensors"))
            if len(shards) == 1:
                return _looks_like_comfy_quant(shards[0])
    except Exception:
        return False
    return False


def get_seeker(model_dir: Union[str, Path], cache_to_ram: bool = False):
    """
    Factory function to return the appropriate tensor seeker.

    - **.gguf file path**: returns a GGUFSeeker that dequantizes weights on the
      fly using pure PyTorch — no custom CUDA compilation required.
    - **Comfy-quant safetensors (INT8 / W4A8, possibly mixed with BF16/F32)**:
      returns a ComfyQuantSeeker that decodes quantized groups at load time
      (pure PyTorch, no comfy-kitchen required). Detection is automatic via
      companion-key presence; plain checkpoints are unaffected.
    - **directory (default)**: returns SafetensorsRAMSeeker when cache_to_ram
      is True, otherwise SafetensorsDiskSeeker (original behaviour).
    """
    override = getattr(_override_local, "weights_path", None)
    if override is not None:
        model_dir = override
        
    model_dir_path = Path(model_dir)

    # ── GGUF: single-file path ending in .gguf ────────────────────────────────
    if model_dir_path.is_file() and model_dir_path.suffix.lower() == ".gguf":
        from weellm.io.ggufs.gguf_seek import GGUFSeeker
        return GGUFSeeker(model_dir_path)
        
    # ── Single File Direct Hub Download ──────────────────────────────────────
    model_dir_str = str(model_dir).replace("\\", "/")
    
    # If it looks like a HuggingFace direct file path: "org/repo/path/to/file.ext"
    # and it doesn't exist locally as a relative path.
    if not model_dir_path.exists() and not model_dir_path.is_absolute():
        parts = model_dir_str.split("/")
        if len(parts) >= 3 and "." in parts[-1]:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            logger.info("  [WeeLLM] Hub file '%s' not found locally. Downloading from repo: %s...", model_dir_str, repo_id)
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model_dir_path = Path(downloaded_path)
            
    # ── GGUF Initialization ──────────────────────────────────────────────────
    if model_dir_path.is_file() and model_dir_path.suffix.lower() == ".gguf":
        from weellm.io.ggufs.gguf_seek import GGUFSeeker
        return GGUFSeeker(model_dir_path)

    original_model_dir_str = str(model_dir).replace("\\", "/")
    explicit_subfolder = None
    if len(original_model_dir_str.split("/")) > 2 and not Path(original_model_dir_str).is_absolute():
        explicit_subfolder = "/".join(original_model_dir_str.split("/")[2:])

    if not model_dir_path.exists():
        parts = original_model_dir_str.split("/")
        is_hub_id = len(parts) >= 2 and not Path(original_model_dir_str).is_absolute()

        if is_hub_id:
            actual_repo_id = f"{parts[0]}/{parts[1]}"

            logger.info("Directory '%s' not found. Attempting to download from Hugging Face Hub (repo: %s)...", model_dir, actual_repo_id)
            from huggingface_hub import snapshot_download, HfApi
            try:
                files = HfApi().list_repo_files(repo_id=actual_repo_id)
                is_pipeline = "model_index.json" in files
            except Exception:
                is_pipeline = False

            # If user provided a subfolder in the string, use it. Otherwise, use the inferred one.
            target_subfolder = explicit_subfolder or getattr(_override_local, "subfolder", None)
            
            if is_pipeline and target_subfolder:
                allow_patterns = [
                    f"{target_subfolder}/*.safetensors",
                    f"{target_subfolder}/*.json",
                    f"{target_subfolder}/*.safetensors.index.json",
                    "model_index.json"
                ]
                logger.info("  Detected pipeline repo. Downloading ONLY subfolder '%s' ...", target_subfolder)
            else:
                allow_patterns = ["*.safetensors", "*.safetensors.index.json", "*.json"]

            model_dir = snapshot_download(
                repo_id=actual_repo_id,
                allow_patterns=allow_patterns,
            )
        else:
            # Absolute local path — try to recover a missing HF cache subfolder.
            repo_id, subfolder = _extract_hub_repo_from_cache_path(model_dir_path)
            if repo_id and subfolder:
                logger.info(
                    "Local subfolder '%s' not found. Downloading '%s' from repo '%s' ...",
                    model_dir_path, subfolder, repo_id,
                )
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        f"{subfolder}/**",
                        f"{subfolder}/*",
                        f"{subfolder}/*.safetensors",
                        f"{subfolder}/*.json",
                    ],
                )
                # snapshot_download places files back into the existing cache;
                # the path should now exist.
                if not model_dir_path.exists():
                    raise FileNotFoundError(
                        f"Download attempted but directory still not found: '{model_dir_path}'"
                    )
            else:
                raise FileNotFoundError(
                    f"Local directory not found and path is not a valid Hub repo_id: '{model_dir}'"
                )
        model_dir_path = Path(model_dir)

    # Automatically append subfolder if the directory is a full pipeline repo
    final_target_subfolder = explicit_subfolder or getattr(_override_local, "subfolder", None)
    if final_target_subfolder and (model_dir_path / "model_index.json").exists():
        # Prevent double-appending if model_dir_path already contains the subfolder
        if not model_dir_path.name == final_target_subfolder:
            model_dir_path = model_dir_path / final_target_subfolder

    # ── Comfy-quant quantized safetensors (INT8 / W4A8, mixed files) ──────
    # Disk-streamed with on-the-fly pure-PyTorch dequant; independent of the
    # cache_to_ram flag (whole-file RAM caching would defeat the purpose).
    try:
        if _looks_like_comfy_quant(model_dir_path):
            from weellm.io.safetensors.comfy_quant_seek import ComfyQuantSeeker
            return ComfyQuantSeeker(model_dir_path)
    except Exception as e:
        logger.debug("Comfy-quant detection skipped (%s); using default seeker.", e)

    if cache_to_ram:
        from weellm.io.safetensors.ram_seek import SafetensorsRAMSeeker
        return SafetensorsRAMSeeker(model_dir_path)
    else:
        from weellm.io.safetensors.disk_seek import SafetensorsDiskSeeker
        return SafetensorsDiskSeeker(model_dir_path)

import os
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import Gemma2Model, Gemma2Config, Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
import threading
from concurrent.futures import ThreadPoolExecutor
from weellm.memory import evict_module, place_tensors
from weellm.seeker import get_seeker
import logging

logger = logging.getLogger("weellm")

class Gemma4UnifiedForConditionalGenerationStreamer:
    """
    Streams Gemma2 9B/12B layer-by-layer for LTX-2.5 Text Encoder.
    """
    def __init__(self, model, seeker, device="cuda", dtype=torch.bfloat16, prefetch=True):
        self.model = model
        self.seeker = seeker
        self.device = device
        self.dtype = dtype
        self.prefetch = prefetch
        
        uses_nested = any(k.startswith("model.language_model.layers.") for k in self.seeker.weight_map)
        self.shard_prefix = "model.language_model.layers" if uses_nested else "model.layers"
        self.layer_count = len(model.layers)
        self._shard_order = [f"{self.shard_prefix}.{i}" for i in range(self.layer_count)]
        
        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._next_future = None
        self._next_future_name = None
        self._lock = threading.Lock()
        
        self._install_hooks()

    def _get_resident_keys(self):
        resident = []
        for k in self.seeker.weight_map.keys():
            if not (k.startswith("model.layers.") or k.startswith("model.language_model.layers.")):
                resident.append(k)
        return resident

    def _install_hooks(self):
        for i, shard_name in enumerate(self._shard_order):
            layer = self.model.layers[i]
            layer._gemma_te_shard = shard_name
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)

    def _pre_hook(self, module, args):
        shard_name = module._gemma_te_shard
        layer_keys = [k for k in self.seeker.weight_map if k.startswith(shard_name + ".")]
        
        with self._lock:
            if self.prefetch and self._next_future_name == shard_name and self._next_future is not None:
                sd = self._next_future.result()
                self._next_future = None
                self._next_future_name = None
            else:
                sd = self.seeker.get_tensors(layer_keys, device=self.device, dtype=self.dtype)
                
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                mapped_sd[k[len("model.language_model."):]] = v
            elif k.startswith("model."):
                mapped_sd[k[len("model."):]] = v
            else:
                mapped_sd[k] = v
                
        for mapped_k, mapped_v in mapped_sd.items():
            try:
                place_tensors(self.model, {mapped_k: mapped_v}, self.device, self.dtype, skip_errors=False)
            except Exception as e:
                if "layer_scalar" not in mapped_k:
                    logger.error(f"FAILED TO PLACE {mapped_k}: {repr(e)}")
                    
        pos = int(shard_name.split(".")[-1])
        next_pos = pos + 1
        if self.prefetch and self._executor is not None and next_pos < self.layer_count:
            next_name = self._shard_order[next_pos]
            next_keys = [k for k in self.seeker.weight_map if k.startswith(next_name + ".")]
            with self._lock:
                self._next_future = self._executor.submit(
                    self.seeker.get_tensors, next_keys, self.device, self.dtype
                )
                self._next_future_name = next_name

    def _post_hook(self, module, args, output):
        evict_module(module)
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        device="cuda",
        dtype=torch.bfloat16,
        prefetch=True,
        cache_to_ram=False,
    ):
        logger.info("Step 1/3 -- Initializing LiveSeeker on Gemma4-12B weights ...")
        seeker = get_seeker(str(model_dir), cache_to_ram=cache_to_ram)
        
        logger.info("  Instantiating Gemma3TextModel on meta device ...")
        # Exact LTX-2.5 Text Encoder (Gemma 3 12B architecture):
        import json
        import os
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
            
        if "text_config" in cfg_dict and "use_bidirectional_attention" in cfg_dict["text_config"]:
            cfg_dict["text_config"]["use_bidirectional_attention"] = False
            
        config = Gemma3Config.from_dict(cfg_dict)
        text_config = config.text_config
        
        # Crucial fix: The original Gemma 3 config has vocab_size 262208, 
        # but the LTX-2.5 safetensors checkpoint has exactly 262144 embeddings.
        text_config.vocab_size = 262144
        
        with init_empty_weights():
            model = Gemma3TextModel(text_config)
            
            # Monkey-patch Gemma 4 (LTX-2.5) specific layer dimensions:
            # Gemma 3 Text Model defines uniform head counts for all layers.
            # However, LTX-2.5 uses 32 Q-heads and 2 KV-heads for full_attention layers,
            # and 16 Q-heads and 8 KV-heads for sliding_attention layers.
            hidden_size = text_config.hidden_size
            head_dim = getattr(text_config, "head_dim", hidden_size // text_config.num_attention_heads)
            for i, layer in enumerate(model.layers):
                if text_config.layer_types[i] == "full_attention":
                    # Patch for 16 q_heads, 1 kv_head, head_dim = 512
                    layer.self_attn.head_dim = 512
                    layer.self_attn.num_key_value_groups = 16 // 1
                    layer.self_attn.q_proj = nn.Linear(hidden_size, 16 * 512, bias=text_config.attention_bias)
                    layer.self_attn.k_proj = nn.Linear(hidden_size, 1 * 512, bias=text_config.attention_bias)
                    layer.self_attn.v_proj = nn.Linear(hidden_size, 1 * 512, bias=text_config.attention_bias)
                    layer.self_attn.o_proj = nn.Linear(16 * 512, hidden_size, bias=text_config.attention_bias)
                    layer.self_attn.q_norm = type(layer.self_attn.q_norm)(dim=512, eps=text_config.rms_norm_eps)
                    layer.self_attn.k_norm = type(layer.self_attn.k_norm)(dim=512, eps=text_config.rms_norm_eps)
                    
            # Patch rotary embedding for full_attention to use dim=512
            base = text_config.rope_parameters["full_attention"]["rope_theta"]
            inv_freq = 1.0 / (base ** (torch.arange(0, 512, 2, dtype=torch.float32, device="meta") / 512.0))
            model.rotary_emb.register_buffer("full_attention_inv_freq", inv_freq, persistent=False)
            model.rotary_emb.register_buffer("full_attention_original_inv_freq", inv_freq.clone(), persistent=False)
                    
        model.eval()

        for buf_name, buf in model.named_buffers():
            if buf is not None and buf.device.type == "meta":
                try:
                    set_module_tensor_to_device(model, buf_name, device, value=torch.zeros_like(buf, device=device))
                except Exception:
                    pass

        logger.info("Step 2/3 -- Hooking streaming layers ...")
        streamer = cls(
            model=model,
            seeker=seeker,
            device=device,
            dtype=dtype,
            prefetch=prefetch,
        )

        logger.info("Step 3/3 -- Loading resident tensors ...")
        resident_keys = streamer._get_resident_keys()
        sd = seeker.get_tensors(resident_keys, device=device, dtype=dtype)
        
        mapped_sd = {}
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                mapped_sd[k[len("model.language_model."):]] = v
            elif k.startswith("model."):
                mapped_sd[k[len("model."):]] = v
            else:
                mapped_sd[k] = v
                
        for mapped_k, mapped_v in mapped_sd.items():
            try:
                place_tensors(model, {mapped_k: mapped_v}, device, dtype, skip_errors=True)
            except Exception:
                pass
        del sd
        
        return streamer

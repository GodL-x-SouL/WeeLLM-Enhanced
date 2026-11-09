# WeeLLM-Enhanced

**Layer-streaming inference for large diffusion models — under 4 GB VRAM, no quantization required.**

WeeLLM streams text-encoder and transformer layers to the GPU instead of forcing the whole model into VRAM. It pins as many blocks as your hardware allows and seamlessly streams the rest layer-by-layer in the background, so massive models run on budgets as low as 4 GB VRAM with full bfloat16 weights. Quantized checkpoints (GGUF, INT8/W4A8 ConvRot) are also supported and decoded on the fly in pure torch when disk or RAM is tight.

## At a glance

- **Tasks** — text → image, text → video, image → video.
- **Studio models (Colab / Gradio)** — three: Ideogram 4 (text → image), MiniMax-H3 (text → video), and LTX 2.5 (image → video), each with recommended quantized checkpoints.
- **Models (Python API / CLI)** — SD 1.5–3.5, SDXL, FLUX.1 / FLUX.2, Qwen-Image, Z-Image, Krea 2, Ideogram 4, CogView4, HiDream, Lumina 2, LongCat, ERNIE-Image, AuraFlow, MiniMax-H3, LTX 2.5, and more.
- **Interfaces** — Python API (`WeeTextToImagePipeline`, `WeeImageToImagePipeline`, `WeeVideoPipeline`), CLI (`main.py`), and **WeeLLM Studio**, a two-cell Colab notebook (`WeeLLM_Studio.ipynb`) serving a Gradio UI with live download/generation progress and logs.
- **Control** — explicit `vram_budget` / `ram_budget`, optional prefetch depth, and raw-block RAM caching.

## Getting started

- **Colab** — open `WeeLLM_Studio.ipynb`, pick a GPU runtime, and run both cells.
- **Kaggle** — open `WeeLLM_Studio_Kaggle.ipynb`, enable GPU, and run both cells; model downloads go to `/kaggle/tmp` (HF cache symlinked there) and the separate `GradioUI.py` is served as a public share URL.
- **Local** — `pip install -r requirements.txt`, then run `python main.py --model <hf-repo> --prompt "..." --vram_budget 4 --ram_budget 4`.
- **UI preview** — `python GradioUI.py` serves the Studio frontend locally (simulated engine when no GPU/weellm); the same file runs the real engine on Colab/Kaggle.

> Streaming is optimized for local NVMe SSDs; cloud instances with restricted disk I/O will be slower. Dense autoregressive image models are impractical without quantization.

MIT license. Based on [Jit-Roy/weellm](https://github.com/Jit-Roy/weellm).
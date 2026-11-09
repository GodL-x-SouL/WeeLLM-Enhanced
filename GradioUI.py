"""Standalone frontend preview of the WeeLLM Studio Gradio UI, extracted from WeeLLM_Studio.ipynb (cell 2). Runs without torch/GPU/weellm/Colab; engine phases are simulated so every UI state can be inspected."""
import os, time, random, threading, inspect, html
import gradio as gr

# ── model registry (verbatim from the notebook) ───────────────
TASKS = [('Image', 't2i'), ('Video', 't2v'), ('Image → Video', 'i2v')]
MODELS = {

 'ideogram-4': dict(task='t2i', label='Ideogram 4', pipe='t2i',
  repo_url='https://huggingface.co/ideogram-ai/ideogram-4-fp8', mirror_url=None, gated=True,
  access='Gated repo — token required (even for quants: encoder + VAE come from here). One GGUF serves both transformer branches.',
  gguf_repo_url='https://huggingface.co/Abiray/ideogram-4-GGUF',
  steps=30, guid=0.0, w=1024, h=1024,
  ignore=['assets/*'],
  quants=[
   ('full', 'Official FP8 release · gated', 'https://huggingface.co/ideogram-ai/ideogram-4-fp8'),
   ('ideogram4-Q4_K.gguf', 'Q4_K · 5.8 GB · recommended (no _M exists for this model)', 'https://huggingface.co/Abiray/ideogram-4-GGUF/blob/main/ideogram4-Q4_K.gguf'),
  ],
  default_q='ideogram4-Q4_K.gguf'),

 'minimax-h3': dict(task='t2v', label='MiniMax-H3 (FL2VA) · experimental', pipe='t2v',
  repo_url='https://huggingface.co/MiniMaxAI/MiniMax-H3', mirror_url=None, gated=False,
  access='Open repo — experimental: pruned-GGUF path only; full weights (144 GB FL2VA) cannot fit Colab.',
  gguf_repo_url='https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF',
  te_gguf=('https://huggingface.co/DeepBeepMeep/MiniMax-H3', 'Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf'), subfolder='FL2VA', fetch_patterns=['*.json', 'FL2VA/*.json', 'FL2VA/tokenizer*', 'FL2VA/tokenizer/*', 'FL2VA/processor/*', 'FL2VA/visual_vae/*', 'FL2VA/audio_vae/*', 'FL2VA/scheduler/*', 'FL2VA/*.py', 'FL2VA/*.txt'],
  steps=6, guid=0.0, w=960, h=544, frames=73,
  ignore=[],
  quants=[
   ('MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf', 'Q4_K_M pruned · 11.6 GB · recommended', 'https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/blob/main/MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf'),
  ],
  default_q='MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf'),
 'ltx25': dict(task='i2v', label='LTX 2.5 · distilled', pipe='i2v',
  repo_url='https://huggingface.co/Lightricks/LTX-2.5-Diffusers', mirror_url=None, gated=True,
  access='Gated repo — token required (even for quants: the encoder stack + VAEs come from here). Click Agree and Access on the repo page.',
  gguf_repo_url='https://huggingface.co/joeygambino/LTX-2.5-Quantized',
  ignore=['ltx-2.5-*.safetensors', '*.webp', 'transformer_full/*', 'loras/*', 'latent_upscale_models/*', 'model_patches/*'],
  steps=25, guid=0.0, w=960, h=544, frames=121,
  quants=[('LTX25-distilled-DiT-Q4_K_M.gguf', 'Q4_K_M · 14.2 GB · recommended · ~55 GB total', 'https://huggingface.co/joeygambino/LTX-2.5-Quantized/blob/main/LTX25-distilled-DiT-Q4_K_M.gguf'),
  ],
  default_q='LTX25-distilled-DiT-Q4_K_M.gguf'),
 'ltx25-w4a8': dict(task='i2v', label='LTX 2.5 W4A8 ConvRot', pipe='i2v',
  repo_url='https://huggingface.co/Lightricks/LTX-2.5-Diffusers', mirror_url=None, gated=True,
  access='Gated base repo (token + Agree) supplies VAEs/configs. The W4A8 DiT + encoder come from the open Winnougan pack; WeeLLM decodes them block-by-block in pure torch (bit-identical to ComfyUI). T4-friendly.',
  gguf_repo_url='https://huggingface.co/Winnougan/ltx-2.5-w4a8-convrot-int4-convrot-Winnougan-Blessing',
  te_safetensors=('https://huggingface.co/Winnougan/ltx-2.5-w4a8-convrot-int4-convrot-Winnougan-Blessing', 'text_encoders/gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors'),
  ignore=['ltx-2.5-*.safetensors', '*.webp', 'transformer_full/*', 'loras/*', 'latent_upscale_models/*', 'model_patches/*'],
  steps=25, guid=0.0, w=960, h=544, frames=121,
  quants=[('diffusion_models/ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors', 'W4A8 ConvRot DiT · 12.5 GB · recommended (pairs with the W4A8 encoder below)', 'https://huggingface.co/Winnougan/ltx-2.5-w4a8-convrot-int4-convrot-Winnougan-Blessing/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors')],
  default_q='diffusion_models/ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors'),
 'ltx25-int8': dict(task='i2v', label='LTX 2.5 INT8 ConvRot (official)', pipe='i2v',
  repo_url='https://huggingface.co/Lightricks/LTX-2.5-Diffusers', mirror_url=None, gated=True,
  access='Fully gated path — token + Agree needed for the base repo AND the official INT8 files. WeeLLM decodes INT8 (incl. ConvRot rotation) in pure torch; natively fast on T4 INT8 cores.',
  gguf_repo_url='https://huggingface.co/Lightricks/LTX-2.5',
  te_safetensors=('https://huggingface.co/Lightricks/LTX-2.5', 'text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors'),
  ignore=['ltx-2.5-*.safetensors', '*.webp', 'transformer_full/*', 'loras/*', 'latent_upscale_models/*', 'model_patches/*'],
  steps=25, guid=0.0, w=960, h=544, frames=121,
  quants=[('diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors', 'INT8 ConvRot DiT · official · recommended (pairs with the INT8 encoder below)', 'https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors')],
  default_q='diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors'),

}
PIPE_CLS = {'t2i': 'WeeTextToImagePipeline', 't2v': 'WeeVideoPipeline', 'i2v': 'WeeVideoPipeline'}
def _rid(url):
    u = url.split('huggingface.co/', 1)[-1]
    for k in ('/blob/', '/tree/', '/resolve/'):
        u = u.split(k)[0]
    return u.strip('/')
OUT_DIR = '/tmp/weellm_studio_preview'
os.makedirs(OUT_DIR, exist_ok=True)

_AUTO_TOKEN = ''  # no Colab secrets outside Colab

# ── realtime log bus (feeds the in-app logs panel) ─────────────
LOG_BUF, LOG_LOCK = [], threading.Lock()
LOG_FILE = os.path.join(OUT_DIR, 'studio.log')
def _push_log(msg):
    with LOG_LOCK:
        LOG_BUF.append(msg)
        del LOG_BUF[:-400]
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as _lf:
            _lf.write(msg + '\n')
    except Exception:
        pass
def _log_text():
    with LOG_LOCK:
        return '\n'.join(LOG_BUF)[-12000:] or '— logs stream here during download and generation —'
def ustatus(msg):
    _push_log('[studio] ' + msg)

# ── helpers ────────────────────────────────────────────────────
def _fmt_gb(n):
    return '%.1f GB' % (n / 1e9) if n >= 1e9 else '%d MB' % (n / 1e6)
def _budgets():
    return 12.5, 8.5  # preview defaults ≈ T4 runtime; no torch/psutil needed

# ── minimal HTML fragments (in-app) ────────────────────────────
def status_html(msg, kind='run'):
    dot = {'run': '#e8c96a', 'ok': '#7ee2a0', 'err': '#ff9d9d', 'idle': '#8a8a95'}[kind]
    return ("<div style='display:flex;align-items:center;gap:8px;font-size:13px;color:#c9c9d1;padding:2px 0;'>"
            "<span style='width:7px;height:7px;border-radius:50%;background:" + dot + ";flex-shrink:0;'></span><span>" + html.escape(msg) + "</span></div>")
def bar_html(pct, label):
    pct = max(0.0, min(100.0, pct))
    return ("<div style='font-size:11px;color:#8a8a95;margin:2px 0 4px;'>" + html.escape(label) + "</div>"
            "<div style='height:4px;border-radius:2px;background:#26262e;overflow:hidden;'>"
            "<div style='height:100%;width:" + ('%.1f' % pct) + "%;background:#c9c9d1;transition:width .3s;'></div></div>")

# ── UI events ──────────────────────────────────────────────────
def _models_for(task):
    return [(m['label'], k) for k, m in MODELS.items() if m['task'] == task]
def on_task(task):
    opts = _models_for(task)
    first = opts[0][1] if opts else None
    e = MODELS[first]
    qdefault = [l for v, l, _u in e['quants'] if v == e['default_q']][0]
    return (gr.update(choices=opts, value=first), gr.update(choices=[l for _v, l, _u in e['quants']], value=qdefault), gr.update(visible=(task in ('t2v', 'i2v'))), gr.update(visible=(task == 'i2v')), gr.update(value=e['steps']), gr.update(value=e['w']), gr.update(value=e['h']), gr.update(value=e.get('frames', 73)), model_hint(e))
def on_model(model_key):
    e = MODELS[model_key]
    return (gr.update(choices=[l for _v, l, _u in e['quants']], value=[l for v, l, _u in e['quants'] if v == e['default_q']][0]), gr.update(value=e['steps']), gr.update(value=e['w']), gr.update(value=e['h']), gr.update(value=e.get('frames', 73)), model_hint(e))
def model_hint(e):
    return ("<div style='font-size:12px;color:#8a8a95;line-height:1.5;'><a style='color:#aab4e8;' href='%s' target='_blank'>%s</a> · %s</div>" % (e['repo_url'], _rid(e['repo_url']), html.escape(e['access'])))

# ---- version-tolerant component constructor (verbatim) ----
def _C(cls, *args, **kw):
    try:
        _ps = inspect.signature(cls).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _ps.values()):
            kw = {k: v for k, v in kw.items() if k in _ps}
    except Exception:
        pass
    try:
        return cls(*args, **kw)
    except TypeError:
        for _drop in ('show_copy_button', 'autoscroll', 'max_lines', 'placeholder'):
            kw.pop(_drop, None)
        return cls(*args, **kw)

# ── simulated engine (UI preview only — no downloads, no GPU) ──
def do_unload():
    ustatus('model unloaded, VRAM released. (simulated)')
    return status_html('Idle — model unloaded.', 'idle')

def _placeholder_image(seed, w, h):
    from PIL import Image
    rng = random.Random(seed)
    top = tuple(rng.randrange(60, 220) for _ in range(3))
    bot = tuple(rng.randrange(20, 90) for _ in range(3))
    strip = Image.new('RGB', (1, 256))
    strip.putdata([tuple(int(a + (b - a) * i / 255) for a, b in zip(top, bot)) for i in range(256)])
    return strip.resize((w, h))

_RUN_LOCK = threading.Lock()

def run_all(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b):
    if not _RUN_LOCK.acquire(blocking=False):
        yield status_html('Another run is already in progress — please wait for it to finish.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    try:
        yield from _run_all_sim(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b)
    finally:
        _RUN_LOCK.release()

def _run_all_sim(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b):
    entry = MODELS.get(model_key)
    if entry is not None:
        _qmap = {lbl: val for val, lbl, _u in entry['quants']}
        quant = _qmap.get(quant, quant)
    if entry is None or not (prompt or '').strip():
        yield status_html('Pick a model and write a prompt first.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    if entry['pipe'] == 'i2v' and start_img is None:
        yield status_html('Image → Video needs a start frame image.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    seed = int(seed)
    if seed < 0:
        seed = random.randrange(2 ** 31 - 1)
    steps = int(steps)
    ustatus('run: %s · %s · %dx%d · %d steps · seed %d (UI PREVIEW — simulated, nothing is downloaded or generated)' % (entry['label'], quant, int(w), int(h), steps, seed))

    dl_state = {'pct': 0, 'txt': 'Download — resolving sizes…'}
    gen_state = {'pct': 0, 'txt': 'Generation — waiting'}
    def snap():
        return status_html('Working — see bars and logs below', 'run'), bar_html(dl_state['pct'], dl_state['txt']), bar_html(gen_state['pct'], gen_state['txt']), _log_text(), gr.skip(), gr.skip()

    # phase 1: simulated download
    yield snap()
    _total = 5.8e9
    for _v, _l, _u in entry['quants']:
        if _v == quant:
            import re
            _m = re.search(r'(\d+(?:\.\d+)?)\s*GB', _l)
            if _m:
                _total = float(_m.group(1)) * 1e9
    ustatus('fetch plan: Transformer %s = %s (simulated)' % (quant, _fmt_gb(_total)))
    for i in range(11):
        dl_state.update(pct=10.0 * i, txt='Download — %s / %s' % (_fmt_gb(_total * i / 10), _fmt_gb(_total)))
        yield snap()
        time.sleep(0.2)
    dl_state.update(pct=100, txt='Download — complete (%s)' % _fmt_gb(_total))
    ustatus('weights at %s (simulated)' % OUT_DIR)
    yield snap()

    # phase 2: simulated load
    gen_state.update(pct=0, txt='Generation — loading pipeline…')
    ustatus('loading %s (%s) … (simulated)' % (entry['label'], quant))
    yield snap()
    time.sleep(1.2)
    ustatus('pipeline ready: %s (simulated)' % PIPE_CLS[entry['pipe']])

    # phase 3: simulated denoising
    t0 = time.time()
    for s in range(steps + 1):
        gen_state.update(pct=100.0 * s / max(1, steps), txt='Generation — denoising %d/%d' % (s, steps))
        yield snap()
        time.sleep(min(0.15, 3.0 / max(1, steps)))
    dt = time.time() - t0
    img = None
    if entry['pipe'] == 't2i':
        img = _placeholder_image(seed, int(w), int(h))
        img.save(os.path.join(OUT_DIR, 'img_%d.png' % int(t0)))
    else:
        ustatus('video export is engine-side — not available in the UI preview.')
    ustatus('done in %.0fs · seed %d (simulated)' % (dt, seed))
    gen_state.update(pct=100, txt='Generation — done in %.0fs' % dt)
    yield status_html('Done in %.0fs · seed %d · UI PREVIEW (simulated output)' % (dt, seed), 'ok'), bar_html(100, 'Download — complete'), bar_html(100, gen_state['txt']), _log_text(), gr.update(value=img, visible=img is not None), gr.skip()

# ── minimal theme: one dark palette for both OS color schemes ──
def _studio_theme():
    t = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=['system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        font_mono=['ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
        text_size=gr.themes.sizes.text_md,
    )
    pal = dict(
        body_background_fill='#101014', body_text_color='#e8e8ea', body_text_color_subdued='#9a9aa3',
        background_fill_primary='#101014', background_fill_secondary='#17171c',
        border_color_primary='#2a2a33', border_color_accent='#3a3a45', border_color_accent_subdued='#26262e',
        color_accent='#e8e8ea', color_accent_soft='#26262e',
        block_background_fill='#15151a', block_border_color='#26262e',
        block_label_background_fill='#15151a', block_label_text_color='#c9c9d1',
        block_title_background_fill='#15151a', block_title_text_color='#e8e8ea',
        block_info_text_color='#8a8a95',
        panel_background_fill='#15151a', panel_border_color='#26262e',
        input_background_fill='#0c0c10', input_background_fill_focus='#0c0c10',
        input_border_color='#2c2c35', input_border_color_focus='#4a4a55',
        input_placeholder_color='#62626d',
        button_primary_background_fill='#e8e8ea', button_primary_background_fill_hover='#ffffff',
        button_primary_text_color='#101014', button_primary_border_color='#e8e8ea',
        button_secondary_background_fill='#1c1c22', button_secondary_background_fill_hover='#26262e',
        button_secondary_text_color='#e8e8ea', button_secondary_border_color='#2c2c35',
        slider_color='#e8e8ea', link_text_color='#aab4e8',
        checkbox_background_color_selected='#e8e8ea', checkbox_border_color_selected='#e8e8ea',
        checkbox_label_background_fill_selected='#26262e',
        accordion_text_color='#e8e8ea', loader_color='#e8e8ea',
    )
    valid = inspect.signature(t.set).parameters
    kw = {}
    for k, v in pal.items():
        if k in valid:
            kw[k] = v
        if k + '_dark' in valid:
            kw[k + '_dark'] = v
    return t.set(**kw)

# ── build the app ──────────────────────────────────────────────
_auto_vram, _auto_ram = _budgets()
GRADIO_CSS = (
 # hide every native scrollbar; wheel/touch scrolling still works
 '*{scrollbar-width:none !important;-ms-overflow-style:none !important;}'
 '*::-webkit-scrollbar{display:none !important;width:0 !important;height:0 !important;}'
 'html,body{background:#101014 !important;}'
 '.gradio-container{max-width:720px !important;width:100% !important;margin:0 auto !important;padding:14px 14px 40px !important;}'
 'footer{display:none !important;}'
 '.wl-h1{font-size:15px;font-weight:600;color:#e8e8ea;margin:0 0 10px;letter-spacing:.02em;}'
 '.gradio-container .block{box-shadow:none !important;}'
 '.gradio-container label span,.gradio-container .label{font-size:12px !important;}'
 '@media (max-width:640px){'
 '.gradio-container{padding:10px 10px 32px !important;}'
 '.gradio-container .row,.gradio-container [data-testid="row"]{flex-wrap:wrap !important;gap:8px !important;}'
 '.gradio-container .row > *,.gradio-container [data-testid="row"] > *,.gradio-container .column,.gradio-container [data-testid="column"]{flex:1 1 100% !important;min-width:0 !important;max-width:100% !important;}'
 'textarea,input[type="text"],input[type="password"],input[type="number"]{font-size:16px !important;}'
 '}'
)
with gr.Blocks(title='WeeLLM Studio', analytics_enabled=False, css=GRADIO_CSS, theme=_studio_theme()) as demo:
    _C(gr.HTML, "<div class='wl-h1'>WeeLLM Studio</div>")
    task_r = _C(gr.Radio, choices=TASKS, value='t2i', label='Task', show_label=False)
    model_d = _C(gr.Dropdown, choices=_models_for('t2i'), value='ideogram-4', label='Model')
    quant_d = _C(gr.Dropdown, choices=[l for _v, l, _u in MODELS['ideogram-4']['quants']], value=[l for v, l, _u in MODELS['ideogram-4']['quants'] if v == MODELS['ideogram-4']['default_q']][0], label='Quantization')
    hint_h = _C(gr.HTML, model_hint(MODELS['ideogram-4']))
    prompt_t = _C(gr.Textbox, label='Prompt', lines=3, placeholder='Describe what to generate…')
    with _C(gr.Row, visible=False) as i2v_box:
        start_im = _C(gr.Image, type='pil', label='Start frame')
    with _C(gr.Row, visible=False) as vid_box:
        frames_s = _C(gr.Slider, 9, 241, value=73, step=1, label='Frames (auto-snapped to the model)')
    with _C(gr.Row):
        run_b = _C(gr.Button, 'Run', variant='primary', scale=4)
        unload_b = _C(gr.Button, 'Unload', variant='secondary', scale=1)
    status_h = _C(gr.HTML, status_html('Idle — configure and press Run', 'idle'))
    dl_h = _C(gr.HTML, bar_html(0, 'Download'))
    gen_h = _C(gr.HTML, bar_html(0, 'Generation'))
    img_o = _C(gr.Image, label='Result', interactive=False, visible=False)
    vid_o = _C(gr.Video, label='Result', visible=False)
    with _C(gr.Accordion, 'Advanced', open=False):
        token_t = _C(gr.Textbox, label='HF token (gated repos — auto-loaded from Colab secrets)', type='password', value=_AUTO_TOKEN, placeholder='hf_…')
        neg_t = _C(gr.Textbox, label='Negative prompt', lines=1, placeholder='blurry, low quality')
        with _C(gr.Row):
            w_s = _C(gr.Slider, 256, 2048, value=1024, step=16, label='Width')
            h_s = _C(gr.Slider, 256, 2048, value=1024, step=16, label='Height')
        with _C(gr.Row):
            steps_s = _C(gr.Slider, 1, 50, value=30, step=1, label='Steps')
            guid_s = _C(gr.Slider, 0, 10, value=0.0, step=0.5, label='Guidance (0 = default)')
        with _C(gr.Row):
            seed_n = _C(gr.Number, value=42, precision=0, label='Seed (−1 = random)')
            dtype_r = _C(gr.Radio, choices=['bfloat16', 'float16'], value='bfloat16', label='Dtype')
        with _C(gr.Row):
            vram_s = _C(gr.Slider, 2, 16, value=_auto_vram, step=0.5, label='VRAM budget (GB)')
            ram_s = _C(gr.Slider, 2, 24, value=_auto_ram, step=0.5, label='RAM budget (GB)')
    with _C(gr.Accordion, 'Logs', open=False):
        logs_t = _C(gr.Textbox, label='Live logs', lines=10, max_lines=20, autoscroll=True, show_copy_button=True, value='— press Run · logs stream here —')
    task_r.change(on_task, inputs=[task_r], outputs=[model_d, quant_d, vid_box, i2v_box, steps_s, w_s, h_s, frames_s, hint_h])
    model_d.change(on_model, inputs=[model_d], outputs=[quant_d, steps_s, w_s, h_s, frames_s, hint_h])
    unload_b.click(do_unload, inputs=[], outputs=[status_h])
    _click_kw = dict(inputs=[task_r, model_d, quant_d, token_t, prompt_t, neg_t, w_s, h_s, steps_s, guid_s, seed_n, frames_s, start_im, dtype_r, vram_s, ram_s], outputs=[status_h, dl_h, gen_h, logs_t, img_o, vid_o])
    try:
        run_b.click(run_all, concurrency_limit=1, **_click_kw)
    except TypeError:
        run_b.click(run_all, **_click_kw)
    def _poll_logs():
        return _log_text()
    try:
        demo.load(_poll_logs, inputs=[], outputs=[logs_t], every=2)
    except TypeError:
        try:
            demo.load(_poll_logs, inputs=[], outputs=[logs_t])
        except Exception:
            pass

if __name__ == '__main__':
    try:
        demo.queue(max_size=8)
    except TypeError:
        demo.queue()
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False, show_error=True)
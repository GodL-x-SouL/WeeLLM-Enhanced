"""WeeLLM Studio — standalone Gradio app. With torch+GPU+weellm present (Colab/Kaggle)
it runs the real layer-streaming engine; without them it falls back to a simulated
preview so the frontend can be inspected locally. Notebooks launch this file via
serve() instead of embedding the UI source."""
import os, time, random, threading, inspect, html, gc, traceback, fnmatch, logging, shutil
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
# ── platform + engine capability ──────────────────────────────
_IS_KAGGLE = os.path.isdir('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
_IS_COLAB = 'COLAB_RELEASE_TAG' in os.environ or os.path.isdir('/content')
try:
    import torch
    _HAS_GPU = torch.cuda.is_available()
except Exception:
    torch = None
    _HAS_GPU = False
try:
    import weellm
    _HAS_WEELLM = True
except Exception:
    _HAS_WEELLM = False
REAL = bool(_HAS_GPU and _HAS_WEELLM)

if os.environ.get('WEELLM_OUT_DIR'):
    OUT_DIR = os.environ['WEELLM_OUT_DIR']
elif _IS_KAGGLE:
    OUT_DIR = '/kaggle/working/weellm_studio'
elif _IS_COLAB:
    OUT_DIR = '/content/weellm_studio'
else:
    OUT_DIR = '/tmp/weellm_studio_preview'
os.makedirs(OUT_DIR, exist_ok=True)

def _detect_token():
    tok = (os.environ.get('HF_TOKEN') or '').strip()
    if tok:
        return tok
    if _IS_COLAB:
        try:
            from google.colab import userdata
            return (userdata.get('HF_TOKEN') or '').strip()
        except Exception:
            return ''
    if _IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            return (UserSecretsClient().get_secret('HF_TOKEN') or '').strip()
        except Exception:
            return ''
    return ''
_AUTO_TOKEN = _detect_token()
if _AUTO_TOKEN:
    os.environ.setdefault('HF_TOKEN', _AUTO_TOKEN)
    os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', _AUTO_TOKEN)

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
def _sim_budgets():
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
def _sim_do_unload():
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

def _sim_run_all(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b):
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

def _budgets():
    import psutil
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    ram = psutil.virtual_memory().total / 1e9
    return round(max(4.0, vram - 3.0), 1), round(max(4.0, min(10.0, ram - 4.0)), 1)
def _hub_dir(repo):
    from huggingface_hub import constants
    root = os.environ.get('HF_HUB_CACHE', constants.HF_HUB_CACHE)
    return os.path.join(root, 'models--' + repo.replace('/', '--'), 'blobs')
_TEMP_SUFFIXES = ('.part', '.incomplete', '.lock', '.tmp')
def _du(path):
    t = 0
    for dp, _dn, fn in os.walk(path):
        for f in fn:
            if f.startswith('tmp') or f.endswith(_TEMP_SUFFIXES):
                continue  # in-flight transfer fragments, not real bytes yet
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t
def _repo_bytes(repo, token, patterns=None, exclude=()):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        info = api.model_info(repo, token=token or None, files_metadata=True)
    except TypeError:
        info = api.model_info(repo, token=token or None)
    total, n = 0, 0
    for s in info.siblings:
        name = s.rfilename
        sz = getattr(s, 'size', None) or 0
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        if any(fnmatch.fnmatch(name, e) for e in exclude):
            continue
        total += sz
        n += 1
    return total, n


def _gguf_bytes(repo, filename, token):
    return _repo_bytes(repo, token, patterns=[filename])
def _snap73(n):
    n = max(9, int(n))
    k = max(0, round((n - 5) / 17))
    return 17 * k + 5
def _snap121(n):
    n = max(9, int(n))
    return max(1, ((n - 1 + 4) // 8) * 8 + 1)

# ── download plan + worker (HF Hub only, real byte progress) ───
def _cached_plan_bytes(repo, patterns, exclude, ggufs, token):
    """Bytes of this exact plan already sitting in the local HF cache.

    Reruns must not be told to need the full total again. Base repo: walk
    only the NEWEST snapshot dir (stale revisions do not count) and sum
    planned files, deduped by inode so shared blobs count once. Single
    quant files: ask the Hub cache without downloading anything.
    """
    cached, seen = 0, set()

    def _add(path):
        nonlocal cached
        try:
            st = os.stat(path)
        except OSError:
            return
        key = (st.st_dev, st.st_ino)
        if key in seen:
            return
        seen.add(key)
        cached += st.st_size

    try:
        from huggingface_hub import constants
        root = os.environ.get("HF_HUB_CACHE", constants.HF_HUB_CACHE)
        snap_root = os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots")
        if os.path.isdir(snap_root):
            snaps = [os.path.join(snap_root, s) for s in os.listdir(snap_root)]
            snaps = [s for s in snaps if os.path.isdir(s)]
            if snaps:
                newest = max(snaps, key=lambda s: os.path.getmtime(s))
                for dp, _dn, fn in os.walk(newest):
                    for f in fn:
                        p = os.path.join(dp, f)
                        try:
                            rel = os.path.relpath(p, newest)
                        except ValueError:
                            continue
                        if patterns and not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
                            continue
                        if exclude and any(fnmatch.fnmatch(rel, e) for e in exclude):
                            continue
                        _add(p)
    except Exception:
        pass
    try:
        from huggingface_hub import hf_hub_download
        for grepo, gfile in (ggufs or []):
            try:
                lp = hf_hub_download(repo_id=grepo, filename=gfile,
                                     token=token or None, local_files_only=True)
                _add(lp)
            except Exception:
                pass
    except Exception:
        pass
    return cached


def plan_download(entry, quant, token):
    repo = _rid(entry['repo_url'])
    if entry.get('mirror_url') and not (token or '').strip():
        repo = _rid(entry['mirror_url'])  # non-gated mirror auto-preferred without a token
    ignore = list(entry.get('ignore', []))
    parts, ggufs = [], []
    if quant == 'full':
        total, n = _repo_bytes(repo, token, patterns=entry.get('fetch_patterns'), exclude=ignore)
        parts.append(('%s full repo (%d files)' % (repo, n), total, 'https://huggingface.co/' + repo))
    else:
        ignore = ignore + ['transformer/*.safetensors', 'unconditional_transformer/*.safetensors']
        # COLAB SLIM (free-tier disk): skip weights that are never executed.
        # - prompt_enhancer: the LTX video adapter hard-skips it (dummy part).
        # - text_encoder weights: only when a quant encoder replaces them
        #   (te_gguf / te_safetensors). Configs + tokenizers (*.json) still fetch.
        if entry.get('pipe') == 'i2v':
            ignore = ignore + ['prompt_enhancer/*.safetensors']
        if entry.get('te_gguf') or entry.get('te_safetensors'):
            ignore = ignore + ['text_encoder/*.safetensors', 'text_encoder_2/*.safetensors',
                               'text_encoder_3/*.safetensors', 'text_encoder_4/*.safetensors']
        grepo = _rid(entry['gguf_repo_url'])
        gsize, _ = _gguf_bytes(grepo, quant, token)
        if gsize == 0:
            raise RuntimeError('Quant file not found on Hub: %s/%s' % (grepo, quant))
        ggufs.append((grepo, quant))
        base_total, _ = _repo_bytes(repo, token, patterns=entry.get('fetch_patterns'), exclude=['transformer/*.safetensors', 'unconditional_transformer/*.safetensors'] + ignore)
        parts.append(('Base %s : encoder + VAE + configs (transformer skipped)' % repo, base_total, 'https://huggingface.co/' + repo))
        parts.append(('Transformer ' + quant, gsize, entry['gguf_repo_url'] + '/blob/main/' + quant))
        if entry.get('te_gguf'):
            trepo = _rid(entry['te_gguf'][0])
            tsize, _ = _gguf_bytes(trepo, entry['te_gguf'][1], token)
            ggufs.append((trepo, entry['te_gguf'][1]))
            parts.append(('Text-encoder ' + entry['te_gguf'][1].rsplit('/', 1)[-1], tsize, entry['te_gguf'][0] + '/blob/main/' + entry['te_gguf'][1]))
        if entry.get('te_safetensors'):
            trepo = _rid(entry['te_safetensors'][0])
            tsize, _ = _gguf_bytes(trepo, entry['te_safetensors'][1], token)
            if tsize == 0:
                raise RuntimeError('Quant file not found on Hub: %s/%s' % (trepo, entry['te_safetensors'][1]))
            ggufs.append((trepo, entry['te_safetensors'][1]))
            parts.append(('Text-encoder ' + entry['te_safetensors'][1].rsplit('/', 1)[-1], tsize, entry['te_safetensors'][0] + '/blob/main/' + entry['te_safetensors'][1]))
        total = sum(p[1] for p in parts)
    _excl = list(ignore) if quant == 'full' else (['transformer/*.safetensors', 'unconditional_transformer/*.safetensors'] + ignore)
    cached = _cached_plan_bytes(repo, entry.get('fetch_patterns'), _excl, ggufs, token)
    need = max(0, total - cached)
    free = shutil.disk_usage(os.path.expanduser('~/.cache/huggingface')).free
    if need > free * 0.92:
        raise RuntimeError('Not enough disk: need %s more (%s total, %s already cached), have %s free. Pick a smaller quant or free space.' % (_fmt_gb(need), _fmt_gb(total), _fmt_gb(cached), _fmt_gb(free)))
    if cached > 0:
        ustatus('preflight: %s already cached, only %s left to fetch.' % (_fmt_gb(cached), _fmt_gb(need)))
    return dict(repo=repo, patterns=entry.get('fetch_patterns'), ignore=ignore, ggufs=ggufs, total=total, parts=parts, skipped=ignore,
                desc=' + '.join('%s (%s)' % (p[0], _fmt_gb(p[1])) for p in parts))


def _download_worker(plan, token, state):
    from huggingface_hub import snapshot_download, hf_hub_download
    try:
        paths, before = {}, {}
        for grepo, _gf in plan['ggufs']:
            d = _hub_dir(grepo)
            os.makedirs(d, exist_ok=True)
            before[grepo] = _du(d)
        bd = _hub_dir(plan['repo'])
        os.makedirs(bd, exist_ok=True)
        before[plan['repo']] = _du(bd)
        state['phase'] = 'snapshot'
        box, done = {}, threading.Event()
        def _run():
            try:
                kw = dict(repo_id=plan['repo'], token=token or None, max_workers=8)
                if plan['patterns']:
                    kw['allow_patterns'] = plan['patterns']
                if plan.get('ignore'):
                    kw['ignore_patterns'] = plan['ignore']
                box['snap'] = snapshot_download(**kw)
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=min(4, max(1, len(plan['ggufs'])))) as _ex:
                    _futs = [_ex.submit(hf_hub_download, repo_id=gr, filename=gf, token=token or None) for gr, gf in plan['ggufs']]
                    for _fu in _cf.as_completed(_futs):
                        _fu.result()
            except Exception as e:
                box['error'] = e
            finally:
                done.set()
        threading.Thread(target=_run, daemon=True).start()
        while not done.is_set():
            got = _du(bd) - before[plan['repo']]
            for grepo, _gf in plan['ggufs']:
                got += _du(_hub_dir(grepo)) - before[grepo]
            state['got'] = max(0, got)
            time.sleep(0.5)
        if 'error' in box:
            raise box['error']
        state['snap'] = box.get('snap')
        state['done'] = True
    except Exception as e:
        state['error'] = '%s: %s' % (type(e).__name__, str(e))
        ustatus('download failed — ' + state['error'])

# ── pipeline loader (signature-tolerant across WeeLLM classes) ─
PIPE_CACHE = {'key': None, 'pipe': None}
def _filtered_call(cls, model_dir, wanted):
    sig = inspect.signature(cls.from_pretrained)
    ps = sig.parameters
    variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())
    filt = {k: v for k, v in wanted.items() if (k in ps or variadic)}
    return cls.from_pretrained(model_dir, **filt)
def _load_worker(entry, quant, model_dir, dtype, vram_b, ram_b, state):
    try:
        import weellm
        cls = getattr(weellm, PIPE_CLS[entry['pipe']])
        kw = dict(device='cuda', torch_dtype=dtype, prefetch=(quant == 'full'), cache_to_ram=False, vae_tile_size=512, vram_budget=vram_b, ram_budget=ram_b)
        if quant != 'full':
            kw['transformer_path'] = '%s/%s' % (_rid(entry['gguf_repo_url']), quant)
            kw['prefetch'] = False
            if entry.get('te_gguf'):
                kw['text_encoder_path'] = '%s/%s' % (_rid(entry['te_gguf'][0]), entry['te_gguf'][1])
            if entry.get('te_safetensors'):
                kw['text_encoder_path'] = '%s/%s' % (_rid(entry['te_safetensors'][0]), entry['te_safetensors'][1])
        ustatus('loading %s (%s) …' % (entry['label'], quant))
        pipe = _filtered_call(cls, model_dir, kw)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        state['pipe'] = pipe
        state['done'] = True
        ustatus('pipeline ready: %s' % type(getattr(pipe, '_pipeline', pipe)).__name__)
    except Exception as e:
        state['error'] = '%s: %s' % (type(e).__name__, str(e))
        ustatus('load failed — ' + state['error'])
def do_unload():
    if PIPE_CACHE['pipe'] is not None:
        try:
            del PIPE_CACHE['pipe']
        except Exception:
            pass
        PIPE_CACHE.update(key=None, pipe=None)
    gc.collect()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    ustatus('model unloaded, VRAM released.')
    return status_html('Idle — model unloaded.', 'idle')

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


# ── inference ──────────────────────────────────────────────────
# Ideogram4's native default guidance schedule: a 7.0 taper that drops to 3.0
# for the last three (polish) steps.  Used as a fallback if the installed
# diffusers cannot be introspected for its own default.
_IDEOGRAM4_GUIDANCE_DEFAULT = (7.0,) * 45 + (3.0,) * 3

def _native_supports(pipe, name):
    return name in _native_params(pipe)
def _native_params(pipe):
    """Parameters of the native pipeline's __call__, looked up defensively.

    diffusers wraps ``__call__`` in ``@torch.no_grad()`` / docstring decorators,
    and the wrapper chain can hide the real signature.  Unwrap manually and fall
    back to the raw ``__defaults__`` mapping so a schedule lookup never fails
    silently (a failed lookup means the native default schedule silently leaks
    through and collides with the requested step count).
    """
    fn = getattr(type(getattr(pipe, "_pipeline", pipe)), "__call__", None)
    if fn is None:
        return {}
    seen = 0
    while hasattr(fn, "__wrapped__") and seen < 16:
        fn = fn.__wrapped__
        seen += 1
    try:
        return inspect.signature(fn).parameters
    except Exception:
        pass
    try:
        code = fn.__code__
        names = code.co_varnames[: code.co_argcount]
        defaults = fn.__defaults__ or ()
        out = {}
        for nm, dv in zip(names[len(names) - len(defaults):], defaults):
            out[nm] = dv
        for nm, dv in (getattr(fn, "__kwdefaults__", None) or {}).items():
            out[nm] = dv
        return out
    except Exception:
        return {}
def _native_default(pipe, name):
    params = _native_params(pipe)
    if name not in params:
        return None
    d = params[name]
    d = getattr(d, "default", None)
    if d is inspect.Parameter.empty:
        return None
    return d
def _as_schedule(seq):
    try:
        return [float(v) for v in seq]
    except Exception:
        return None
def _resample_schedule(seq, n):
    seq = [float(v) for v in seq]
    if not seq or n <= 0:
        return seq
    if len(seq) == n:
        return seq
    if n == 1:
        return [seq[0]]
    last = len(seq) - 1
    out = []
    for i in range(n):
        p = i * last / (n - 1)
        lo = int(p)
        hi = min(lo + 1, last)
        f = p - lo
        out.append(seq[lo] * (1 - f) + seq[hi] * f)
    return out
def _guidance_plan(pipe, a):
    """Decide the guidance kwargs for this run.

    Ideogram4 accepts EITHER a constant ``guidance_scale`` OR a per-step
    ``guidance_schedule`` (mutually exclusive), and its native default schedule
    has a fixed length (48) that must equal ``num_inference_steps``.  So the
    schedule always has to be spelled out explicitly.
    """
    n = int(a['steps'])
    sched = _as_schedule(_native_default(pipe, 'guidance_schedule'))
    if sched is None and not _native_supports(pipe, 'guidance_schedule'):
        # Pipeline without a per-step schedule -> plain constant guidance.
        if a['guid'] > 0 and _native_supports(pipe, 'guidance_scale'):
            return {'guidance_scale': float(a['guid'])}, 'scale'
        return {}, 'none'
    if a['guid'] > 0:
        return {'guidance_scale': float(a['guid']), 'guidance_schedule': None}, 'scale'
    base = sched if sched else list(_IDEOGRAM4_GUIDANCE_DEFAULT)
    return {'guidance_schedule': _resample_schedule(base, n)}, 'schedule(%d)' % n
def _run_t2i(pipe, a, cb, use_cb):
    kw = dict(prompt=a['prompt'], height=a['h'], width=a['w'], num_inference_steps=a['steps'], seed=a['seed'])
    gk, gmode = _guidance_plan(pipe, a)
    kw.update(gk)
    ustatus('guidance mode: %s' % gmode)
    if a['neg'] and _native_supports(pipe, 'negative_prompt'):
        kw['negative_prompt'] = a['neg']
    if use_cb and _native_supports(pipe, 'callback_on_step_end'):
        kw['callback_on_step_end'] = cb
    try:
        return pipe.generate(**kw)
    except ValueError as e:
        # The native default schedule has a fixed length: if it slipped through
        # (native default instead of our explicit one) resample it and retry.
        if 'guidance_schedule' not in str(e) or 'guidance_schedule' in kw:
            raise
        base = _as_schedule(_native_default(pipe, 'guidance_schedule'))
        if base is None:
            base = list(_IDEOGRAM4_GUIDANCE_DEFAULT)
        kw['guidance_schedule'] = _resample_schedule(base, int(a['steps']))
        ustatus('guidance schedule length mismatch -> retried with %d-step schedule'
                % len(kw['guidance_schedule']))
        return pipe.generate(**kw)
def _run_video(pipe, entry, a, cb, use_cb):
    n = _snap73(a['frames']) if entry['pipe'] == 't2v' else _snap121(a['frames'])
    if n != a['frames']:
        ustatus('frames snapped %d → %d (architecture requirement)' % (a['frames'], n))
    kw = dict(prompt=a['prompt'], height=a['h'], width=a['w'], num_frames=n, num_inference_steps=a['steps'], seed=a['seed'], no_cache=True)
    if entry['pipe'] == 'i2v':
        kw['image'] = a['start']
    if a['guid'] > 0:
        kw['guidance_scale'] = a['guid']
    if use_cb:
        kw['callback_on_step_end'] = cb
    try:
        return pipe(**kw), n
    except TypeError as e:
        if 'callback_on_step_end' in str(e):
            ustatus('native pipeline drops step callbacks — progress will be coarse.')
            kw.pop('callback_on_step_end', None)
            return pipe(**kw), n
        raise
def _save_video_out(out, entry_label):
    from diffusers.utils import encode_video
    if hasattr(out, 'videos') and out.videos is not None:
        videos = out.videos[0] if isinstance(out.videos, list) else out.videos
    elif hasattr(out, 'video') and out.video is not None:
        videos = out.video[0] if isinstance(out.video, list) else out.video
    elif hasattr(out, 'frames') and out.frames is not None:
        videos = out.frames[0] if isinstance(out.frames, list) else out.frames
    else:
        raise RuntimeError('Unrecognized video output from %s' % entry_label)
    audio = getattr(out, 'audio', None)
    if isinstance(audio, list):
        audio = audio[0] if audio else None
    fps = getattr(out, 'fps', None) or 24
    path = os.path.join(OUT_DIR, 'out_%d.mp4' % int(time.time()))
    try:
        encode_video(videos, fps=fps, output_path=path, audio=audio, audio_sample_rate=getattr(out, 'sampling_rate', 24000))
    except Exception:
        import torchvision.io
        v = videos
        if isinstance(v, torch.Tensor):
            if v.dim() == 4 and v.shape[0] == 3:
                v = v.permute(1, 0, 2, 3)
            if v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                v = (v * 255).clamp(0, 255).to(torch.uint8)
            torchvision.io.write_video(path, v.permute(0, 2, 3, 1), fps=24, audio_array=audio, audio_fps=24000, audio_codec='aac')
        else:
            raise
    return path

# ── master generator (streams status · bars · logs · media) ────
def _run_all_inner(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b):
    entry = MODELS.get(model_key)
    if entry is not None:  # Dropdown returns the label — map it back to the quant value
        _qmap = {lbl: val for val, lbl, _u in entry['quants']}
        quant = _qmap.get(quant, quant)
    if entry is None or not (prompt or '').strip():
        yield status_html('Pick a model and write a prompt first.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    if entry['pipe'] == 'i2v' and start_img is None:
        yield status_html('Image → Video needs a start frame image.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    token = (token or '').strip()
    if entry['gated'] and not token and not entry.get('mirror_url'):
        ustatus('WARNING: %s is gated and has no mirror — a token will be required at download.' % entry['label'])
    seed = int(seed)
    if seed < 0:
        seed = random.randrange(2 ** 31 - 1)
    dtype = torch.float16 if dtype_name == 'float16' else torch.bfloat16
    a = dict(prompt=prompt.strip(), neg=(neg or '').strip(), w=int(w), h=int(h), steps=int(steps), guid=float(guid or 0), seed=seed, frames=int(frames), start=start_img)
    if entry.get('supported') is False:
        _why = entry.get('unsupported_reason', 'This model cannot be loaded by WeeLLM yet.')
        ustatus('refused — ' + _why)
        yield status_html('Cannot run %s: %s' % (entry['label'], _why), 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    ustatus('run: %s · %s · %dx%d · %d steps · seed %d' % (entry['label'], quant, a['w'], a['h'], a['steps'], seed))

    def snap():
        return status_html('Working — see bars and logs below', 'run'), bar_html(dl_state.get('pct', 0), dl_state.get('txt', 'Download')), bar_html(gen_state.get('pct', 0), gen_state.get('txt', 'Generation')), _log_text(), gr.skip(), gr.skip()

    # — phase 1: download (HF Hub only) —
    dl_state = {'pct': 0, 'txt': 'Download — resolving sizes…'}
    gen_state = {'pct': 0, 'txt': 'Generation — waiting'}
    yield snap()
    try:
        plan = plan_download(entry, quant, token)
    except Exception as e:
        _msg = str(e)
        if '401' in _msg or 'Unauthorized' in _msg:
            _msg += ' — this repo is gated: paste an HF token above AND click Agree/Accept on the repo page (%s).' % (_rid(entry['repo_url']))
        ustatus('preflight failed — %s' % _msg)
        yield status_html('Download blocked: %s' % _msg, 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    if not plan['total']:
        ustatus('WARNING: the Hub did not report file sizes — downloading without a progress denominator.')
    for _plabel, _pbytes, _purl in plan['parts']:
        ustatus('fetch plan: %s = %s <%s>' % (_plabel, _fmt_gb(_pbytes), _purl))
    if plan.get('skipped'):
        ustatus('skipped as assumed-unused: ' + ', '.join(plan['skipped']))
    dl = {'got': 0, 'done': False, 'error': None, 'snap': None}
    threading.Thread(target=_download_worker, args=(plan, token, dl), daemon=True).start()
    _warned_over = False
    while not dl['done'] and not dl.get('error'):
        denom = max(plan['total'], dl['got'])  # Hub listing can lag real bytes; never show >100%
        pct = 100.0 * min(1.0, dl['got'] / max(1, denom))
        dl_state.update(pct=pct, txt='Download — %s / %s' % (_fmt_gb(dl['got']), _fmt_gb(denom)))
        if not _warned_over and dl['got'] > plan['total'] and plan['total'] > 0:
            _warned_over = True
            ustatus('note: fetched %s vs %s planned — Hub sizes lagged; extra bytes are harmless cache, continuing.' % (_fmt_gb(dl['got']), _fmt_gb(plan['total'])))
        yield snap()
        time.sleep(0.4)
    if dl.get('error'):
        yield status_html('Download failed: %s' % dl['error'], 'err'), bar_html(dl_state['pct'], 'Download — failed'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    dl_state.update(pct=100, txt='Download — complete (%s)' % _fmt_gb(plan['total']))
    model_dir = dl['snap']
    if entry.get('subfolder'):
        model_dir = os.path.join(model_dir, entry['subfolder'])
    ustatus('weights at %s' % model_dir)
    yield snap()

    # — phase 2: load (cached, instant on repeat runs) —
    key = (model_key, quant, dtype_name, vram_b, ram_b)
    if PIPE_CACHE['key'] != key or PIPE_CACHE['pipe'] is None:
        if PIPE_CACHE['pipe'] is not None:
            ustatus('pipeline config changed — rebuilding automatically, no manual unload needed.')
            do_unload()
        gen_state.update(pct=0, txt='Generation — loading pipeline…')
        yield snap()
        ls, _tries = {}, 0
        while True:
            _tries += 1
            ls = {}
            threading.Thread(target=_load_worker, args=(entry, quant, model_dir, dtype, vram_b, ram_b, ls), daemon=True).start()
            while not ls.get('done') and not ls.get('error'):
                yield snap()
                time.sleep(0.4)
            if ls.get('error') and 'out of memory' in ls['error'].lower() and _tries == 1:
                ustatus('load hit CUDA OOM — purging VRAM and retrying once…')
                do_unload()
                continue
            break
        if ls.get('error'):
            yield status_html('Load failed: %s' % ls['error'], 'err'), bar_html(100, 'Download — complete'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
            return
        PIPE_CACHE.update(key=key, pipe=ls['pipe'])
    else:
        ustatus('reusing cached pipeline (sampling-only change — no rebuild needed).')
    pipe = PIPE_CACHE['pipe']

    # — phase 3: generate —
    gstate, gbox = {'step': 0, 'total': a['steps']}, {}
    def _cb(_p, i, _t, cb_kwargs):
        try:
            gstate['step'] = int(i) + 1
        except Exception:
            pass
        return cb_kwargs
    gen_state.update(pct=1, txt='Generation — denoising 0/%d' % a['steps'])
    yield snap()
    t0 = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    def _infer():
        try:
            if entry['pipe'] == 't2i':
                gbox['img'] = _run_t2i(pipe, a, _cb, True)
            else:
                out, _n = _run_video(pipe, entry, a, _cb, True)
                gbox['vid'] = _save_video_out(out, entry['label'])
        except Exception as e:
            gbox['error'] = '%s: %s' % (type(e).__name__, str(e))
            ustatus('inference failed — ' + gbox['error'])
            ustatus(traceback.format_exc(limit=5)[-1500:])
    threading.Thread(target=_infer, daemon=True).start()
    while 'img' not in gbox and 'vid' not in gbox and 'error' not in gbox:
        pct = 100.0 * min(1.0, gstate['step'] / max(1, gstate['total']))
        gen_state.update(pct=pct, txt='Generation — denoising %d/%d' % (min(gstate['step'], gstate['total']), gstate['total']))
        yield snap()
        time.sleep(0.4)
    if 'error' in gbox:
        if 'out of memory' in gbox['error'].lower():
            do_unload()
            ustatus('VRAM exhausted — model unloaded automatically. Lower resolution/steps or raise the VRAM budget, then Run again.')
        yield status_html('Generation failed: %s' % gbox['error'], 'err'), bar_html(100, 'Download — complete'), bar_html(gen_state['pct'], 'Generation — failed'), _log_text(), gr.skip(), gr.skip()
        return
    dt = time.time() - t0
    try:
        peak = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        peak = -1.0
    img, vid = gbox.get('img'), gbox.get('vid')
    _blank = False
    if img is not None:
        img.save(os.path.join(OUT_DIR, 'img_%d.png' % int(t0)))
        try:
            _ex = img.convert('RGB').getextrema()
            _blank = all((mx - mn) == 0 for mn, mx in _ex)
            if _blank:
                ustatus('WARNING: output is one flat color (classic NaN black-image failure) — re-run with dtype bfloat16.')
        except Exception:
            pass
    ustatus('done in %.0fs · peak %.2f GB · seed %d' % (dt, peak, seed))
    gen_state.update(pct=100, txt='Generation — done in %.0fs' % dt)
    _done_txt = 'Done in %.0fs · peak %.2f GB VRAM · seed %d' % (dt, peak, seed)
    _done_kind = 'ok'
    if _blank:
        _done_txt += ' — WARNING: output looks blank. Re-run with dtype bfloat16.'
        _done_kind = 'err'
    yield status_html(_done_txt, _done_kind), bar_html(100, 'Download — complete'), bar_html(100, gen_state['txt']), _log_text(), gr.update(value=img, visible=img is not None), gr.update(value=vid, visible=vid is not None)

_RUN_LOCK = threading.Lock()

def run_all(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b):
    # One run at a time: a second RUN click while downloading/loading would
    # build a whole second pipeline next to the first and OOM the box.
    if not _RUN_LOCK.acquire(blocking=False):
        yield status_html('Another run is already in progress — please wait for it to finish.', 'err'), bar_html(0, 'Download'), bar_html(0, 'Generation'), _log_text(), gr.skip(), gr.skip()
        return
    try:
        yield from _run_all_inner(task, model_key, quant, token, prompt, neg, w, h, steps, guid, seed, frames, start_img, dtype_name, vram_b, ram_b)
    finally:
        _RUN_LOCK.release()



# ── bind real engine or simulated preview ─────────────────────
if REAL:
    class _BusHandler(logging.Handler):
        def emit(self, record):
            try:
                _push_log('%s: %s' % (record.name, record.getMessage()))
            except Exception:
                pass
    for _lname in ('weellm', 'weellm.cache'):
        _lg = logging.getLogger(_lname)
        _lg.setLevel(logging.INFO)
        _lg.propagate = False
        if not any(isinstance(h, _BusHandler) for h in _lg.handlers):
            _lg.addHandler(_BusHandler())
else:
    _budgets = _sim_budgets
    do_unload = _sim_do_unload
    run_all = _sim_run_all

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

def serve(share=None):
    """Queue + launch the app and return its URL. Non-blocking so a notebook
    cell can call this and then keep itself alive with a heartbeat."""
    try:
        demo.queue(max_size=8)
    except TypeError:
        demo.queue()
    if share is None:
        share = REAL or _IS_KAGGLE or _IS_COLAB
    if share:
        demo.launch(server_name='0.0.0.0', share=True, inline=False, quiet=True,
                    prevent_thread_lock=True, show_error=False)
    else:
        demo.launch(server_name='127.0.0.1', server_port=7860, share=False,
                    show_error=True, prevent_thread_lock=True)
    return getattr(demo, 'share_url', None) or getattr(demo, 'local_url', None)

if __name__ == '__main__':
    _url = serve()
    print(_url or 'http://127.0.0.1:7860', flush=True)
    try:
        while True:  # hold the process (and any hosted session) alive
            time.sleep(60)
    except KeyboardInterrupt:
        pass
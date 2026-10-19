#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPVisualizer.py
双后端：默认legacy手写FFT；--use-librosa-analyze启用librosa STFT‑mel
新增：WmpClassicAmbienceScene (基于逆向 WMP mpvis.dll 的 3D 水波纹引擎)
"""
import argparse
import colorsys
import math
import os
import random
import shutil
import subprocess
import sys
import json
import inspect
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
try:
    import librosa
except ImportError:
    librosa = None

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
BG = (5, 6, 12)
TEXT_COL = (200, 210, 220)
BEAT_COL = (40, 220, 120)
BPM_COL = (255, 220, 90)

class Scene:
    name = "scene"
    def __init__(self, W, H, ctx, params=None): self.W = W; self.H = H; self.ctx = ctx; self.params = params
    def render(self, i, intensity): raise NotImplementedError

# ==========================================================================
# 新增：逆向微软 WMP 的 3D 水波纹引擎 (Ambience)
# ==========================================================================
class WmpClassicAmbienceScene(Scene):
    name = "WMP经典水波"

    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.theta = 0.0
        # 模拟微软底层的小画布 (GDI Memory Surface)
        self.render_w = 320
        self.render_h = 240
        self.cx = self.render_w // 2
        self.cy = self.render_h // 2
        
        # 残影buffer (模拟GDI的拉伸残影)
        self.trail_buf = np.zeros((self.render_h, self.render_w, 3), dtype=np.float32)
        self.decay = 0.88  # 对应汇编里的衰减常数

    def render(self, i, intensity):
        # 1. 对应汇编代码中的 `fcn.48e6c0f6` 正弦波时钟累加
        bass = float(self.ctx["bass"][i])
        # 频率随低频动态变化
        self.theta += 0.05 + bass * 0.15 
        
        # 计算正弦值：(timer / cycle_limit) * (pi/2)
        # 这里用极坐标模拟那种扭曲的涟漪感
        pulse = (math.sin(self.theta) + 1.0) / 2.0
        
        # 2. 在小画布上画波纹 (对应 GDI 的像素级操作)
        # 我们用 Pillow 在小画布上画，比较省 CPU
        small_canvas = Image.new("RGB", (self.render_w, self.render_h), (0, 0, 0))
        draw = ImageDraw.Draw(small_canvas, "RGBA")
        
        # 核心：计算波形半径
        base_r = int(30 + 80 * pulse + 100 * bass)
        
        # 动态颜色 (映射到 WMP 那个经典的蓝白/黑白光晕)
        hue = (self.theta * 0.1) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8)  # 低饱和，高亮度
        col = (int(r * 255), int(g * 255), int(b * 255), 255)
        
        # 画多层同心圆模拟波纹的厚度
        for offset in range(0, 40, 4):
            alpha = max(0, 255 - offset * 6)
            outline_col = (int(r * 255), int(g * 255), int(b * 255), alpha)
            rr = base_r - offset
            if rr > 0:
                draw.ellipse([self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr],
                             outline=outline_col, width=2)
                             
        # 模拟随机粒子闪现 (对应汇编里的 msvcrt.dll_rand)
        for _ in range(int(bass * 20)):
            px = self.cx + random.randint(-base_r, base_r)
            py = self.cy + random.randint(-base_r, base_r)
            draw.point((px, py), fill=(255, 255, 255, 200))

        # 3. 拉伸放大 (完美复刻 StretchBlt)
        # 核心：不是直接画大图，而是把小图强行拉伸铺满大屏，利用插值模糊制造光晕感
        big_canvas = small_canvas.resize((self.W, self.H), Image.BILINEAR)
        
        # 4. 叠加全局残影
        arr = np.asarray(big_canvas, np.float32)
        self.trail_buf *= self.decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)
        
        final_img = Image.fromarray(self.trail_buf.astype(np.uint8))
        
        # 5. 终极魔法：高斯模糊 (模拟老 CRT 屏幕的光晕扩散)
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=4))
        
        return final_img

# ==========================================================================
# 旧版 wmpvis 读写模块
# ==========================================================================
class WmpVisStyle:
    @staticmethod
    def get_template_style():
        return {
            "meta": {"title": "WMPVisualizer Style Template", "version": "1.0"},
            "global": {"fps": 60, "width": 1280, "height": 720, "min_dur": 6.0, "max_dur": 14.0, "fade": 0.6, "seed": 0},
            "presets": [
                {"class_name": "WmpClassicAmbienceScene", "display_name": "WMP经典水波", "enable": True, "params": {}},
                {"class_name": "WaterfallScene", "display_name": "瀑布", "enable": True, "params": {"bands": 160, "rows": 110}},
                {"class_name": "VortexScene", "display_name": "漩涡", "enable": True, "params": {"bands": 180, "rot_speed_base": 0.008, "trail_decay": 0.90}},
                {"class_name": "ParticlesScene", "display_name": "粒子", "enable": True, "params": {"particle_count": 380}},
                {"class_name": "RadialBarsScene", "display_name": "圆环", "enable": True, "params": {"bands": 96, "rot_speed_base": 0.004}},
                {"class_name": "BarsScene", "display_name": "竖条", "enable": True, "params": {"bands": 72}}
            ]
        }

    @staticmethod
    def export_template(filepath: str):
        data = WmpVisStyle.get_template_style()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已导出旧模板：{filepath}", file=sys.stderr)

    @staticmethod
    def load_style(filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

# ==========================================================================
# 通用配置解析器 GenericVisParser
# ==========================================================================
class GenericVisParser:
    @staticmethod
    def get_generic_template():
        return {
            "meta": {"title": "Generic WMP Vis Template", "version": "2.0-generic"},
            "global": {"fps": 60, "width": 1280, "height": 720, "trail": {"enable": True, "decay": 0.91}},
            "layers": [{"type": "rect_bars", "count": 72, "gap": 2, "base_y": 0.78, "max_bar_height": 0.55, "alpha_trail": True}]
        }

    @staticmethod
    def export_generic_template(filepath: str):
        d = GenericVisParser.get_generic_template()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        print(f"✅ 已导出通用模板：{filepath}", file=sys.stderr)

    def __init__(self, w: int, h: int, style_cfg: dict):
        self.W = w
        self.H = h
        self.cfg = style_cfg
        self.global_cfg = style_cfg["global"]
        self.layers = style_cfg["layers"]
        self.trail_enable = self.global_cfg.get("trail", {}).get("enable", False)
        self.trail_decay = self.global_cfg.get("trail", {}).get("decay", 0.92)
        self.trail_buf = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.rot_accum = {}

    def _eval_bind(self, prop_cfg, frame_audio: dict):
        if not isinstance(prop_cfg, dict): return prop_cfg
        base = prop_cfg.get("base", 0.0)
        bind_key = prop_cfg.get("bind", None)
        scale = prop_cfg.get("scale", 1.0)
        if bind_key is None: return base
        v = frame_audio.get(bind_key, 0.0)
        return base + float(v) * scale

    def _render_single_layer(self, layer, frame_idx: int, frame_audio: dict) -> Image.Image:
        canvas = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")
        short_side = min(self.W, self.H)
        ltype = layer["type"]
        lid = layer["_lid"]
        rot_val = self.rot_accum.get(f"rot_{lid}", 0.0)
        if ltype == "rect_bars":
            count = int(layer["count"]); gap = layer.get("gap", 2)
            base_y = int(self.H * layer["base_y"]); max_bar_h = int(self.H * layer["max_bar_height"])
            bw = max(2, (self.W - gap * (count - 1)) // count)
            spec = frame_audio["spec"]; idx_arr = np.linspace(0, len(spec) - 1, count).astype(int)
            for k in range(count):
                sv = float(spec[idx_arr[k]]); bh = int(max_bar_h * sv)
                x = k * (bw + gap); hue = (k / count * 0.85 + frame_idx * 0.02 / 60.0) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.4 + 0.6 * sv)
                col = (int(r * 255), int(g * 255), int(b * 255))
                draw.rectangle([x, base_y - bh, x + bw, base_y], fill=col)
                if layer.get("alpha_trail", True):
                    for j in range(bh):
                        a = int(120 * (1.0 - j / max(bh, 1)))
                        if a <= 0: continue
                        draw.rectangle([x, base_y + j, x + bw, base_y + j + 1], fill=col + (a,))
        elif ltype == "radial_lines":
            count = int(layer["count"])
            r_inner = self._eval_bind(layer["radius_inner"], frame_audio) * short_side
            r_outer_base = layer["radius_outer"]; cx, cy = self.W / 2, self.H / 2
            spec = frame_audio["spec"]; idx_arr = np.linspace(0, len(spec) - 1, count).astype(int)
            for k in range(count):
                sv = float(spec[idx_arr[k]]); ba = {**frame_audio, "spectrum": sv}
                ro = r_inner + self._eval_bind(r_outer_base, ba) * short_side
                ang = rot_val + (k / count) * math.pi * 2
                x0 = cx + math.cos(ang) * r_inner; y0 = cy + math.sin(ang) * r_inner
                x1 = cx + math.cos(ang) * ro; y1 = cy + math.sin(ang) * ro
                hue = (k / count + rot_val * 0.2) % 1.0
                br = self._eval_bind(layer["brightness"], ba)
                r, g, b = colorsys.hsv_to_rgb(hue, layer["saturation"], br)
                lw = max(1, int(self._eval_bind(layer["line_width"], ba)))
                draw.line([x0, y0, x1, y1], fill=(int(r * 255), int(g * 255), int(b * 255), layer["alpha"]), width=lw)
        elif ltype == "ellipse":
            cx = self.W * self._eval_bind(layer["cx"], frame_audio); cy = self.H * self._eval_bind(layer["cy"], frame_audio)
            rr = self._eval_bind(layer["radius"], frame_audio) * short_side; alpha = int(self._eval_bind(layer["alpha"], frame_audio))
            if layer.get("outline", False):
                ww = int(layer["outline_width"])
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 255, 255, alpha), width=ww)
        elif ltype == "particle_system":
            p_count = int(layer["particle_count"])
            if not hasattr(self, f"_p_{lid}"):
                rng = np.random.default_rng(42 + lid)
                setattr(self, f"_p_{lid}", {
                    "px": rng.uniform(0, self.W, p_count).astype(np.float32),
                    "py": rng.uniform(0, self.H, p_count).astype(np.float32),
                    "pz": rng.uniform(0.1, 1.0, p_count).astype(np.float32),
                    "hue": rng.uniform(0, 1, p_count).astype(np.float32)
                })
            pd = getattr(self, f"_p_{lid}")
            px, py, pz, hue_arr = pd["px"], pd["py"], pd["pz"], pd["hue"]
            cxx, cyy = self.W / 2, self.H / 2
            speed = 0.4 + frame_audio["rms"] * 5.0 + frame_audio["bass"] * 5.0 + frame_audio["intensity"] * 3.0
            vx = cxx - px; vy = cyy - py; d = np.sqrt(vx ** 2 + vy ** 2) + 1e-8
            px += vx / d * speed; py += vy / d * speed
            out_mask = (d < 6) | (px < -20) | (px > self.W + 20) | (py < -20) | (py > self.H + 20)
            if np.any(out_mask):
                rng = np.random.default_rng(frame_idx + lid); k = int(out_mask.sum())
                ang = rng.uniform(0, math.pi * 2, k); rr = max(self.W, self.H) * 0.5 + rng.uniform(0, 200, k)
                px[out_mask] = cxx + np.cos(ang) * rr; py[out_mask] = cyy + np.sin(ang) * rr
                hue_arr[out_mask] = rng.uniform(0, 1, k)
            pd["px"], pd["py"] = px, py
            glow_r = int(70 + 220 * frame_audio["bass"])
            draw.ellipse([cxx - glow_r, cyy - glow_r, cxx + glow_r, cyy + glow_r], fill=(80, 140, 255, int(50 + 150 * frame_audio["bass"])))
            for pi in range(p_count):
                z = pz[pi]; rsz = 1.0 + z * 2.0 + frame_audio["treble"] * 2.0
                hu = (hue_arr[pi] + frame_idx * 0.002) % 1.0; cr, cg, cb = colorsys.hsv_to_rgb(hu, 0.85, 0.5 + 0.5 * z)
                al = int(80 + 175 * z); x, y = px[pi], py[pi]
                draw.ellipse([x - rsz, y - rsz, x + rsz, y + rsz], fill=(int(cr * 255), int(cg * 255), int(cb * 255), al))
        elif ltype == "waterfall":
            bands = int(layer["bands"]); rows = int(layer["rows"])
            if not hasattr(self, f"_wfbuf_{lid}"):
                setattr(self, f"_wfbuf_{lid}", np.zeros((rows, bands, 3), dtype=np.uint8))
            wfbuf = getattr(self, f"_wfbuf_{lid}")
            spec = frame_audio["spec"]; idx_arr = np.linspace(0, len(spec) - 1, bands).astype(int)
            vals = np.clip(spec[idx_arr] * (0.7 + 0.6 * frame_audio["intensity"]), 0, 1)
            wfbuf[:-1] = wfbuf[1:]
            stops = np.array([[0.00, 0, 0, 50], [0.15, 0, 0, 160], [0.30, 0, 70, 240], [0.45, 0, 190, 220], [0.60, 70, 240, 130], [0.75, 220, 230, 60], [0.90, 255, 130, 30], [1.00, 255, 50, 40]])
            lut = []
            for vv in vals:
                pos = np.clip(vv, 0, 1); idx_pos = np.searchsorted(stops[:, 0], pos)
                low = stops[max(idx_pos - 1, 0)]; high = stops[min(idx_pos, len(stops) - 1)]
                f = (pos - low[0]) / (high[0] - low[0] + 1e-8)
                rgb = (low[1:] * (1 - f) + high[1:] * f).astype(np.uint8); lut.append(rgb)
            wfbuf[-1] = np.array(lut)
            im = Image.fromarray(wfbuf, "RGB").resize((self.W, self.H), Image.BILINEAR)
            top = im.crop((0, 0, self.W, self.H // 2)); arr = np.asarray(top, np.float32)[::-1]
            grad = np.linspace(1.0, 0.15, self.H - self.H // 2)[:, None, None]
            bot = Image.fromarray(np.clip(arr * grad, 0, 255).astype(np.uint8))
            canvas.paste(top, (0, 0)); canvas.paste(bot, (0, self.H // 2))
        return canvas

    def _update_rotation(self, layer, frame_audio: dict):
        lid = layer["_lid"]
        rot_cfg = layer.get("rotate")
        if isinstance(rot_cfg, dict):
            dr = self._eval_bind(rot_cfg, frame_audio)
            k = f"rot_{lid}"
            self.rot_accum[k] = self.rot_accum.get(k, 0.0) + dr

    def apply_trail(self, image: Image.Image) -> Image.Image:
        if not self.trail_enable: return image
        arr = np.asarray(image, np.float32)
        self.trail_buf *= self.trail_decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)
        return Image.fromarray(self.trail_buf.astype(np.uint8))

    def render_all_layers(self, frame_idx: int, frame_audio: dict) -> Image.Image:
        canvas = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        for lid, ly in enumerate(self.layers):
            ly["_lid"] = lid
            self._update_rotation(ly, frame_audio)
            sub = self._render_single_layer(ly, frame_idx, frame_audio)
            canvas.paste(sub, (0, 0))
        return self.apply_trail(canvas)


# ==========================================================================
# 音频模块
# ==========================================================================
def _decode_ffmpeg(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    line = p.stdout.strip().splitlines()[0]; parts = line.split(",")
    sr = int(float(parts[0])); ch = int(float(parts[1]))
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).copy()
    return data, sr

def load_audio(path):
    if not os.path.isfile(path): raise SystemExit(f"找不到文件: {path}")
    if not FFMPEG or not FFPROBE: raise SystemExit("需要 ffmpeg 和 ffprobe")
    return _decode_ffmpeg(path)

def find_or_create_vocals(audio_path, cache_dir=".vocals_cache"):
    os.makedirs(cache_dir, exist_ok=True); base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_path = os.path.join(cache_dir, base + "_vocals.wav")
    if os.path.isfile(cache_path): return cache_path
    demucs_bin = shutil.which("demucs")
    if not demucs_bin: return None
    tmp_out = os.path.join(cache_dir, "_tmp_" + base); os.makedirs(tmp_out, exist_ok=True)
    cmd = [demucs_bin, "--two-stems", "vocals", "-o", tmp_out, "--float32", "-n", "htdemucs", "-d", "cpu", audio_path]
    try: p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired: shutil.rmtree(tmp_out, ignore_errors=True); return None
    if p.returncode != 0: shutil.rmtree(tmp_out, ignore_errors=True); return None
    vocals_src = None
    for root, _, files in os.walk(tmp_out):
        for fn in files:
            if fn == "vocals.wav": vocals_src = os.path.join(root, fn)
    if not vocals_src: shutil.rmtree(tmp_out, ignore_errors=True); return None
    shutil.move(vocals_src, cache_path); shutil.rmtree(tmp_out, ignore_errors=True)
    return cache_path

def smooth1d(x, k):
    y = np.empty_like(x); cur = 0.0
    for i in range(len(x)): cur += k * (x[i] - cur); y[i] = cur
    return y

def analyze_legacy(mono, sr, fps, fft_size=4096, bands=128, fmin=30.0, fmax=16000.0):
    n = len(mono); hop = sr / float(fps); n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    edges = np.geomspace(max(fmin, 20.0), min(fmax, sr / 2 - 1), bands + 1)
    band_idx = []
    for k in range(bands):
        lo, hi = edges[k], edges[k + 1]; idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0: idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2))])
        band_idx.append(idx)
    spec = np.zeros((n_frames, bands), dtype=np.float32); rms = np.zeros(n_frames, dtype=np.float32)
    pad = fft_size // 2; win_len = int(sr * 0.035)
    for i in range(n_frames):
        c = int((i + 0.5) * hop); a = c - pad; b = a + fft_size
        if a < 0: seg = np.zeros(fft_size, dtype=np.float32); seg[-a:] = mono[:b]
        elif b > n: seg = np.zeros(fft_size, dtype=np.float32); seg[:n - a] = mono[a:]
        else: seg = mono[a:b]
        s = np.abs(np.fft.rfft(seg * win)); s = 20.0 * np.log10(np.maximum(s / (fft_size / 4), 1e-10))
        for k, idx in enumerate(band_idx): spec[i, k] = s[idx].max()
        a2 = max(0, c - win_len // 2); b2 = min(n, c + win_len // 2)
        if b2 > a2: rms[i] = np.sqrt(np.mean(mono[a2:b2] ** 2))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10)); norm = np.clip((spec + 90.0) / 80.0, 0, 1); rms_norm = np.clip((rms_db + 60.0) / 60.0, 0, 1)
    nb = max(1, bands // 8); bass = norm[:, :nb].mean(axis=1); treble = norm[:, -nb:].mean(axis=1)
    rms_s = smooth1d(rms_norm, 0.15); novelty = np.abs(np.diff(rms_s, prepend=rms_s[0])); novelty = smooth1d(novelty, 0.25); novelty = novelty / (novelty.max() + 1e-9)
    return dict(spec=norm, rms=rms_norm, bass=bass, treble=treble, novelty=novelty, n_frames=n_frames, bands=bands, sr=sr, fps=fps)

def detect_bpm(mono, sr, fps, bpm_min=60.0, bpm_max=200.0):
    hop = sr / float(fps); n_frames = max(1, int(math.ceil(len(mono) / hop))); energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a = int(i * hop); b = int((i + 1) * hop); seg = mono[a:b]
        if len(seg): energy[i] = np.sum(seg ** 2)
    onset = np.diff(energy, prepend=energy[0]); onset = np.maximum(onset, 0); onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]; ac = ac / (ac[0] + 1e-9)
    lag_min = max(1, int(fps * 60.0 / bpm_max)); lag_max = min(len(ac) - 1, int(fps * 60.0 / bpm_min))
    if lag_max <= lag_min: return 120.0
    best_lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1])); bpm = 60.0 * fps / best_lag
    while bpm < bpm_min: bpm *= 2
    while bpm > bpm_max: bpm /= 2
    return float(bpm)

def build_timeline(ctx, min_dur, max_dur, scene_cls_list=None, seed=0):
    rng = np.random.default_rng(seed); fps = ctx["fps"]; n = ctx["n_frames"]; nov = ctx["novelty"]
    min_f = int(min_dur * fps); max_f = int(max_dur * fps); timeline = []; i = 0; last_scene = -1
    if scene_cls_list is None: scene_cls_list = SCENE_CLASSES
    scene_idx = int(rng.integers(0, len(scene_cls_list)))
    while i < n:
        end_min = min(n, i + min_f); end_max = min(n, i + max_f)
        if end_max <= end_min: timeline.append((i, scene_idx)); break
        window = nov[end_min:end_max]; peak = int(np.argmax(window)) + end_min if len(window) > 0 else end_max
        timeline.append((i, scene_idx)); i = peak
        choices = [k for k in range(len(scene_cls_list)) if k != last_scene]; scene_idx = int(rng.choice(choices)); last_scene = scene_idx
    return timeline

# ==========================================================================
# 原始 Scene 类
# ==========================================================================

class WaterfallScene(Scene): 
    name = "瀑布"
    def __init__(self, W, H, ctx, params=None): super().__init__(W, H, ctx, params); self.bands = 160; self.rows = 110; self.buf = np.zeros((self.rows, self.bands, 3), dtype=np.uint8)
    def render(self, i, intensity):
        spec = self.ctx["spec"][i]; idx = np.linspace(0, len(spec) - 1, self.bands).astype(int); vals = np.clip(spec[idx] * (0.7 + 0.6 * intensity), 0, 1)
        self.buf[:-1] = self.buf[1:]; self.buf[-1] = np.array([int(v*255) for v in vals])[:, None].repeat(3, axis=1)
        return Image.fromarray(self.buf, "RGB").resize((self.W, self.H), Image.BILINEAR)

class VortexScene(Scene): 
    name = "漩涡"
    def __init__(self, W, H, ctx, params=None): super().__init__(W, H, ctx, params); self.bands = 180; self.cx = W // 2; self.cy = H // 2; self.rot = 0.0
    def render(self, i, intensity):
        canvas = Image.new("RGB", (self.W, self.H), (4, 5, 12)); draw = ImageDraw.Draw(canvas, "RGBA")
        self.rot += 0.01
        for k in range(self.bands):
            ang = self.rot + k / self.bands * 2 * math.pi; v = float(self.ctx["spec"][i][k % len(self.ctx["spec"][i])])
            draw.line([self.cx, self.cy, self.cx + math.cos(ang)*200*v, self.cy + math.sin(ang)*200*v], fill=(255,255,255,100), width=2)
        return canvas

class ParticlesScene(Scene): 
    name = "粒子"
    def __init__(self, W, H, ctx, params=None): super().__init__(W, H, ctx, params)
    def render(self, i, intensity): return Image.new("RGB", (self.W, self.H), (0,0,0))

class RadialBarsScene(Scene): 
    name = "圆环"
    def __init__(self, W, H, ctx, params=None): super().__init__(W, H, ctx, params)
    def render(self, i, intensity): return Image.new("RGB", (self.W, self.H), (0,0,0))

class BarsScene(Scene): 
    name = "竖条"
    def __init__(self, W, H, ctx, params=None): super().__init__(W, H, ctx, params)
    def render(self, i, intensity): return Image.new("RGB", (self.W, self.H), (0,0,0))


# 注册场景
SCENE_CLASSES = [WmpClassicAmbienceScene, WaterfallScene, VortexScene, ParticlesScene, RadialBarsScene, BarsScene]
SCENE_CLASS_MAP = {
    "WmpClassicAmbienceScene": WmpClassicAmbienceScene,
    "WaterfallScene": WaterfallScene,
    "VortexScene": VortexScene,
    "ParticlesScene": ParticlesScene,
    "RadialBarsScene": RadialBarsScene,
    "BarsScene": BarsScene
}

# ==========================================================================
# 渲染与主循环
# ==========================================================================
def render(ctx, timeline, cfg, scene_list, out_path):
    W = cfg["width"] // 2 * 2; H = cfg["height"] // 2 * 2; fps = ctx["fps"]; n = ctx["n_frames"]
    fade_frames = int(cfg["fade"] * fps)
    scenes = [cls(W, H, ctx) for cls in scene_list] if inspect.isclass(scene_list[0]) else scene_list
    scene_of = np.zeros(n, dtype=np.int32)
    for k, (start, sidx) in enumerate(timeline):
        end = timeline[k + 1][0] if (k + 1 < len(timeline)) else n
        scene_of[start:end] = sidx
    intensity = np.clip(0.5 * ctx["rms"] + 0.5 * ctx["novelty"], 0, 1); intensity = smooth1d(intensity, 0.2)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", cfg["input"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); report = max(1, n // 100); prev_scene = -1; prev_frame = None; fade_left = 0
    try:
        for i in range(n):
            sidx = int(scene_of[i]); cur = scenes[sidx].render(i, float(intensity[i]))
            if sidx != prev_scene: fade_left = fade_frames; prev_scene = sidx
            if fade_left > 0 and prev_frame is not None:
                a = 1.0 - fade_left / max(fade_frames, 1); cur = Image.blend(prev_frame, cur, a); fade_left -= 1
            proc.stdin.write(cur.tobytes("raw", "RGB")); prev_frame = cur
            if i % report == 0: print(f"\r  渲染 {i * 100 // n:3d}%", end="", file=sys.stderr, flush=True)
    finally:
        try: proc.stdin.close()
        except: pass
        proc.wait()
    print("\r渲染完成 ✔", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60); ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-dur", type=float, default=6.0); ap.add_argument("--max-dur", type=float, default=14.0); ap.add_argument("--fade", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", type=str, default=None); ap.add_argument("--export-style", type=str, default=None)
    a = ap.parse_args()
    if a.export_style: WmpVisStyle.export_template(a.export_style); return
    
    data, sr = load_audio(a.input); mono_audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    ctx = analyze_legacy(mono_audio, sr, a.fps)
    out_path = a.output or "output_wmp.mp4"
    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height, fade=a.fade)
    
    if a.style:
        style_data = WmpVisStyle.load_style(a.style)
        scene_list = [SCENE_CLASS_MAP[p["class_name"]] for p in style_data["presets"] if p.get("enable", True)]
        timeline = build_timeline(ctx, a.min_dur, a.max_dur, scene_cls_list=scene_list, seed=a.seed)
        render(ctx, timeline, cfg, scene_list, out_path)
    else:
        timeline = build_timeline(ctx, a.min_dur, a.max_dur, seed=a.seed)
        render(ctx, timeline, cfg, SCENE_CLASSES, out_path)

if __name__ == "__main__":
    main()

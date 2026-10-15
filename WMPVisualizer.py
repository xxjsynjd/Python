#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPVisualizer.py
模式:
1. 不指定--style / --generic-style : 使用原始硬编码5个场景
2. --style xxx.wmpvis : 旧模式，对原有Scene类做参数覆盖
3. --generic-style xxx.wmpvis : 【通用解析器模式】全部动画由配置文件驱动，无硬编码场景逻辑

新增命令:
    --export-style FILE           导出旧版模板
    --export-generic-style FILE  导出通用解析器模板
    --generic-style FILE          使用通用配置解析器

用法示例:
    # 原版
    python WMPVisualizer.py test.mp3 -o out.mp4

    # 旧配置模式
    python WMPVisualizer.py test.mp3 --style template.wmpvis -o out.mp4

    # 通用解析器模式
    python WMPVisualizer.py test.mp3 --generic-style generic_template.wmpvis -o gen.mp4

    # 导出通用模板
    python WMPVisualizer.py --export-generic-style generic_template.wmpvis
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
from PIL import Image, ImageDraw

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
BG = (5, 6, 12)
TEXT_COL = (200, 210, 220)
BEAT_COL = (40, 220, 120)
BPM_COL = (255, 220, 90)

# ==========================================================================
# 旧版 wmpvis 读写模块（原有保留）
# ==========================================================================
class WmpVisStyle:
    @staticmethod
    def get_template_style():
        return {
            "meta": {
                "title": "WMPVisualizer Style Template",
                "version": "1.0",
                "note": "不使用此配置时不要传--style"
            },
            "global": {
                "fps": 60,
                "width": 1280,
                "height": 720,
                "min_dur": 6.0,
                "max_dur": 14.0,
                "fade": 0.6,
                "seed": 0
            },
            "presets": [
                {"class_name":"WaterfallScene","display_name":"瀑布","enable":True,"params":{"bands":160,"rows":110}},
                {"class_name":"VortexScene","display_name":"漩涡","enable":True,"params":{"bands":180,"rot_speed_base":0.008,"trail_decay":0.90}},
                {"class_name":"ParticlesScene","display_name":"粒子","enable":True,"params":{"particle_count":380}},
                {"class_name":"RadialBarsScene","display_name":"圆环","enable":True,"params":{"bands":96,"rot_speed_base":0.004}},
                {"class_name":"BarsScene","display_name":"竖条","enable":True,"params":{"bands":72}}
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
# 【新增】通用配置解析器 GenericVisParser
# ==========================================================================
class GenericVisParser:
    @staticmethod
    def get_generic_template():
        return {
            "meta": {"title":"Generic WMP Vis Template","version":"2.0-generic"},
            "global": {
                "fps":60,"width":1280,"height":720,
                "trail":{"enable":True,"decay":0.91}
            },
            "layers":[
                {
                    "type":"rect_bars",
                    "count":72,
                    "gap":2,
                    "base_y":0.78,
                    "max_bar_height":0.55,
                    "alpha_trail":True
                }
            ]
        }

    @staticmethod
    def export_generic_template(filepath:str):
        d = GenericVisParser.get_generic_template()
        with open(filepath,"w",encoding="utf-8") as f:
            json.dump(d,f,indent=2,ensure_ascii=False)
        print(f"✅ 已导出通用模板：{filepath}",file=sys.stderr)

    def __init__(self, w:int, h:int, style_cfg:dict):
        self.W = w
        self.H = h
        self.cfg = style_cfg
        self.global_cfg = style_cfg["global"]
        self.layers = style_cfg["layers"]
        self.trail_enable = self.global_cfg.get("trail",{}).get("enable",False)
        self.trail_decay = self.global_cfg.get("trail",{}).get("decay",0.92)
        self.trail_buf = np.zeros((self.H, self.W,3), dtype=np.float32)
        self.rot_accum = {}

    def _eval_bind(self, prop_cfg, frame_audio:dict):
        """
        解析绑定: {"base":xx,"bind":"xxx","scale":xx}
        frame_audio: {"spec":ndarray,"rms":float,"bass":float,"treble":float,"novelty":float,"intensity":float}
        """
        if not isinstance(prop_cfg, dict):
            return prop_cfg
        base = prop_cfg.get("base",0.0)
        bind_key = prop_cfg.get("bind", None)
        scale = prop_cfg.get("scale",1.0)
        if bind_key is None:
            return base
        v = 0.0
        if bind_key == "intensity": v = frame_audio["intensity"]
        elif bind_key == "rms": v = frame_audio["rms"]
        elif bind_key == "bass": v = frame_audio["bass"]
        elif bind_key == "treble": v = frame_audio["treble"]
        elif bind_key == "novelty": v = frame_audio["novelty"]
        return base + float(v)*scale

    def render_frame(self, frame_idx:int, frame_audio:dict) -> Image.Image:
        """
        frame_audio: {spec:np.ndarray(128,), rms, bass, treble, novelty, intensity}
        返回PIL Image
        """
        # 更新旋转累加器
        for lidx,layer in enumerate(self.layers):
            rot_cfg = layer.get("rotate")
            if isinstance(rot_cfg, dict):
                dr = self._eval_bind(rot_cfg, frame_audio)
                k = f"rot_{lidx}"
                self.rot_accum[k] = self.rot_accum.get(k,0.0)+dr

        canvas = Image.new("RGB",(self.W,self.H),(0,0,0))
        draw = ImageDraw.Draw(canvas,"RGBA")
        short_side = min(self.W,self.H)

        for lidx,layer in enumerate(self.layers):
            ltype = layer["type"]
            rot_val = self.rot_accum.get(f"rot_{lidx}",0.0)

            if ltype == "rect_bars":
                count = int(layer["count"])
                gap = layer.get("gap",2)
                base_y = int(self.H * layer["base_y"])
                max_bar_h = int(self.H * layer["max_bar_height"])
                bw = max(2, (self.W - gap*(count-1)) // count)
                spec = frame_audio["spec"]
                idx_arr = np.linspace(0, len(spec)-1, count).astype(int)
                for k in range(count):
                    sv = float(spec[idx_arr[k]])
                    bh = int(max_bar_h * sv)
                    x = k*(bw+gap)
                    hue = (k/count*0.85 + frame_idx*0.02/60.0) %1.0
                    r,g,b = colorsys.hsv_to_rgb(hue,0.85, 0.4+0.6*sv)
                    col = (int(r*255),int(g*255),int(b*255))
                    draw.rectangle([x, base_y-bh, x+bw, base_y], fill=col)
                    if layer.get("alpha_trail",True):
                        for j in range(bh):
                            a = int(120*(1.0 - j/max(bh,1)))
                            if a<=0: continue
                            draw.rectangle([x,base_y+j, x+bw, base_y+j+1], fill=col+(a,))

            elif ltype == "radial_lines":
                count = int(layer["count"])
                r_inner = self._eval_bind(layer["radius_inner"], frame_audio)*short_side
                r_outer_base = layer["radius_outer"]
                cx, cy = self.W/2, self.H/2
                spec = frame_audio["spec"]
                idx_arr = np.linspace(0, len(spec)-1, count).astype(int)
                for k in range(count):
                    sv = float(spec[idx_arr[k]])
                    ba = {**frame_audio, "spectrum":sv}
                    ro = r_inner + self._eval_bind(r_outer_base, ba)*short_side
                    ang = rot_val + (k/count)*math.pi*2
                    x0 = cx + math.cos(ang)*r_inner
                    y0 = cy + math.sin(ang)*r_inner
                    x1 = cx + math.cos(ang)*ro
                    y1 = cy + math.sin(ang)*ro
                    hue = (k/count + rot_val*0.2) %1.0
                    br = self._eval_bind(layer["brightness"], ba)
                    r,g,b = colorsys.hsv_to_rgb(hue, layer["saturation"], br)
                    lw = max(1,int(self._eval_bind(layer["line_width"], ba)))
                    draw.line([x0,y0,x1,y1], fill=(int(r*255),int(g*255),int(b*255), layer["alpha"]), width=lw)

            elif ltype == "ellipse":
                cx = self.W * self._eval_bind(layer["cx"], frame_audio)
                cy = self.H * self._eval_bind(layer["cy"], frame_audio)
                rr = self._eval_bind(layer["radius"], frame_audio)*short_side
                alpha = int(self._eval_bind(layer["alpha"], frame_audio))
                if layer.get("outline",False):
                    ww = int(layer["outline_width"])
                    draw.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], outline=(255,255,255,alpha), width=ww)

            elif ltype == "particle_system":
                p_count = int(layer["particle_count"])
                if not hasattr(self, f"_p_{lidx}"):
                    rng = np.random.default_rng(42+lidx)
                    setattr(self,f"_p_{lidx}",{
                        "px":rng.uniform(0,self.W,p_count).astype(np.float32),
                        "py":rng.uniform(0,self.H,p_count).astype(np.float32),
                        "pz":rng.uniform(0.1,1.0,p_count).astype(np.float32),
                        "hue":rng.uniform(0,1,p_count).astype(np.float32)
                    })
                pd = getattr(self,f"_p_{lidx}")
                px,py,pz,hue_arr = pd["px"],pd["py"],pd["pz"],pd["hue"]
                cxx, cyy = self.W/2, self.H/2
                speed = 0.4 + frame_audio["rms"]*5.0 + frame_audio["bass"]*5.0 + frame_audio["intensity"]*3.0
                vx = cxx - px
                vy = cyy - py
                d = np.sqrt(vx**2+vy**2)+1e-8
                px += vx/d * speed
                py += vy/d * speed
                out_mask = (d<6)|(px<-20)|(px>self.W+20)|(py<-20)|(py>self.H+20)
                if np.any(out_mask):
                    rng = np.random.default_rng(frame_idx+lidx)
                    k = int(out_mask.sum())
                    ang = rng.uniform(0, math.pi*2, k)
                    rr = max(self.W,self.H)*0.5 + rng.uniform(0,200,k)
                    px[out_mask] = cxx + np.cos(ang)*rr
                    py[out_mask] = cyy + np.sin(ang)*rr
                    hue_arr[out_mask] = rng.uniform(0,1,k)
                pd["px"],pd["py"] = px,py
                glow_r = int(70 + 220*frame_audio["bass"])
                draw.ellipse([cxx-glow_r, cyy-glow_r, cxx+glow_r, cyy+glow_r], fill=(80,140,255,int(50+150*frame_audio["bass"])))
                for pi in range(p_count):
                    z = pz[pi]
                    rsz = 1.0 + z*2.0 + frame_audio["treble"]*2.0
                    hu = (hue_arr[pi] + frame_idx*0.002) %1.0
                    cr,cg,cb = colorsys.hsv_to_rgb(hu,0.85,0.5+0.5*z)
                    al = int(80+175*z)
                    x,y = px[pi],py[pi]
                    draw.ellipse([x-rsz, y-rsz, x+rsz, y+rsz], fill=(int(cr*255),int(cg*255),int(cb*255),al))

            elif ltype == "waterfall":
                bands = int(layer["bands"])
                rows = int(layer["rows"])
                if not hasattr(self,f"_wfbuf_{lidx}"):
                    setattr(self,f"_wfbuf_{lidx}", np.zeros((rows,bands,3),dtype=np.uint8))
                wfbuf = getattr(self,f"_wfbuf_{lidx}")
                spec = frame_audio["spec"]
                idx_arr = np.linspace(0,len(spec)-1,bands).astype(int)
                vals = np.clip(spec[idx_arr]*(0.7+0.6*frame_audio["intensity"]),0,1)
                wfbuf[:-1] = wfbuf[1:]
                stops = np.array([[0.00,0,0,50],[0.15,0,0,160],[0.30,0,70,240],[0.45,0,190,220],[0.60,70,240,130],[0.75,220,230,60],[0.90,255,130,30],[1.00,255,50,40]])
                lut = []
                for vv in vals:
                    pos = np.clip(vv,0,1)
                    idx_pos = np.searchsorted(stops[:,0],pos)
                    low = stops[max(idx_pos-1,0)]
                    high = stops[min(idx_pos, len(stops)-1)]
                    f = (pos-low[0])/(high[0]-low[0]+1e-8)
                    rgb = (low[1:]*(1-f)+high[1:]*f).astype(np.uint8)
                    lut.append(rgb)
                wfbuf[-1] = np.array(lut)
                im = Image.fromarray(wfbuf,"RGB").resize((self.W,self.H),Image.BILINEAR)
                top = im.crop((0,0,self.W,self.H//2))
                arr = np.asarray(top,np.float32)[::-1]
                grad = np.linspace(1.0,0.15,self.H-self.H//2)[:,None,None]
                bot = Image.fromarray(np.clip(arr*grad,0,255).astype(np.uint8))
                canvas.paste(top,(0,0))
                canvas.paste(bot,(0,self.H//2))

        # 残影缓冲区
        if self.trail_enable:
            arr = np.asarray(canvas, np.float32)
            self.trail_buf *= self.trail_decay
            self.trail_buf += arr
            self.trail_buf = np.clip(self.trail_buf,0,255)
            canvas = Image.fromarray(self.trail_buf.astype(np.uint8))
        return canvas

# ==========================================================================
# 音频模块（原始完全保留）
# ==========================================================================
def _decode_ffmpeg(path):
    p = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    line = p.stdout.strip().splitlines()[0]
    parts = line.split(",")
    sr = int(float(parts[0]))
    ch = int(float(parts[1]))
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path,
         "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).copy()
    return data, sr

def load_audio(path):
    if not os.path.isfile(path):
        raise SystemExit(f"找不到文件: {path}")
    if not FFMPEG or not FFPROBE:
        raise SystemExit("需要 ffmpeg 和 ffprobe")
    return _decode_ffmpeg(path)

def find_or_create_vocals(audio_path, cache_dir=".vocals_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_path = os.path.join(cache_dir, base + "_vocals.wav")
    if os.path.isfile(cache_path):
        print(f"      缓存命中: {cache_path}", file=sys.stderr)
        return cache_path
    demucs_bin = shutil.which("demucs")
    if not demucs_bin:
        print("      未找到 demucs，跳过分离", file=sys.stderr)
        return None
    print("      正在分离人声...", file=sys.stderr)
    tmp_out = os.path.join(cache_dir, "_tmp_" + base)
    os.makedirs(tmp_out, exist_ok=True)
    cmd = [demucs_bin, "--two-stems", "vocals",
           "-o", tmp_out, "--float32",
           "-n", "htdemucs", "-d", "cpu", audio_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        print("      分离超时", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None
    if p.returncode != 0:
        print(f"      分离失败: {p.stderr.strip()[-300:]}", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None
    vocals_src = None
    inst_src = None
    for root, _, files in os.walk(tmp_out):
        for fn in files:
            if fn == "vocals.wav":
                vocals_src = os.path.join(root, fn)
            elif fn == "no_vocals.wav":
                inst_src = os.path.join(root, fn)
    if not vocals_src:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None
    shutil.move(vocals_src, cache_path)
    if inst_src:
        try:
            shutil.move(inst_src, os.path.join(cache_dir, base + "_instrumental.wav"))
        except Exception:
            pass
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"      人声已缓存: {cache_path}", file=sys.stderr)
    return cache_path

def smooth1d(x, k):
    y = np.empty_like(x)
    cur = 0.0
    for i in range(len(x)):
        cur += k * (x[i] - cur)
        y[i] = cur
    return y

def analyze(mono, sr, fps, fft_size=4096, bands=128, fmin=30.0, fmax=16000.0):
    n = len(mono)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    edges = np.geomspace(max(fmin, 20.0), min(fmax, sr / 2 - 1), bands + 1)
    band_idx = []
    for k in range(bands):
        lo, hi = edges[k], edges[k + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2))])
        band_idx.append(idx)
    spec = np.zeros((n_frames, bands), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    pad = fft_size // 2
    win_len = int(sr * 0.035)
    for i in range(n_frames):
        c = int((i + 0.5) * hop)
        a = c - pad
        b = a + fft_size
        if a < 0:
            seg = np.zeros(fft_size, dtype=np.float32)
            seg[-a:] = mono[:b]
        elif b > n:
            seg = np.zeros(fft_size, dtype=np.float32)
            seg[:n - a] = mono[a:]
        else:
            seg = mono[a:b]
        s = np.abs(np.fft.rfft(seg * win))
        s = 20.0 * np.log10(np.maximum(s / (fft_size / 4), 1e-10))
        for k, idx in enumerate(band_idx):
            spec[i, k] = s[idx].max()
        a2 = max(0, c - win_len // 2)
        b2 = min(n, c + win_len // 2)
        if b2 > a2:
            rms[i] = np.sqrt(np.mean(mono[a2:b2] ** 2))
        if i % 400 == 0:
            print(f"\r  分析 {i * 100 // n_frames:3d}%", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    norm = np.clip((spec + 90.0) / 80.0, 0, 1)
    rms_norm = np.clip((rms_db + 60.0) / 60.0, 0, 1)
    nb = max(1, bands // 8)
    bass = norm[:, :nb].mean(axis=1)
    treble = norm[:, -nb:].mean(axis=1)
    rms_s = smooth1d(rms_norm, 0.15)
    novelty = np.abs(np.diff(rms_s, prepend=rms_s[0]))
    novelty = smooth1d(novelty, 0.25)
    novelty = novelty / (novelty.max() + 1e-9)
    return dict(spec=norm, rms=rms_norm, bass=bass, treble=treble, novelty=novelty, n_frames=n_frames, bands=bands, sr=sr, fps=fps)

def detect_bpm(mono, sr, fps, bpm_min=60.0, bpm_max=200.0):
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(len(mono) / hop)))
    energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a = int(i * hop)
        b = int((i + 1) * hop)
        seg = mono[a:b]
        if len(seg):
            energy[i] = np.sum(seg ** 2)
    onset = np.diff(energy, prepend=energy[0])
    onset = np.maximum(onset, 0)
    onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset)-1:]
    ac = ac / (ac[0] + 1e-9)
    lag_min = max(1, int(fps * 60.0 / bpm_max))
    lag_max = min(len(ac)-1, int(fps * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return 120.0
    best_lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1]))
    bpm = 60.0 * fps / best_lag
    while bpm < bpm_min:
        bpm *= 2
    while bpm > bpm_max:
        bpm /= 2
    return float(bpm)

# ==========================================================================
# 原始Scene类（完整保留，无改动逻辑）
# ==========================================================================
class Scene:
    name = "scene"
    def __init__(self, W, H, ctx, params=None):
        self.W = W
        self.H = H
        self.ctx = ctx
        self.params = params
    def render(self, i, intensity):
        raise NotImplementedError

class WaterfallScene(Scene):
    name = "瀑布"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.bands = min(160, ctx["bands"])
        self.rows = 110
        if self.params is not None:
            self.bands = self.params.get("bands", self.bands)
            self.rows = self.params.get("rows", self.rows)
        self.buf = np.zeros((self.rows, self.bands, 3), dtype=np.uint8)
        self.lut = self._make_lut()
    def _make_lut(self, n=256):
        stops = np.array([[0.00,0,0,50],[0.15,0,0,160],[0.30,0,70,240],[0.45,0,190,220],[0.60,70,240,130],[0.75,220,230,60],[0.90,255,130,30],[1.00,255,50,40]], dtype=np.float64)
        t = np.linspace(0,1,n)
        lut = np.empty((n,3),dtype=np.uint8)
        for k in range(3):
            lut[:,k] = np.interp(t, stops[:,0], stops[:,1+k])
        return lut
    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0, len(spec)-1, self.bands).astype(int)
        vals = np.clip(spec[idx]*(0.7+0.6*intensity),0,1)
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = self.lut[(vals*255).astype(np.uint8)]
        img = Image.fromarray(self.buf,"RGB").resize((self.W,self.H),Image.BILINEAR)
        top = img.crop((0,0,self.W,self.H//2))
        arr = np.asarray(top,np.float32)[::-1]
        grad = np.linspace(1.0,0.15,self.H-self.H//2)[:,None,None]
        bot = Image.fromarray(np.clip(arr*grad,0,255).astype(np.uint8))
        canvas = Image.new("RGB",(self.W,self.H))
        canvas.paste(top,(0,0))
        canvas.paste(bot,(0,self.H//2))
        return canvas

class VortexScene(Scene):
    name = "漩涡"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.bands =180
        self.rot_speed_base=0.008
        self.trail_decay=0.90
        if self.params is not None:
            self.bands = self.params.get("bands",self.bands)
            self.rot_speed_base = self.params.get("rot_speed_base",self.rot_speed_base)
            self.trail_decay = self.params.get("trail_decay",self.trail_decay)
        self.cx = W//2
        self.cy = H//2
        self.max_r = int(min(W,H)*0.45)
        self.rot=0.0
        self.buf=None
    def render(self,i,intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0,len(spec)-1,self.bands).astype(int)
        vals = spec[idx]
        bass = self.ctx["bass"][i]
        treble = self.ctx["treble"][i]
        self.rot += self.rot_speed_base + (treble*0.06 + intensity*0.03)
        if self.buf is None:
            self.buf = np.zeros((self.H,self.W,3),dtype=np.uint8)
        self.buf = (self.buf*(self.trail_decay-0.05*intensity)).astype(np.uint8)
        canvas = Image.fromarray(self.buf,"RGB")
        draw = ImageDraw.Draw(canvas,"RGBA")
        n=self.bands
        for k in range(n):
            ang = self.rot + k/n*2*math.pi
            v = float(vals[k])*(0.6+0.6*intensity)
            r0 = self.max_r*0.20
            r1 = self.max_r*(0.28+0.72*v)
            x0=self.cx+math.cos(ang)*r0
            y0=self.cy+math.sin(ang)*r0
            x1=self.cx+math.cos(ang)*r1
            y1=self.cy+math.sin(ang)*r1
            hue = (k/n + self.rot*0.1)%1.0
            r,g,b = colorsys.hsv_to_rgb(hue,0.9,0.3+0.7*v)
            draw.line([x0,y0,x1,y1], fill=(int(r*255),int(g*255),int(b*255),200), width=max(1,int(1+v*3)))
        ring_r = self.max_r*(0.15+0.5*bass)
        draw.ellipse([self.cx-ring_r,self.cy-ring_r,self.cx+ring_r,self.cy+ring_r], outline=(255,255,255,int(80+150*bass)), width=2)
        self.buf = np.asarray(canvas,np.uint8).copy()
        return canvas

class ParticlesScene(Scene):
    name = "粒子"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.n = 380
        if self.params is not None:
            self.n = self.params.get("particle_count",self.n)
        rng = np.random.default_rng(42)
        self.px = rng.uniform(0,W,self.n).astype(np.float32)
        self.py = rng.uniform(0,H,self.n).astype(np.float32)
        self.pz = rng.uniform(0.1,1.0,self.n).astype(np.float32)
        self.hue = rng.uniform(0,1,self.n).astype(np.float32)
        self.cx = W/2
        self.cy = H/2
    def render(self,i,intensity):
        rms = float(self.ctx["rms"][i])
        bass = float(self.ctx["bass"][i])
        treble = float(self.ctx["treble"][i])
        speed = 0.4 + rms*5.0 + bass*5.0 + intensity*3.0
        vx = self.cx-self.px
        vy = self.cy-self.py
        d = np.sqrt(vx**2+vy**2)+1e-3
        self.px += vx/d*speed
        self.py += vy/d*speed
        out = (d<6)|(self.px<-20)|(self.px>self.W+20)|(self.py<-20)|(self.py>self.H+20)
        if out.any():
            rng = np.random.default_rng(i)
            k=int(out.sum())
            ang = rng.uniform(0,2*math.pi,k)
            r = max(self.W,self.H)*0.5 + rng.uniform(0,200,k)
            self.px[out] = self.cx + np.cos(ang)*r
            self.py[out] = self.cy + np.sin(ang)*r
            self.hue[out] = rng.uniform(0,1,k)
        canvas = Image.new("RGB",(self.W,self.H),(4,4,10))
        draw = ImageDraw.Draw(canvas,"RGBA")
        glow = int(50+150*bass)
        gr = int(70+220*bass)
        draw.ellipse([self.cx-gr,self.cy-gr,self.cx+gr,self.cy+gr], fill=(80,140,255,glow))
        for k in range(self.n):
            z=self.pz[k]
            rsz =1.0+z*2.0+treble*2.0
            hue=(self.hue[k]+i*0.002)%1.0
            cr,cg,cb = colorsys.hsv_to_rgb(hue,0.85,0.5+0.5*z)
            al=int(80+175*z)
            x,y = self.px[k],self.py[k]
            draw.ellipse([x-rsz,y-rsz,x+rsz,y+rsz], fill=(int(cr*255),int(cg*255),int(cb*255),al))
        return canvas

class RadialBarsScene(Scene):
    name = "圆环"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.bands=96
        self.rot_speed_base=0.004
        if self.params is not None:
            self.bands = self.params.get("bands",self.bands)
            self.rot_speed_base = self.params.get("rot_speed_base",self.rot_speed_base)
        self.cx=W//2
        self.cy=H//2
        self.r0=int(min(W,H)*0.12)
        self.max_len=int(min(W,H)*0.35)
        self.rot=0.0
    def render(self,i,intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0,len(spec)-1,self.bands).astype(int)
        vals = spec[idx]
        bass = float(self.ctx["bass"][i])
        self.rot += self.rot_speed_base + intensity*0.01
        canvas = Image.new("RGB",(self.W,self.H),(4,5,12))
        draw = ImageDraw.Draw(canvas,"RGBA")
        n=self.bands
        for k in range(n):
            ang = self.rot + k/n*2*math.pi
            v = float(vals[k])*(0.6+0.6*intensity)
            r1 = self.r0 + self.max_len*v
            x0=self.cx+math.cos(ang)*self.r0
            y0=self.cy+math.sin(ang)*self.r0
            x1=self.cx+math.cos(ang)*r1
            y1=self.cy+math.sin(ang)*r1
            hue=(k/n+self.rot*0.2)%1.0
            r,g,b = colorsys.hsv_to_rgb(hue,0.9,0.35+0.65*v)
            draw.line([x0,y0,x1,y1], fill=(int(r*255),int(g*255),int(b*255),220), width=max(1,int(1+v*4)))
        cr = self.r0*(0.6+0.6*bass)
        draw.ellipse([self.cx-cr,self.cy-cr,self.cx+cr,self.cy+cr], outline=(255,255,255,int(60+150*bass)), width=2)
        return canvas

class BarsScene(Scene):
    name = "竖条"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.bands=72
        if self.params is not None:
            self.bands = self.params.get("bands",self.bands)
    def render(self,i,intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0,len(spec)-1,self.bands).astype(int)
        vals = np.clip(spec[idx]*(0.7+0.6*intensity),0,1)
        canvas = Image.new("RGB",(self.W,self.H),(5,6,12))
        draw = ImageDraw.Draw(canvas,"RGBA")
        gap=2
        bw = max(2,(self.W-gap*(self.bands-1))//self.bands)
        base_y=int(self.H*0.78)
        max_h=int(self.H*0.55)
        for k in range(self.bands):
            v=float(vals[k])
            h=int(max_h*v)
            x = k*(bw+gap)
            hue = (k/self.bands*0.85 + 0.02*i/60) %1.0
            r,g,b = colorsys.hsv_to_rgb(hue,0.85,0.4+0.6*v)
            col=(int(r*255),int(g*255),int(b*255))
            draw.rectangle([x,base_y-h,x+bw,base_y], fill=col)
            for j in range(h):
                a=int(120*(1-j/max(h,1)))
                if a<=0:continue
                draw.rectangle([x,base_y+j,x+bw,base_y+j+1], fill=col+(a,))
        return canvas

SCENE_CLASSES = [WaterfallScene,VortexScene,ParticlesScene,RadialBarsScene,BarsScene]
SCENE_CLASS_MAP = {
    "WaterfallScene":WaterfallScene,
    "VortexScene":VortexScene,
    "ParticlesScene":ParticlesScene,
    "RadialBarsScene":RadialBarsScene,
    "BarsScene":BarsScene
}

def build_timeline(ctx, min_dur, max_dur, scene_cls_list=None, seed=0):
    rng = np.random.default_rng(seed)
    fps = ctx["fps"]
    n = ctx["n_frames"]
    nov = ctx["novelty"]
    min_f = int(min_dur * fps)
    max_f = int(max_dur * fps)
    timeline = []
    i = 0
    last_scene = -1
    if scene_cls_list is None:
        scene_cls_list = SCENE_CLASSES
    scene_idx = int(rng.integers(0, len(scene_cls_list)))
    while i < n:
        end_min = min(n, i + min_f)
        end_max = min(n, i + max_f)
        if end_max <= end_min:
            timeline.append((i, scene_idx))
            break
        window = nov[end_min:end_max]
        if len(window)>0:
            peak = int(np.argmax(window)) + end_min
        else:
            peak = end_max
        timeline.append((i, scene_idx))
        i = peak
        choices = [k for k in range(len(scene_cls_list)) if k != last_scene]
        scene_idx = int(rng.choice(choices))
        last_scene = scene_idx
    return timeline

def render(ctx, timeline, cfg, scene_list, out_path):
    W = cfg["width"]//2*2
    H = cfg["height"]//2*2
    fps = ctx["fps"]
    n = ctx["n_frames"]
    fade_frames = int(cfg["fade"] * fps)
    if inspect.isclass(scene_list[0]):
        scenes = [cls(W,H,ctx) for cls in scene_list]
    else:
        scenes = scene_list
    names = [s.name for s in scenes]
    print("  场景: "+", ".join(names), file=sys.stderr)
    scene_of = np.zeros(n,dtype=np.int32)
    for k,(start,sidx) in enumerate(timeline):
        end = timeline[k+1][0] if (k+1 < len(timeline)) else n
        scene_of[start:end] = sidx
    intensity = np.clip(0.5*ctx["rms"] + 0.5*ctx["novelty"],0,1)
    intensity = smooth1d(intensity,0.2)
    cmd = [
        FFMPEG, "-y", "-loglevel","error",
        "-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(fps),
        "-i","-",
        "-i",cfg["input"],
        "-map","0:v:0","-map","1:a:0",
        "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart", out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    report = max(1,n//100)
    prev_scene = -1
    prev_frame = None
    fade_left = 0
    try:
        for i in range(n):
            sidx = int(scene_of[i])
            cur = scenes[sidx].render(i, float(intensity[i]))
            if sidx != prev_scene:
                fade_left = fade_frames
                prev_scene = sidx
            if fade_left>0 and prev_frame is not None:
                a = 1.0-fade_left/max(fade_frames,1)
                cur = Image.blend(prev_frame,cur,a)
                fade_left-=1
            if cfg["show_label"]:
                draw = ImageDraw.Draw(cur,"RGBA")
                scene_name = scenes[sidx].name
                draw.text((20,14),f"WMP  {scene_name}", fill=TEXT_COL)
                if cfg["beats"] and cfg["bpm"]>0:
                    cur_sec = i/float(fps)
                    sec_per_bar = 60.0/cfg["bpm"] * cfg["beats_per_bar"]
                    last_bar = int(cur_sec/sec_per_bar)
                    for k in range(max(0,last_bar-3), last_bar+1):
                        t = k*sec_per_bar
                        if t>cur_sec:continue
                        fade = 1.0-(cur_sec-t)/(sec_per_bar*4)
                        fade = max(0.0,min(1.0,fade))
                        x = int(W*0.95-W*0.9*fade)
                        draw.line([(x,40),(x,H-20)], fill=BEAT_COL+(int(200*fade),), width=2)
                    bbox = draw.textbbox((0,0),f"{int(cfg['bpm'])} BPM")
                    draw.text((W-20-(bbox[2]-bbox[0]),14), f"{int(cfg['bpm'])} BPM", fill=BPM_COL)
            proc.stdin.write(cur.tobytes("raw","RGB"))
            prev_frame = cur
            if i%report ==0:
                print(f"\r  渲染 {i*100//n:3d}%", end="", file=sys.stderr, flush=True)
    finally:
        try: proc.stdin.close()
        except Exception:pass
        proc.wait()
    print("\r渲染完成 ✔", file=sys.stderr)

# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="WMP可视化，支持原始模式 / --style旧配置 / --generic-style通用解析器")
    ap.add_argument("input", nargs="?", default=None)
    ap.add_argument("-o","--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-dur", type=float, default=6.0)
    ap.add_argument("--max-dur", type=float, default=14.0)
    ap.add_argument("--fade", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-vocals", dest="auto_vocals", action="store_true")
    ap.add_argument("--cache-dir", default=".vocals_cache")
    ap.add_argument("--beats", dest="beats", action="store_true")
    ap.add_argument("--bpm", type=float, default=0.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--no-label", dest="show_label", action="store_false")
    ap.add_argument("--style", type=str, default=None, help="旧模式wmpvis，给原有Scene做参数覆盖")
    ap.add_argument("--export-style", type=str, default=None, help="导出旧版模板")
    ap.add_argument("--generic-style", type=str, default=None, help="【通用解析器模式】全部动画由json配置驱动")
    ap.add_argument("--export-generic-style", type=str, default=None, help="导出通用解析器模板")
    ap.set_defaults(show_label=True)
    a = ap.parse_args()

    if a.export_style is not None:
        WmpVisStyle.export_template(a.export_style)
        return
    if a.export_generic_style is not None:
        GenericVisParser.export_generic_template(a.export_generic_style)
        return
    if a.input is None:
        raise SystemExit("必须提供输入音频文件")
    if not FFMPEG:
        raise SystemExit("未找到ffmpeg")

    print("[1/4] 准备音频", file=sys.stderr)
    src_path = a.input
    if a.auto_vocals:
        vp = find_or_create_vocals(a.input, a.cache_dir)
        if vp:
            src_path = vp
            print(f"      使用人声轨: {vp}", file=sys.stderr)
        else:
            print("      分离失败，退回原曲", file=sys.stderr)
    data, sr = load_audio(src_path)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s", file=sys.stderr)

    print("[2/4] 分析频谱", file=sys.stderr)
    ctx = analyze(data.mean(axis=1) if data.shape[1]>1 else data[:,0], sr, a.fps)
    mono_for_bpm = data.mean(axis=1) if data.shape[1]>1 else data[:,0]
    bpm = 0.0
    if a.beats:
        if a.bpm>0:
            bpm = a.bpm
        else:
            bpm = detect_bpm(mono_for_bpm, sr, a.fps)
    print(f"      BPM = {bpm:.1f}", file=sys.stderr)

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_wmp.mp4"

    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height, fade=a.fade,
               beats=a.beats, bpm=bpm, beats_per_bar=a.beats_per_bar, show_label=a.show_label)

    # ========== 分支1：通用解析器模式 generic-style ==========
    if a.generic_style is not None:
        print(f"✅ 使用通用解析器，配置文件:{a.generic_style}", file=sys.stderr)
        with open(a.generic_style,"r",encoding="utf-8") as f:
            g_cfg = json.load(f)
        parser = GenericVisParser(a.width, a.height, g_cfg)
        W = a.width//2*2
        H = a.height//2*2
        n_frames = ctx["n_frames"]
        intensity_arr = np.clip(0.5*ctx["rms"] + 0.5*ctx["novelty"],0,1)
        cmd = [
            FFMPEG, "-y", "-loglevel","error",
            "-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(a.fps),
            "-i","-",
            "-i",a.input,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart", out_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        report = max(1, n_frames//100)
        print("[4/4] 通用解析器渲染", file=sys.stderr)
        try:
            for i in range(n_frames):
                fa = {
                    "spec":ctx["spec"][i],
                    "rms":float(ctx["rms"][i]),
                    "bass":float(ctx["bass"][i]),
                    "treble":float(ctx["treble"][i]),
                    "novelty":float(ctx["novelty"][i]),
                    "intensity":float(intensity_arr[i])
                }
                img = parser.render_frame(i, fa)
                proc.stdin.write(img.tobytes("raw","RGB"))
                if i % report ==0:
                    print(f"\r  渲染 {i*100//n_frames:3d}%", end="", file=sys.stderr, flush=True)
        finally:
            try:proc.stdin.close()
            except Exception:pass
            proc.wait()
        print("\r渲染完成 ✔", file=sys.stderr)
        print(f"输出: {out_path}", file=sys.stderr)
        return

    # ========== 分支2：旧模式 --style / 原版无配置 ==========
    use_wmpvis = a.style is not None
    scene_instances = None
    if use_wmpvis:
        print(f"✅ 加载旧版wmpvis配置:{a.style}", file=sys.stderr)
        style_data = WmpVisStyle.load_style(a.style)
        scene_instances = []
        for preset in style_data["presets"]:
            if not preset.get("enable",True): continue
            cls = SCENE_CLASS_MAP[preset["class_name"]]
            inst = cls(a.width, a.height, ctx, params=preset["params"])
            scene_instances.append(inst)
        if len(scene_instances)==0:
            raise SystemExit("wmpvis没有启用任何场景！")
        timeline = build_timeline(ctx, a.min_dur, a.max_dur, scene_cls_list=scene_instances, seed=a.seed)
    else:
        print("ℹ 未指定配置，运行原始硬编码场景", file=sys.stderr)
        timeline = build_timeline(ctx, a.min_dur, a.max_dur, seed=a.seed)

    for start,sidx in timeline:
        t = start / a.fps
        if use_wmpvis:
            name = scene_instances[sidx].name
        else:
            name = SCENE_CLASSES[sidx].name
        print(f"      {t:6.1f}s → {name}", file=sys.stderr)

    print("[4/4] 渲染", file=sys.stderr)
    if use_wmpvis:
        render(ctx, timeline, cfg, scene_instances, out_path)
    else:
        render(ctx, timeline, cfg, SCENE_CLASSES, out_path)
    print(f"输出: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPVisualizer.py —— 随机场景可视化（跟着音乐切换）

特点:
  - 5 个场景：瀑布 / 漩涡 / 粒子 / 圆环 / 竖条
  - 场景切换由 novelty（能量变化率）驱动，在音乐转折点切，不是纯随机
  - 每个场景内部跟着 bass / treble / rms 抖动
  - 切换时淡入淡出，不硬切
  - 可选 Demucs 分离人声（--auto-vocals）
  - 可选 BPM 检测（--beats）

用法:
    python3 WMPVisualizer.py 歌曲.mp3 -o out.mp4
    python3 WMPVisualizer.py 歌曲.mp3 --min-dur 6 --max-dur 14
    python3 WMPVisualizer.py 歌曲.mp3 --auto-vocals --beats
"""

import argparse
import colorsys
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (5, 6, 12)
TEXT_COL = (200, 210, 220)
BEAT_COL = (40, 220, 120)
BPM_COL = (255, 220, 90)


# ==========================================================================
# 音频
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


# ==========================================================================
# Demucs 分离（可选）
# ==========================================================================
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
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=7200)
    except subprocess.TimeoutExpired:
        print("      分离超时", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    if p.returncode != 0:
        print(f"      分离失败: {p.stderr.strip()[-300:]}",
              file=sys.stderr)
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
            shutil.move(inst_src,
                        os.path.join(cache_dir,
                                     base + "_instrumental.wav"))
        except Exception:
            pass
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"      人声已缓存: {cache_path}", file=sys.stderr)
    return cache_path


# ==========================================================================
# 分析
# ==========================================================================
def smooth1d(x, k):
    y = np.empty_like(x)
    cur = 0.0
    for i in range(len(x)):
        cur += k * (x[i] - cur)
        y[i] = cur
    return y


def analyze(mono, sr, fps, fft_size=4096, bands=128,
            fmin=30.0, fmax=16000.0):
    n = len(mono)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    edges = np.geomspace(max(fmin, 20.0),
                         min(fmax, sr / 2 - 1), bands + 1)
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
            print(f"\r  分析 {i * 100 // n_frames:3d}%",
                  end="", file=sys.stderr, flush=True)
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

    return dict(spec=norm, rms=rms_norm, bass=bass, treble=treble,
                novelty=novelty, n_frames=n_frames, bands=bands,
                sr=sr, fps=fps)


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
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag_min = max(1, int(fps * 60.0 / bpm_max))
    lag_max = min(len(ac) - 1, int(fps * 60.0 / bpm_min))
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
# 场景基类
# ==========================================================================
class Scene:
    name = "scene"

    def __init__(self, W, H, ctx):
        self.W = W
        self.H = H
        self.ctx = ctx

    def render(self, i, intensity):
        raise NotImplementedError


# --------------------------------------------------------------------------
# 场景 1：瀑布
# --------------------------------------------------------------------------
class WaterfallScene(Scene):
    name = "瀑布"

    def __init__(self, W, H, ctx):
        super().__init__(W, H, ctx)
        self.bands = min(160, ctx["bands"])
        self.rows = 110
        self.buf = np.zeros((self.rows, self.bands, 3), dtype=np.uint8)
        self.lut = self._make_lut()

    def _make_lut(self, n=256):
        stops = np.array([
            [0.00,   0,   0,  50], [0.15,   0,   0, 160],
            [0.30,   0,  70, 240], [0.45,   0, 190, 220],
            [0.60,  70, 240, 130], [0.75, 220, 230,  60],
            [0.90, 255, 130,  30], [1.00, 255,  50,  40],
        ], dtype=np.float64)
        t = np.linspace(0, 1, n)
        lut = np.empty((n, 3), dtype=np.uint8)
        for k in range(3):
            lut[:, k] = np.interp(t, stops[:, 0], stops[:, 1 + k])
        return lut

    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0, len(spec) - 1, self.bands).astype(int)
        vals = np.clip(spec[idx] * (0.7 + 0.6 * intensity), 0, 1)
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = self.lut[(vals * 255).astype(np.uint8)]
        img = Image.fromarray(self.buf, "RGB").resize(
            (self.W, self.H), Image.BILINEAR)
        top = img.crop((0, 0, self.W, self.H // 2))
        arr = np.asarray(top, dtype=np.float32)[::-1]
        grad = np.linspace(1.0, 0.15,
                           self.H - self.H // 2)[:, None, None]
        bot = Image.fromarray(
            np.clip(arr * grad, 0, 255).astype(np.uint8))
        canvas = Image.new("RGB", (self.W, self.H))
        canvas.paste(top, (0, 0))
        canvas.paste(bot, (0, self.H // 2))
        return canvas


# --------------------------------------------------------------------------
# 场景 2：漩涡
# --------------------------------------------------------------------------
class VortexScene(Scene):
    name = "漩涡"

    def __init__(self, W, H, ctx):
        super().__init__(W, H, ctx)
        self.bands = 180
        self.cx = W // 2
        self.cy = H // 2
        self.max_r = int(min(W, H) * 0.45)
        self.rot = 0.0
        self.buf = None

    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0, len(spec) - 1, self.bands).astype(int)
        vals = spec[idx]
        bass = self.ctx["bass"][i]
        treble = self.ctx["treble"][i]

        self.rot += 0.008 + (treble * 0.06 + intensity * 0.03)
        if self.buf is None:
            self.buf = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.buf = (self.buf * (0.90 - 0.05 * intensity)).astype(np.uint8)

        canvas = Image.fromarray(self.buf, "RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        n = self.bands
        for k in range(n):
            ang = self.rot + k / n * 2 * math.pi
            v = float(vals[k]) * (0.6 + 0.6 * intensity)
            r0 = self.max_r * 0.20
            r1 = self.max_r * (0.28 + 0.72 * v)
            x0 = self.cx + math.cos(ang) * r0
            y0 = self.cy + math.sin(ang) * r0
            x1 = self.cx + math.cos(ang) * r1
            y1 = self.cy + math.sin(ang) * r1
            hue = (k / n + self.rot * 0.1) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.3 + 0.7 * v)
            draw.line([x0, y0, x1, y1],
                      fill=(int(r * 255), int(g * 255), int(b * 255), 200),
                      width=max(1, int(1 + v * 3)))

        ring_r = self.max_r * (0.15 + 0.5 * bass)
        draw.ellipse([self.cx - ring_r, self.cy - ring_r,
                      self.cx + ring_r, self.cy + ring_r],
                     outline=(255, 255, 255, int(80 + 150 * bass)),
                     width=2)

        self.buf = np.asarray(canvas, dtype=np.uint8).copy()
        return canvas


# --------------------------------------------------------------------------
# 场景 3：粒子
# --------------------------------------------------------------------------
class ParticlesScene(Scene):
    name = "粒子"

    def __init__(self, W, H, ctx):
        super().__init__(W, H, ctx)
        self.n = 380
        rng = np.random.default_rng(42)
        self.px = rng.uniform(0, W, self.n).astype(np.float32)
        self.py = rng.uniform(0, H, self.n).astype(np.float32)
        self.pz = rng.uniform(0.1, 1.0, self.n).astype(np.float32)
        self.hue = rng.uniform(0, 1, self.n).astype(np.float32)
        self.cx = W / 2
        self.cy = H / 2

    def render(self, i, intensity):
        rms = float(self.ctx["rms"][i])
        bass = float(self.ctx["bass"][i])
        treble = float(self.ctx["treble"][i])

        speed = 0.4 + rms * 5.0 + bass * 5.0 + intensity * 3.0
        vx = (self.cx - self.px)
        vy = (self.cy - self.py)
        d = np.sqrt(vx ** 2 + vy ** 2) + 1e-3
        self.px += vx / d * speed
        self.py += vy / d * speed

        out = (d < 6) | (self.px < -20) | (self.px > self.W + 20) | \
              (self.py < -20) | (self.py > self.H + 20)
        if out.any():
            rng = np.random.default_rng(i)
            k = int(out.sum())
            ang = rng.uniform(0, 2 * math.pi, k)
            r = max(self.W, self.H) * 0.5 + rng.uniform(0, 200, k)
            self.px[out] = self.cx + np.cos(ang) * r
            self.py[out] = self.cy + np.sin(ang) * r
            self.hue[out] = rng.uniform(0, 1, k)

        canvas = Image.new("RGB", (self.W, self.H), (4, 4, 10))
        draw = ImageDraw.Draw(canvas, "RGBA")
        glow = int(50 + 150 * bass)
        gr = int(70 + 220 * bass)
        draw.ellipse([self.cx - gr, self.cy - gr,
                      self.cx + gr, self.cy + gr],
                     fill=(80, 140, 255, glow))

        for k in range(self.n):
            z = self.pz[k]
            r = 1.0 + z * 2.0 + treble * 2.0
            hue = (self.hue[k] + i * 0.002) % 1.0
            cr, cg, cb = colorsys.hsv_to_rgb(hue, 0.85, 0.5 + 0.5 * z)
            alpha = int(80 + 175 * z)
            x, y = self.px[k], self.py[k]
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(int(cr * 255), int(cg * 255),
                               int(cb * 255), alpha))
        return canvas


# --------------------------------------------------------------------------
# 场景 4：圆环
# --------------------------------------------------------------------------
class RadialBarsScene(Scene):
    name = "圆环"

    def __init__(self, W, H, ctx):
        super().__init__(W, H, ctx)
        self.bands = 96
        self.cx = W // 2
        self.cy = H // 2
        self.r0 = int(min(W, H) * 0.12)
        self.max_len = int(min(W, H) * 0.35)
        self.rot = 0.0

    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0, len(spec) - 1, self.bands).astype(int)
        vals = spec[idx]
        bass = float(self.ctx["bass"][i])

        self.rot += 0.004 + intensity * 0.01

        canvas = Image.new("RGB", (self.W, self.H), (4, 5, 12))
        draw = ImageDraw.Draw(canvas, "RGBA")
        n = self.bands
        for k in range(n):
            ang = self.rot + k / n * 2 * math.pi
            v = float(vals[k]) * (0.6 + 0.6 * intensity)
            r1 = self.r0 + self.max_len * v
            x0 = self.cx + math.cos(ang) * self.r0
            y0 = self.cy + math.sin(ang) * self.r0
            x1 = self.cx + math.cos(ang) * r1
            y1 = self.cy + math.sin(ang) * r1
            hue = (k / n + self.rot * 0.2) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.35 + 0.65 * v)
            draw.line([x0, y0, x1, y1],
                      fill=(int(r * 255), int(g * 255), int(b * 255), 220),
                      width=max(1, int(1 + v * 4)))

        cr = self.r0 * (0.6 + 0.6 * bass)
        draw.ellipse([self.cx - cr, self.cy - cr,
                      self.cx + cr, self.cy + cr],
                     outline=(255, 255, 255, int(60 + 150 * bass)),
                     width=2)
        return canvas


# --------------------------------------------------------------------------
# 场景 5：竖条
# --------------------------------------------------------------------------
class BarsScene(Scene):
    name = "竖条"

    def __init__(self, W, H, ctx):
        super().__init__(W, H, ctx)
        self.bands = 72

    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        idx = np.linspace(0, len(spec) - 1, self.bands).astype(int)
        vals = np.clip(spec[idx] * (0.7 + 0.6 * intensity), 0, 1)

        canvas = Image.new("RGB", (self.W, self.H), (5, 6, 12))
        draw = ImageDraw.Draw(canvas, "RGBA")
        gap = 2
        bw = max(2, (self.W - gap * (self.bands - 1)) // self.bands)
        base_y = int(self.H * 0.78)
        max_h = int(self.H * 0.55)

        for k in range(self.bands):
            v = float(vals[k])
            h = int(max_h * v)
            x = k * (bw + gap)
            hue = (k / self.bands * 0.85 + 0.02 * i / 60) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.4 + 0.6 * v)
            col = (int(r * 255), int(g * 255), int(b * 255))
            draw.rectangle([x, base_y - h, x + bw, base_y], fill=col)
            for j in range(h):
                a = int(120 * (1 - j / max(h, 1)))
                if a <= 0:
                    continue
                draw.rectangle([x, base_y + j, x + bw, base_y + j + 1],
                               fill=col + (a,))
        return canvas


SCENE_CLASSES = [WaterfallScene, VortexScene, ParticlesScene,
                 RadialBarsScene, BarsScene]


# ==========================================================================
# 调度器
# ==========================================================================
def build_timeline(ctx, min_dur, max_dur, seed=0):
    rng = np.random.default_rng(seed)
    fps = ctx["fps"]
    n = ctx["n_frames"]
    nov = ctx["novelty"]

    min_f = int(min_dur * fps)
    max_f = int(max_dur * fps)

    timeline = []
    i = 0
    last_scene = -1
    scene_idx = int(rng.integers(0, len(SCENE_CLASSES)))

    while i < n:
        end_min = min(n, i + min_f)
        end_max = min(n, i + max_f)
        if end_max <= end_min:
            timeline.append((i, scene_idx))
            break

        window = nov[end_min:end_max]
        if len(window) > 0:
            peak = int(np.argmax(window)) + end_min
        else:
            peak = end_max

        timeline.append((i, scene_idx))
        i = peak

        choices = [k for k in range(len(SCENE_CLASSES))
                   if k != last_scene]
        scene_idx = int(rng.choice(choices))
        last_scene = scene_idx

    return timeline


# ==========================================================================
# 渲染
# ==========================================================================
def render(ctx, timeline, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = ctx["fps"]
    n = ctx["n_frames"]
    fade_frames = int(cfg["fade"] * fps)

    scenes = [cls(W, H, ctx) for cls in SCENE_CLASSES]
    names = [s.name for s in scenes]
    print("  场景: " + ", ".join(names), file=sys.stderr)

    scene_of = np.zeros(n, dtype=np.int32)
    for k, (start, sidx) in enumerate(timeline):
        end = timeline[k + 1][0] if k + 1 < len(timeline) else n
        scene_of[start:end] = sidx

    intensity = np.clip(0.5 * ctx["rms"] + 0.5 * ctx["novelty"], 0, 1)
    intensity = smooth1d(intensity, 0.2)

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-i", cfg["input"],
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    report = max(1, n // 100)

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

            if fade_left > 0 and prev_frame is not None:
                a = 1.0 - fade_left / max(fade_frames, 1)
                cur = Image.blend(prev_frame, cur, a)
                fade_left -= 1

            # 顶部标签 + 小节线
            if cfg["show_label"]:
                draw = ImageDraw.Draw(cur, "RGBA")
                scene_name = scenes[sidx].name
                draw.text((20, 14), f"WMP  {scene_name}", fill=TEXT_COL)

                if cfg["beats"] and cfg["bpm"] > 0:
                    cur_sec = i / float(fps)
                    sec_per_bar = 60.0 / cfg["bpm"] * cfg["beats_per_bar"]
                    last_bar = int(cur_sec / sec_per_bar)
                    for k in range(max(0, last_bar - 3), last_bar + 1):
                        t = k * sec_per_bar
                        if t > cur_sec:
                            continue
                        fade = 1.0 - (cur_sec - t) / (sec_per_bar * 4)
                        fade = max(0.0, min(1.0, fade))
                        x = int(W * 0.95 - W * 0.9 * fade)
                        draw.line([(x, 40), (x, H - 20)],
                                  fill=BEAT_COL + (int(200 * fade),),
                                  width=2)
                    bbox = draw.textbbox((0, 0), f"{int(cfg['bpm'])} BPM")
                    draw.text((W - 20 - (bbox[2] - bbox[0]), 14),
                              f"{int(cfg['bpm'])} BPM", fill=BPM_COL)

            proc.stdin.write(cur.tobytes("raw", "RGB"))
            prev_frame = cur

            if i % report == 0:
                print(f"\r  渲染 {i * 100 // n:3d}%",
                      end="", file=sys.stderr, flush=True)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    print("\r渲染完成 ✔", file=sys.stderr)


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="WMP 随机场景可视化（跟着音乐切换）")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--min-dur", type=float, default=6.0,
                    help="每个场景最短持续时间（秒）")
    ap.add_argument("--max-dur", type=float, default=14.0,
                    help="每个场景最长持续时间（秒）")
    ap.add_argument("--fade", type=float, default=0.6,
                    help="切换淡入淡出时长（秒）")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--auto-vocals", dest="auto_vocals",
                    action="store_true",
                    help="用 Demucs 分离人声，分析人声轨")
    ap.add_argument("--cache-dir", default=".vocals_cache")

    ap.add_argument("--beats", dest="beats", action="store_true",
                    help="叠加 BPM 小节线")
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="手动 BPM，0=自动")
    ap.add_argument("--beats-per-bar", type=int, default=4)

    ap.add_argument("--no-label", dest="show_label",
                    action="store_false", help="不显示左上角标签")
    ap.set_defaults(show_label=True)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_wmp.mp4"

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
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s",
          file=sys.stderr)

    print("[2/4] 分析频谱", file=sys.stderr)
    ctx = analyze(data.mean(axis=1) if data.shape[1] > 1 else data[:, 0],
                  sr, a.fps)

    mono_for_bpm = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if a.bpm > 0:
        bpm = a.bpm
    else:
        bpm = detect_bpm(mono_for_bpm, sr, a.fps) if a.beats else 0.0
    print(f"      BPM = {bpm:.1f}", file=sys.stderr)

    print("[3/4] 规划场景时间线", file=sys.stderr)
    timeline = build_timeline(ctx, a.min_dur, a.max_dur, a.seed)
    for start, sidx in timeline:
        t = start / a.fps
        print(f"      {t:6.1f}s → {SCENE_CLASSES[sidx].name}",
              file=sys.stderr)

    cfg = dict(input=a.input, fps=a.fps,
               width=a.width, height=a.height,
               fade=a.fade,
               beats=a.beats, bpm=bpm,
               beats_per_bar=a.beats_per_bar,
               show_label=a.show_label)

    print("[4/4] 渲染", file=sys.stderr)
    render(ctx, timeline, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

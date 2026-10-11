#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpeakerMeter.py —— 低音炮 + 高音喇叭模拟仪

原理:
  低频段（20~200Hz）→ 低音炮：大圆、慢波纹、深蓝色
  高频段（4k~16k） → 高音喇叭：小号角、快波纹、橙黄色
  每个喇叭有自己的 attack/release，波纹用环形缓存做残影

用法:
    python3 SpeakerMeter.py 歌曲.mp3 -o out.mp4
    python3 SpeakerMeter.py 歌曲.mp3 --sub-max 120 --tweet-min 6000
"""

import argparse
import colorsys
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (10, 11, 15)
TEXT_COL = (200, 210, 220)
PANEL_BG = (18, 20, 26)
PANEL_LINE = (55, 60, 72)

# 低音炮：深蓝
SUB_DARK = (20, 30, 60)
SUB_MID = (40, 90, 200)
SUB_BRIGHT = (110, 190, 255)

# 高音：橙黄
TW_DARK = (60, 35, 10)
TW_MID = (220, 130, 40)
TW_BRIGHT = (255, 220, 120)


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
        raise SystemExit("需要 ffmpeg 和 ffprobe，请先 pkg install ffmpeg")
    return _decode_ffmpeg(path)


# ==========================================================================
# FFT 分带
# ==========================================================================
def spectrum_bands(data, sr, fps, fft_size=4096, bands=96,
                   fmin=20.0, fmax=16000.0):
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    n = len(mono)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    edges = np.geomspace(max(fmin, 15.0),
                         min(fmax, sr / 2 - 1), bands + 1)
    band_idx = []
    for k in range(bands):
        lo, hi = edges[k], edges[k + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2))])
        band_idx.append(idx)

    out = np.zeros((n_frames, bands), dtype=np.float32)
    pad = fft_size // 2
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
            out[i, k] = s[idx].max()
        if i % 300 == 0:
            print(f"\r  频谱 {i * 100 // n_frames:3d}%",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # 频率坐标表（每个 band 的中心频率）
    band_freqs = np.array([freqs[idx].mean() for idx in band_idx])
    return out, band_freqs


# ==========================================================================
# 分频 + 平滑
# ==========================================================================
def extract_bands(spec_db, band_freqs, fps,
                  sub_max=200.0, tweet_min=4000.0,
                  sub_attack=0.6, sub_release=0.10,
                  tw_attack=0.75, tw_release=0.22):
    sub_idx = np.where(band_freqs <= sub_max)[0]
    tw_idx = np.where(band_freqs >= tweet_min)[0]
    if len(sub_idx) == 0:
        sub_idx = np.array([0])
    if len(tw_idx) == 0:
        tw_idx = np.array([len(band_freqs) - 1])

    sub = spec_db[:, sub_idx].mean(axis=1)
    tw = spec_db[:, tw_idx].mean(axis=1)

    def norm(x, db_min, db_max):
        return np.clip((x - db_min) / (db_max - db_min), 0.0, 1.0)

    sub = norm(sub, -50.0, -5.0)
    tw = norm(tw, -55.0, -10.0)

    def smooth(x, a_up, a_dn):
        out = np.empty_like(x)
        cur = 0.0
        for i in range(len(x)):
            k = a_up if x[i] > cur else a_dn
            cur += k * (x[i] - cur)
            out[i] = cur
        return out

    return smooth(sub, sub_attack, sub_release), \
           smooth(tw, tw_attack, tw_release)


# ==========================================================================
# 环形缓存（波纹残影）
# ==========================================================================
class RingBuffer:
    def __init__(self, max_age=14):
        self.max_age = max_age
        self.items = []  # [[age, strength], ...]

    def push(self, strength):
        self.items.append([0, strength])

    def tick(self):
        for it in self.items:
            it[0] += 1
        self.items = [it for it in self.items if it[0] <= self.max_age]


# ==========================================================================
# 绘制：低音炮
# ==========================================================================
def draw_subwoofer(canvas, cx, cy, r, energy, rings):
    draw = ImageDraw.Draw(canvas, "RGBA")
    e = float(energy)

    # 波纹
    for age, strength in rings:
        rr = r + 10 + age * (10 + 22 * strength)
        alpha = max(0, int((1.0 - age / 14.0) * 130 * strength))
        if alpha <= 0:
            continue
        col = SUB_MID if age < 7 else SUB_DARK
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     outline=col + (alpha,), width=2)

    # 外框
    draw.ellipse([cx - r - 8, cy - r - 8,
                  cx + r + 8, cy + r + 8],
                 outline=PANEL_LINE, width=2)

    # 盆体
    bulge = 1.0 + 0.12 * e
    r_in = r * bulge
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in],
                 fill=SUB_DARK)
    # 盆面（亮度随能量）
    face = int(40 + 90 * e)
    draw.ellipse([cx - r_in * 0.72, cy - r_in * 0.72,
                  cx + r_in * 0.72, cy + r_in * 0.72],
                 fill=(face // 3, face, min(255, face + 40)))
    # 防尘帽
    draw.ellipse([cx - r_in * 0.30, cy - r_in * 0.30,
                  cx + r_in * 0.30, cy + r_in * 0.30],
                 fill=(18, 22, 30))
    # 能量溢出光
    if e > 0.55:
        glow = int((e - 0.55) / 0.45 * 180)
        draw.ellipse([cx - r_in - 4, cy - r_in - 4,
                      cx + r_in + 4, cy + r_in + 4],
                     outline=SUB_BRIGHT + (glow,), width=3)


# ==========================================================================
# 绘制：高音喇叭
# ==========================================================================
def draw_tweeter(canvas, cx, cy, r, energy, rings):
    draw = ImageDraw.Draw(canvas, "RGBA")
    e = float(energy)

    # 波纹
    for age, strength in rings:
        rr = r + 6 + age * (7 + 15 * strength)
        alpha = max(0, int((1.0 - age / 10.0) * 150 * strength))
        if alpha <= 0:
            continue
        col = TW_MID if age < 5 else TW_DARK
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     outline=col + (alpha,), width=2)

    # 号角外框（梯形）
    hw = r * 1.7
    draw.polygon([
        (cx - hw * 0.45, cy - r * 0.55),
        (cx + hw * 0.45, cy - r * 0.55),
        (cx + hw * 0.90, cy + r * 0.90),
        (cx - hw * 0.90, cy + r * 0.90),
    ], outline=PANEL_LINE, width=2)

    # 振膜
    r_in = r * (1.0 + 0.15 * e)
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in],
                 fill=TW_DARK)
    face = int(120 + 100 * e)
    draw.ellipse([cx - r_in * 0.62, cy - r_in * 0.62,
                  cx + r_in * 0.62, cy + r_in * 0.62],
                 fill=(face, max(0, face - 40), max(0, face - 90)))
    if e > 0.5:
        glow = int((e - 0.5) / 0.5 * 190)
        draw.ellipse([cx - r_in - 4, cy - r_in - 4,
                      cx + r_in + 4, cy + r_in + 4],
                     outline=TW_BRIGHT + (glow,), width=3)


# ==========================================================================
# 渲染
# ==========================================================================
def render(sub, tw, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n = len(sub)

    sub_cx = W // 4
    tw_cx = W * 3 // 4
    cy = H // 2
    sub_r = int(min(W, H) * 0.20)
    tw_r = int(min(W, H) * 0.13)

    sub_rings = RingBuffer(max_age=14)
    tw_rings = RingBuffer(max_age=10)

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

    try:
        for i in range(n):
            canvas = Image.new("RGB", (W, H), BG)
            sub_e = float(sub[i])
            tw_e = float(tw[i])

            if sub_e > 0.30:
                sub_rings.push(sub_e)
            if tw_e > 0.40:
                tw_rings.push(tw_e)

            draw_subwoofer(canvas, sub_cx, cy, sub_r, sub_e, sub_rings.items)
            draw_tweeter(canvas, tw_cx, cy, tw_r, tw_e, tw_rings.items)

            # 标签
            draw = ImageDraw.Draw(canvas, "RGBA")
            draw.text((sub_cx - 40, cy + sub_r + 30), "SUBWOOFER",
                      fill=SUB_BRIGHT)
            draw.text((tw_cx - 30, cy + tw_r + 30), "TWEETER",
                      fill=TW_BRIGHT)

            sub_rings.tick()
            tw_rings.tick()

            proc.stdin.write(canvas.tobytes("raw", "RGB"))
            if i % report == 0:
                print(f"\r渲染中… {i * 100 // n:3d}%",
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
    ap = argparse.ArgumentParser(description="低音炮 + 高音喇叭模拟仪")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--sub-max", type=float, default=200.0,
                    help="低音分频点 Hz")
    ap.add_argument("--tweet-min", type=float, default=4000.0,
                    help="高音分频点 Hz")
    ap.add_argument("--fft-size", type=int, default=4096)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_speaker.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data) / sr:.2f}s",
          file=sys.stderr)

    print("[2/3] 频谱分带", file=sys.stderr)
    spec, freqs = spectrum_bands(data, sr, a.fps,
                                 a.fft_size, 96, 20.0, 16000.0)
    sub, tw = extract_bands(spec, freqs, a.fps,
                            a.sub_max, a.tweet_min)

    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height)

    print(f"[3/3] 渲染 {len(sub)} 帧 → {out_path}", file=sys.stderr)
    render(sub, tw, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

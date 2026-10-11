#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum3DAudacious.py —— Audacious 风格 3D 频谱（第六版）

修改:
  - 加入俯视角度：相机从上方往下斜看频谱墙
  - 摆动周期放慢（默认 15 秒一个来回）
  - 保留角度渐变

空间:
  X = 频率
  Y = 时间（往纵深）
  Z = 响度（高度）
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (4, 5, 10)
TEXT_COL = (200, 210, 220)


# ==========================================================================
# 色环（红起，360°）
# ==========================================================================
def hue_to_rgb(h):
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    q = 1 - f
    if i == 0:   r, g, b = 1, f, 0
    elif i == 1: r, g, b = q, 1, 0
    elif i == 2: r, g, b = 0, 1, f
    elif i == 3: r, g, b = 0, q, 1
    elif i == 4: r, g, b = f, 0, 1
    else:        r, g, b = 1, 0, q
    return (int(r * 255), int(g * 255), int(b * 255))


def shade(col, k):
    return tuple(int(min(255, max(0, c * k))) for c in col)


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
# FFT
# ==========================================================================
def spectrum_bands(data, sr, fps, fft_size, bands,
                   fmin=40.0, fmax=14000.0):
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
        if i % 400 == 0:
            print(f"\r  FFT {i * 100 // n_frames:3d}%",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


# ==========================================================================
# 投影（俯视 + 旋转）
# ==========================================================================
def project(x, y, z, W, H, yaw, cfg):
    """
    x: 频率轴 [-1, 1]
    y: 时间轴 [0, 1]，0=最近，1=最远
    z: 响度 [0, 1]

    俯视效果:
      相机从上方斜着往下看。俯视角 pitch 决定：
        pitch = 0°  → 纯侧视（只能看到柱子的侧面）
        pitch = 90° → 纯俯视（只能看到柱子的顶面）
        pitch = 30° → 斜俯视（既看到侧面又看到纵深）
    """
    wx = x * cfg["x_span"]
    wy = (y - 0.5) * cfg["depth"]
    wz = z

    # 水平旋转（绕 Z 轴）
    yaw_total = yaw + math.radians(cfg["yaw_offset"])
    ca = math.cos(yaw_total)
    sa = math.sin(yaw_total)
    rx = wx * ca - wy * sa
    ry = wx * sa + wy * ca   # 深度方向（时间轴）
    rz = wz                  # 高度

    # 俯视：把 y 轴和 z 轴按 pitch 混合到屏幕的纵向
    pitch = math.radians(cfg["pitch"])
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    # 屏幕上的纵向 = 高度 * cos(pitch) + 纵深 * sin(pitch)
    # 注意：纵深越大（越远），屏幕上越往上 → 加负号
    screen_y = -rz * cp - ry * sp
    # 深度（用于透视和排序）
    depth = ry * cp - rz * sp

    # 透视
    persp = 1.0 / (1.0 + depth * cfg["persp"])

    sx = W * 0.5 + rx * W * cfg["scale_x"] * persp
    sy = H * cfg["y_offset"] + screen_y * H * cfg["scale_y"] * persp

    return sx, sy, depth


# ==========================================================================
# 角度渐变
# ==========================================================================
def angular_color(x, y):
    dx = x
    dy = (y - 0.5) * 2.0
    theta = math.atan2(dy, dx)
    hue = (theta / (2 * math.pi) + 0.5) % 1.0
    return hue_to_rgb(hue)


# ==========================================================================
# 渲染
# ==========================================================================
def render(spec_db, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames, bands = spec_db.shape
    db_min, db_max = cfg["db_min"], cfg["db_max"]

    norm = np.clip((spec_db - db_min) / (db_max - db_min), 0, 1)
    norm = norm ** 0.7

    rows = cfg["rows"]

    bar_xs = np.linspace(-1.0, 1.0, bands)
    bar_w = (bar_xs[1] - bar_xs[0]) * 0.42 if bands > 1 else 0.05
    t_step = 1.0 / max(rows - 1, 1)
    half_d = t_step * 0.5

    osc_period = cfg["osc_period"]
    osc_amp = math.radians(cfg["osc_angle"])

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
    report = max(1, n_frames // 100)

    try:
        for i in range(n_frames):
            t = i / float(fps)
            phase = (t / osc_period) * 2 * math.pi
            yaw = math.sin(phase) * osc_amp

            canvas = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(canvas, "RGBA")

            prims = []

            start = max(0, i - rows + 1)
            n_used = i - start + 1

            for j in range(n_used):
                frame_idx = start + j
                t_pos = 1.0 - (n_used - 1 - j) * t_step
                row = norm[frame_idx]

                for b in range(bands):
                    h = float(row[b])
                    if h < 0.008:
                        continue
                    x_c = float(bar_xs[b])

                    p_bl = project(x_c - bar_w, t_pos + half_d, 0.0,
                                   W, H, yaw, cfg)
                    p_br = project(x_c + bar_w, t_pos + half_d, 0.0,
                                   W, H, yaw, cfg)
                    p_fr = project(x_c + bar_w, t_pos - half_d, 0.0,
                                   W, H, yaw, cfg)
                    p_fl = project(x_c - bar_w, t_pos - half_d, 0.0,
                                   W, H, yaw, cfg)

                    top_bl = project(x_c - bar_w, t_pos + half_d, h,
                                     W, H, yaw, cfg)
                    top_br = project(x_c + bar_w, t_pos + half_d, h,
                                     W, H, yaw, cfg)
                    top_fr = project(x_c + bar_w, t_pos - half_d, h,
                                     W, H, yaw, cfg)
                    top_fl = project(x_c - bar_w, t_pos - half_d, h,
                                     W, H, yaw, cfg)

                    depth = (p_bl[2] + p_br[2] + p_fr[2] + p_fl[2]) * 0.25

                    col = angular_color(x_c, t_pos)

                    prims.append({
                        "depth": depth,
                        "bottom": [p_bl[:2], p_br[:2], p_fr[:2], p_fl[:2]],
                        "top": [top_bl[:2], top_br[:2],
                                top_fr[:2], top_fl[:2]],
                        "col": col,
                    })

            prims.sort(key=lambda p: -p["depth"])

            for p in prims:
                bot = p["bottom"]
                top = p["top"]
                col = p["col"]

                side_shades = [0.55, 0.78, 0.65, 0.5]
                for k in range(4):
                    k2 = (k + 1) % 4
                    face = [bot[k], bot[k2], top[k2], top[k]]
                    draw.polygon(face, fill=shade(col, side_shades[k]))

                draw.polygon(top, fill=shade(col, 1.18))

            draw.text((24, 22), "AUDACIOUS 3D SPECTRUM",
                      fill=TEXT_COL)

            proc.stdin.write(canvas.tobytes("raw", "RGB"))

            if i % report == 0:
                print(f"\r  渲染 {i * 100 // n_frames:3d}%",
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
        description="Audacious gl-spectrum 风格（第六版）")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--bars", type=int, default=28)
    ap.add_argument("--rows", type=int, default=36)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--fmin", type=float, default=40.0)
    ap.add_argument("--fmax", type=float, default=14000.0)
    ap.add_argument("--db-min", type=float, default=-85.0)
    ap.add_argument("--db-max", type=float, default=-5.0)

    # 摆动（放慢）
    ap.add_argument("--osc-period", type=float, default=15.0,
                    help="摆动周期（秒），越大越慢")
    ap.add_argument("--osc-angle", type=float, default=120.0,
                    help="摆动幅度（度）")

    # 相机
    ap.add_argument("--pitch", type=float, default=35.0,
                    help="俯视角（度）。0=侧视，90=纯俯视")
    ap.add_argument("--yaw-offset", type=float, default=20.0,
                    help="初始偏航角（度）")

    # 透视
    ap.add_argument("--persp", type=float, default=0.10)
    ap.add_argument("--x-span", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=1.6)
    ap.add_argument("--scale-x", type=float, default=0.42)
    ap.add_argument("--scale-y", type=float, default=0.32)
    ap.add_argument("--y-offset", type=float, default=0.68)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_audacious.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s",
          file=sys.stderr)

    print("[2/3] FFT 分带", file=sys.stderr)
    spec = spectrum_bands(data, sr, a.fps, a.fft_size, a.bars,
                          a.fmin, a.fmax)

    cfg = dict(
        input=a.input, fps=a.fps, width=a.width, height=a.height,
        bars=a.bars, rows=a.rows,
        db_min=a.db_min, db_max=a.db_max,
        osc_period=a.osc_period, osc_angle=a.osc_angle,
        pitch=a.pitch, yaw_offset=a.yaw_offset,
        persp=a.persp, x_span=a.x_span, depth=a.depth,
        scale_x=a.scale_x, scale_y=a.scale_y,
        y_offset=a.y_offset,
    )

    print(f"[3/3] 渲染 {spec.shape[0]} 帧 → {out_path}",
          file=sys.stderr)
    render(spec, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

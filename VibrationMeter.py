#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VibrationMeter.py —— 震力计（低频震动强度表）

原理：提取 120Hz 以下低频能量，平滑后驱动指针 + 震动波纹。
支持指针回落、峰值保持。

用法:
    python3 VibrationMeter.py 歌曲.mp3
    python3 VibrationMeter.py 歌曲.mp3 --release 0.6 --lowpass 100
    python3 VibrationMeter.py 歌曲.mp3 --width 720 --height 1280 --fps 30
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

# 配色
BG = (16, 17, 21)
DIAL = (30, 32, 40)
TICK = (180, 185, 195)
TEXT = (210, 215, 225)
NEEDLE = (255, 90, 70)
PEAK = (255, 255, 255)
GLOW = (255, 120, 90)


# ==========================================================================
# 音频读取
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
# 震动包络（全 numpy 向量化）
# ==========================================================================
def compute_vibration(data, sr, fps, lowpass_hz=120.0, window_ms=50.0):
    """
    低频震动强度：移动平均低通 + 短窗 RMS
    """
    n, ch = data.shape
    mono = data.mean(axis=1) if ch > 1 else data[:, 0]

    # 移动平均低通（近似一阶低通，向量化）
    win_lp = max(1, int(sr / lowpass_hz))
    kernel = np.ones(win_lp, dtype=np.float64) / win_lp
    lp = np.convolve(mono.astype(np.float64), kernel, mode="same")

    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    win = max(1, int(sr * window_ms / 1000.0))
    half = win // 2

    sq = lp ** 2
    csum = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(sq, out=csum[1:])

    centers = ((np.arange(n_frames) + 0.5) * hop).astype(np.int64)
    a = np.clip(centers - half, 0, n - 1)
    b = np.clip(centers + half, 1, n)
    b = np.maximum(b, a + 1)

    rms = np.sqrt(np.maximum((csum[b] - csum[a]) / (b - a), 0))

    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    norm = np.clip((db + 50.0) / 50.0, 0.0, 1.0)
    return norm.astype(np.float32)


def apply_ballistics(norm, fps, release_sec=0.5, peak_hold_sec=1.5):
    n = len(norm)
    ptr = np.empty(n, dtype=np.float32)
    pk = np.empty(n, dtype=np.float32)
    rel = 1.0 / max(release_sec * fps, 1e-6)
    pks = 1.0 / max(peak_hold_sec * fps, 1e-6)

    cur = 0.0
    peak = 0.0
    for i in range(n):
        v = float(norm[i])
        cur = max(v, cur - rel)
        peak = max(v, peak - pks)
        ptr[i] = cur
        pk[i] = peak
    return ptr, pk


# ==========================================================================
# 绘制震力计
# ==========================================================================
def draw_vibration_meter(canvas, cx, cy, radius,
                         value, peak=None,
                         label="VIBRATION",
                         ring_count=3):
    draw = ImageDraw.Draw(canvas, "RGBA")
    v = max(0.0, min(1.0, float(value)))

    # 外圈震动波纹
    for k in range(ring_count):
        rr = radius + 8 + k * 14
        alpha = int(max(0, 120 - k * 35) * (0.3 + 0.7 * v))
        if alpha <= 0:
            continue
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     outline=GLOW + (alpha,), width=2)

    # 表盘底
    draw.ellipse([cx - radius, cy - radius,
                  cx + radius, cy + radius], fill=DIAL)
    draw.ellipse([cx - radius, cy - radius,
                  cx + radius, cy + radius],
                 outline=(70, 75, 85, 220), width=2)

    # 刻度
    n_ticks = 28
    for i in range(n_ticks + 1):
        t = i / n_ticks
        ang = math.radians(-135 + 270 * t)
        r0 = radius * 0.85
        r1 = radius * (0.95 if i % 4 == 0 else 0.90)
        x0 = cx + math.cos(ang) * r0
        y0 = cy + math.sin(ang) * r0
        x1 = cx + math.cos(ang) * r1
        y1 = cy + math.sin(ang) * r1
        col = TICK if i % 4 == 0 else (TICK[0], TICK[1], TICK[2], 120)
        draw.line([x0, y0, x1, y1], fill=col,
                  width=2 if i % 4 == 0 else 1)

    # 指针
    ang = math.radians(-135 + 270 * v)
    nx = cx + math.cos(ang) * radius * 0.75
    ny = cy + math.sin(ang) * radius * 0.75
    draw.line([cx, cy, nx, ny], fill=NEEDLE, width=3)
    draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=NEEDLE)

    # 峰值小三角
    if peak is not None:
        pv = max(0.0, min(1.0, float(peak)))
        pang = math.radians(-135 + 270 * pv)
        px = cx + math.cos(pang) * radius * 0.82
        py = cy + math.sin(pang) * radius * 0.82
        draw.polygon([(px, py - 5), (px - 5, py + 4), (px + 5, py + 4)],
                     fill=PEAK)

    # 中心数值
    pct = int(v * 100)
    txt = f"{pct}%"
    bbox = draw.textbbox((0, 0), txt)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw / 2, cy + radius * 0.45), txt, fill=TEXT)

    # 标签
    if label:
        bbox = draw.textbbox((0, 0), label)
        lw = bbox[2] - bbox[0]
        draw.text((cx - lw / 2, cy + radius + 10), label, fill=TEXT)


# ==========================================================================
# 渲染视频
# ==========================================================================
def render(ptr, pk, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n = len(ptr)
    radius = cfg["radius"]

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
            draw_vibration_meter(canvas, W // 2, H // 2, radius,
                                 float(ptr[i]),
                                 float(pk[i]) if cfg["peak_hold"] else None,
                                 label="VIBRATION")
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
# 主函数
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="震力计（低频震动强度表）")
    ap.add_argument("input", help="输入音频文件")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--radius", type=int, default=220,
                    help="表盘半径（像素）")

    ap.add_argument("--lowpass", type=float, default=120.0,
                    help="低频截止频率（Hz）")
    ap.add_argument("--window", type=float, default=50.0,
                    help="RMS 窗口（毫秒）")
    ap.add_argument("--release", type=float, default=0.5,
                    help="指针回落时间（秒）")
    ap.add_argument("--peak-hold", type=float, default=1.5,
                    help="峰值保持时间（秒）")

    ap.add_argument("--no-peak-hold", dest="peak_hold",
                    action="store_false", help="关闭峰值保持")
    ap.set_defaults(peak_hold=True)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装")

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_vibration.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data) / sr:.2f}s",
          file=sys.stderr)

    print("[2/3] 计算震动包络", file=sys.stderr)
    vib = compute_vibration(data, sr, a.fps, a.lowpass, a.window)
    ptr, pk = apply_ballistics(vib, a.fps, a.release, a.peak_hold)

    cfg = {
        "input": a.input, "fps": a.fps, "width": a.width,
        "height": a.height, "radius": a.radius,
        "peak_hold": a.peak_hold,
    }

    print(f"[3/3] 渲染 {len(ptr)} 帧 → {out_path}", file=sys.stderr)
    render(ptr, pk, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

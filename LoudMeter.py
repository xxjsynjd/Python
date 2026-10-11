#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoudMeter.py —— 实时响度计（BS.1770 Momentary LUFS）

用法:
    python3 LoudMeter.py 歌曲.mp3
    python3 LoudMeter.py 歌曲.mp3 -o 输出.mp4 --release 0.5
    python3 LoudMeter.py 歌曲.mp3 --width 720 --height 1280 --fps 30
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
DIAL_BG = (30, 32, 40)
TICK_COL = (180, 185, 195)
TEXT_COL = (210, 215, 225)
BAR_COL = (80, 200, 120)
CURSOR_COL = (255, 255, 255)
TARGET_COL = (255, 200, 40)
PEAK_COL = (255, 255, 255)


# ==========================================================================
# 音频读取（用 ffmpeg 解码，避开 soundfile 在安卓上的兼容问题）
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
# K 加权滤波器（BS.1770 标准，不依赖 scipy，自己实现双线性变换）
# ==========================================================================
def design_k_weighting_sos(fs):
    filters = []

    # 第一级：高架 +4dB @ ~1.5kHz
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    K = math.tan(math.pi * fc / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b0 = (1.0 + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - 1.0) / a0
    b2 = (1.0 - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    filters.append([b0, b1, b2, 1.0, a1, a2])

    # 第二级：高通 -3dB @ ~38Hz
    G, Q, fc = 0.0, 0.5003270373238773, 38.13547087602444
    K = math.tan(math.pi * fc / fs)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b0 = (1.0 + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - 1.0) / a0
    b2 = (1.0 - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    filters.append([b0, b1, b2, 1.0, a1, a2])

    return np.array(filters)


def apply_filter(sos, x):
    """简单的二阶节 IIR 滤波（直接型 I）"""
    y = np.zeros_like(x)
    for b0, b1, b2, a0, a1, a2 in sos:
        x1 = x2 = y1 = y2 = 0.0
        for i in range(len(x)):
            v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            y[i] = v
            x2, x1 = x1, x[i]
            y2, y1 = y1, v
    return y


# ==========================================================================
# 实时响度计算
# ==========================================================================
def realtime_lufs(data, sr, fps, window_ms=400.0):
    """
    逐帧瞬时响度（Momentary LUFS）
    """
    if data.ndim == 1:
        data = data[:, None]

    sos = design_k_weighting_sos(sr)
    filtered = np.empty_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        filtered[:, ch] = apply_filter(sos, data[:, ch].astype(np.float64))

    # 声道能量求和
    sq = np.sum(filtered ** 2, axis=1)

    n = len(sq)
    win = max(1, int(sr * window_ms / 1000.0))
    half = win // 2

    csum = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(sq, out=csum[1:])

    hop = sr / float(fps)
    n_frames = max(1, int(np.ceil(n / hop)))
    centers = ((np.arange(n_frames) + 0.5) * hop).astype(np.int64)

    a = np.clip(centers - half, 0, n - 1)
    b = np.clip(centers + half, 1, n)
    b = np.maximum(b, a + 1)

    means = (csum[b] - csum[a]) / (b - a)

    lufs = -0.691 + 10.0 * np.log10(np.maximum(means, 1e-10))
    return lufs


def apply_ballistics(lufs, fps, release_sec=0.5, peak_hold_sec=1.5):
    n = len(lufs)
    ptr = np.empty(n, dtype=np.float32)
    pk = np.empty(n, dtype=np.float32)
    rel = 1.0 / max(release_sec * fps, 1e-6)
    pks = 1.0 / max(peak_hold_sec * fps, 1e-6)

    cur = lufs[0] if n > 0 else -60.0
    peak = cur

    for i in range(n):
        v = lufs[i]
        cur = max(v, cur - rel)
        peak = max(v, peak - pks)
        ptr[i] = cur
        pk[i] = peak
    return ptr, pk


# ==========================================================================
# 绘制响度计
# ==========================================================================
def draw_lufs_meter(canvas, x, y, w, h,
                    lufs_now, peak_val,
                    lufs_target=-14.0,
                    db_min=-60.0, db_max=0.0):
    draw = ImageDraw.Draw(canvas, "RGBA")

    def xp(db):
        return x + int((db - db_min) / (db_max - db_min) * w)

    # 底槽
    draw.rectangle([x, y, x + w, y + h], fill=DIAL_BG)

    # 刻度区段：绿 / 黄 / 红
    draw.rectangle([xp(-24), y, xp(-9), y + h], fill=(40, 130, 70))
    draw.rectangle([xp(-9), y, xp(-6), y + h], fill=(170, 140, 40))
    draw.rectangle([xp(-6), y, x + w, y + h], fill=(170, 50, 45))

    # 刻度线
    for db in (-36, -24, -14, -9, -6):
        px = xp(db)
        draw.line([px, y - 3, px, y + h + 3], fill=TICK_COL, width=1)

    # 当前值游标
    cur_x = max(x, min(x + w, xp(lufs_now)))
    draw.rectangle([x, y, cur_x, y + h], fill=BAR_COL)
    draw.line([cur_x, y - 4, cur_x, y + h + 4], fill=CURSOR_COL, width=2)

    # 峰值线
    if peak_val is not None:
        px = max(x, min(x + w - 2, xp(peak_val)))
        draw.rectangle([px, y, px + 2, y + h], fill=PEAK_COL)

    # 目标线（小三角）
    tx = xp(lufs_target)
    draw.polygon([(tx, y - 6), (tx - 5, y - 14), (tx + 5, y - 14)],
                 fill=TARGET_COL)

    # 数值文字
    txt = f"{lufs_now:.1f} LUFS"
    draw.text((x + 6, y + h + 10), txt, fill=TEXT_COL)


# ==========================================================================
# 渲染视频
# ==========================================================================
def render(ptr, pk, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n = len(ptr)

    mw = int(W * 0.8)
    mh = int(H * 0.08)
    mx = (W - mw) // 2
    my = int(H * 0.5)

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
            draw_lufs_meter(canvas, mx, my, mw, mh,
                            float(ptr[i]), float(pk[i]),
                            lufs_target=cfg["lufs_target"])
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
    ap = argparse.ArgumentParser(description="实时响度计 LUFS")
    ap.add_argument("input", help="输入音频文件")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--release", type=float, default=0.5,
                    help="指针回落时间（秒）")
    ap.add_argument("--peak-hold", type=float, default=1.5,
                    help="峰值保持时间（秒）")
    ap.add_argument("--window", type=float, default=400.0,
                    help="响度窗口（毫秒）")
    ap.add_argument("--lufs-target", type=float, default=-14.0,
                    help="目标 LUFS")

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装")

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_lufs.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data) / sr:.2f}s",
          file=sys.stderr)

    print("[2/3] 计算响度", file=sys.stderr)
    lufs = realtime_lufs(data, sr, a.fps, a.window)
    ptr, pk = apply_ballistics(lufs, a.fps, a.release, a.peak_hold)

    cfg = {
        "input": a.input, "fps": a.fps, "width": a.width,
        "height": a.height, "lufs_target": a.lufs_target,
    }

    print(f"[3/3] 渲染 {len(ptr)} 帧 → {out_path}", file=sys.stderr)
    render(ptr, pk, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

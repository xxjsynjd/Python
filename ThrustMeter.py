#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThrustMeter.py —— 音频推力计（低频冲击感 + 指针回落 + 峰值保持）

用法:
    python3 ThrustMeter.py 歌曲.mp3
    python3 ThrustMeter.py 歌曲.mp3 -o out.mp4 --release 0.4
    python3 ThrustMeter.py 歌曲.mp3 --width 720 --height 1280 --fps 30
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
RAIL = (40, 43, 52)
TEXT_COL = (200, 205, 215)
PEAK_COL = (255, 255, 255, 240)

# 推力颜色渐变：绿 → 黄 → 橙 → 红
COLOR_STOPS = [
    (0.00, (40, 180, 90)),
    (0.45, (150, 220, 60)),
    (0.70, (250, 200, 45)),
    (0.85, (255, 130, 40)),
    (1.00, (255, 55, 45)),
]


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
# 推力包络（全 numpy 向量化，不用 scipy）
# ==========================================================================
def compute_thrust(data, sr, fps, window_ms=40.0, lowpass_hz=150.0):
    """
    推力 = 短窗 RMS 的加权（低频权重更高，更接近"推力"感）
    返回 (n_frames,) 0~1
    """
    n, ch = data.shape
    mono = data.mean(axis=1) if ch > 1 else data[:, 0]

    # 低频提取：用移动平均近似低通（向量化，比 IIR 循环快得多）
    win_lp = max(1, int(sr / lowpass_hz))
    kernel = np.ones(win_lp, dtype=np.float64) / win_lp
    lp = np.convolve(mono.astype(np.float64), kernel, mode="same")

    # 滑窗 RMS
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    win = max(1, int(sr * window_ms / 1000.0))
    half = win // 2

    sq_full = mono.astype(np.float64) ** 2
    sq_low = lp ** 2

    csum_full = np.zeros(n + 1, dtype=np.float64)
    csum_low = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(sq_full, out=csum_full[1:])
    np.cumsum(sq_low, out=csum_low[1:])

    centers = ((np.arange(n_frames) + 0.5) * hop).astype(np.int64)
    a = np.clip(centers - half, 0, n - 1)
    b = np.clip(centers + half, 1, n)
    b = np.maximum(b, a + 1)
    cnt = (b - a).astype(np.float64)

    rms_full = np.sqrt((csum_full[b] - csum_full[a]) / cnt)
    rms_low = np.sqrt((csum_low[b] - csum_low[a]) / cnt)

    # 推力 = 0.4 * 全频 + 0.6 * 低频
    thrust = 0.4 * rms_full + 0.6 * rms_low

    db = 20.0 * np.log10(np.maximum(thrust, 1e-10))
    norm = np.clip((db + 50.0) / 50.0, 0.0, 1.0)
    return norm.astype(np.float32)


def apply_ballistics(norm, fps, release_sec=0.4, peak_hold_sec=1.2):
    """指针回落 + 峰值保持（向量化近似，用 np.maximum.accumulate 不方便，
    这里还是用循环，但只对 0~1 的数组，帧数不多）"""
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
# 绘制推力计
# ==========================================================================
def color_for(frac):
    for i in range(len(COLOR_STOPS) - 1):
        a = COLOR_STOPS[i]
        b = COLOR_STOPS[i + 1]
        if frac <= b[0]:
            t = (frac - a[0]) / max(b[0] - a[0], 1e-6)
            return tuple(int(a[1][k] + (b[1][k] - a[1][k]) * t) for k in range(3))
    return COLOR_STOPS[-1][1]


def draw_thrust_meter(canvas, x, y, w, h,
                      thrust, peak=None,
                      label="THRUST", seg=4):
    draw = ImageDraw.Draw(canvas, "RGBA")

    radius = max(3, h // 4)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=BG)

    pad = 3
    ix0, iy0 = x + pad, y + pad
    ix1, iy1 = x + w - pad, y + h - pad
    draw.rounded_rectangle([ix0, iy0, ix1, iy1],
                           radius=max(2, radius - 2), fill=RAIL)

    frac = max(0.0, min(1.0, float(thrust)))
    if frac > 0.001:
        fill_w = int((ix1 - ix0) * frac)
        seg_w = max(1, fill_w // 200)
        for px in range(0, fill_w, seg_w):
            f = px / max(ix1 - ix0, 1)
            col = color_for(f)
            draw.rectangle([ix0 + px, iy0,
                            ix0 + min(px + seg_w, fill_w), iy1],
                           fill=col)

    if seg > 1:
        for px in range(0, ix1 - ix0, seg):
            draw.line([(ix0 + px, iy0), (ix0 + px, iy1)],
                      fill=(BG[0], BG[1], BG[2], 90), width=1)

    if peak is not None:
        pfrac = max(0.0, min(1.0, float(peak)))
        if pfrac > 0.005:
            px = ix0 + int((ix1 - ix0) * pfrac)
            px = max(ix0, min(ix1 - 2, px))
            draw.rectangle([px, iy0, px + 2, iy1], fill=PEAK_COL)

    if label:
        draw.text((x + 6, y - 14), label, fill=TEXT_COL)

    pct = int(frac * 100)
    num = f"{pct:3d}%"
    draw.text((x + w - 40, y - 14), num, fill=TEXT_COL)


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
            draw_thrust_meter(canvas, mx, my, mw, mh,
                              float(ptr[i]),
                              float(pk[i]) if cfg["peak_hold"] else None,
                              label="THRUST",
                              seg=cfg["seg"])
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
    ap = argparse.ArgumentParser(description="音频推力计（低频冲击感）")
    ap.add_argument("input", help="输入音频文件")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--window", type=float, default=40.0,
                    help="推力计算窗口（毫秒）")
    ap.add_argument("--release", type=float, default=0.4,
                    help="指针回落时间（秒）")
    ap.add_argument("--peak-hold", type=float, default=1.2,
                    help="峰值保持时间（秒）")
    ap.add_argument("--seg", type=int, default=4,
                    help="分段刻痕像素，0 表示不分段")

    ap.add_argument("--no-peak-hold", dest="peak_hold",
                    action="store_false", help="关闭峰值保持")
    ap.set_defaults(peak_hold=True)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装")

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_thrust.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data) / sr:.2f}s",
          file=sys.stderr)

    print("[2/3] 计算推力", file=sys.stderr)
    thrust = compute_thrust(data, sr, a.fps, a.window)
    ptr, pk = apply_ballistics(thrust, a.fps, a.release, a.peak_hold)

    cfg = {
        "input": a.input, "fps": a.fps, "width": a.width,
        "height": a.height, "peak_hold": a.peak_hold, "seg": a.seg,
    }

    print(f"[3/3] 渲染 {len(ptr)} 帧 → {out_path}", file=sys.stderr)
    render(ptr, pk, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

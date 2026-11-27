#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhaseMeter.py —— 立体声相位表（Vectorscope / Goniometer）
原理:
  X 轴 = 差信号 (L - R) / sqrt(2)  —— 决定立体声宽度
  Y 轴 = 和信号 (L + R) / sqrt(2)  —— 决定单声道能量
  纯单声道（人声居中）→ 垂直一条线
  左右完全反相 → 水平一条线
  宽广立体声 → 45度角散开的云雾
用法:
    python3 PhaseMeter.py 歌曲.mp3 -o out.mp4
    python3 PhaseMeter.py --window 40 --decay 0.85
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

BG = (6, 8, 12)
GRID_COL = (30, 35, 45)
TEXT_COL = (180, 190, 205)
DIM_COL = (90, 95, 105)
TRACE_COL = (0, 255, 180)      # 荧光绿
TRACE_HOT = (255, 100, 80)     # 反相预警红
CORR_BAR_BG = (25, 28, 35)
CORR_POS = (60, 200, 120)
CORR_NEG = (255, 70, 60)

# ==========================================================================
# 音频读取（立体声专用）
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
    # 强制读出 2 声道，单声道会自动复制成双声道
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()
    return data, sr

def load_audio(path):
    if not os.path.isfile(path):
        raise SystemExit(f"找不到文件: {path}")
    if not FFMPEG or not FFPROBE:
        raise SystemExit("需要 ffmpeg 和 ffprobe")
    return _decode_ffmpeg(path)

# ==========================================================================
# 相位计算
# ==========================================================================
def compute_vectorscope(data, sr, fps, window_ms=30.0):
    """
    逐帧计算 Lissajous 点集和相关系数。
    返回 (n_frames, N, 2) 的坐标，以及 (n_frames,) 的相关系数。
    """
    n_samples = len(data)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n_samples / hop)))
    win = max(64, int(sr * window_ms / 1000.0))

    # 为了渲染速度，每帧均匀采样 N 个点
    N = 400

    vectors = np.zeros((n_frames, N, 2), dtype=np.float32)
    corr = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        c = int((i + 0.5) * hop)
        a = c - win // 2
        b = a + win

        # 边界处理（用 0 填充）
        L = np.zeros(win, dtype=np.float32)
        R = np.zeros(win, dtype=np.float32)
        if a < 0:
            L[-a:] = data[:b, 0]
            R[-a:] = data[:b, 1]
        elif b > n_samples:
            L[:n_samples - a] = data[a:, 0]
            R[:n_samples - a] = data[a:, 1]
        else:
            L[:] = data[a:b, 0]
            R[:] = data[a:b, 1]

        # 计算相关系数 (Correlation)
        std_L = np.std(L)
        std_R = np.std(R)
        if std_L > 1e-6 and std_R > 1e-6:
            corr[i] = np.corrcoef(L, R)[0, 1]
        else:
            corr[i] = 0.0

        # 抽取 N 个点
        idx = np.linspace(0, win - 1, N).astype(np.int64)
        # 映射到 X/Y 坐标
        # X = (L - R), Y = (L + R)
        vectors[i, :, 0] = L[idx] - R[idx]
        vectors[i, :, 1] = L[idx] + R[idx]

        if i % 500 == 0:
            print(f"\r  相位计算 {i*100//n_frames:3d}%", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return vectors, corr

# ==========================================================================
# 渲染
# ==========================================================================
def render(vectors, corr, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames = vectors.shape[0]
    # 屏幕中心与缩放
    cx = W // 2
    cy = H // 2
    disk_radius = min(W, H) * 0.42
    scale = disk_radius * cfg["scale"]
    # 持久化荧光屏缓冲
    buf = Image.new("RGB", (W, H), BG)
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
            # 1. 创建当帧的绘图层
            layer = Image.new("RGB", (W, H), (0, 0, 0))
            draw_layer = ImageDraw.Draw(layer)
            # 绘制所有采样点
            for k in range(vectors.shape[1]):
                x = cx + vectors[i, k, 0] * scale
                y = cy - vectors[i, k, 1] * scale
                if 0 <= x < W and 0 <= y < H:
                    energy = math.sqrt(vectors[i, k, 0]**2 + vectors[i, k, 1]**2)
                    if energy > 0.12:
                        draw_layer.point((x, y), fill=(100, 255, 200))
                    elif energy > 0.04:
                        draw_layer.point((x, y), fill=(60, 180, 130))
                    else:
                        draw_layer.point((x, y), fill=(35, 90, 65))

            # 收集点，绘制最外层轮廓外框
            points = []
            for k in range(vectors.shape[1]):
                x = cx + vectors[i, k, 0] * scale
                y = cy - vectors[i, k, 1] * scale
                energy = math.sqrt(vectors[i, k, 0]**2 + vectors[i, k, 1]**2)
                if energy > 0.04 and 0 <= x < W and 0 <= y < H:
                    points.append((x, y))

            if len(points) > 30:
                points = np.array(points, dtype=np.int32)
                angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
                order = np.argsort(angles)
                points = points[order]
                step = max(1, len(points) // 120)
                contour = points[::step]
                if len(contour) > 2:
                    contour = np.vstack([contour, contour[0]])
                    draw_layer.line(
                        [(int(p[0]), int(p[1])) for p in contour],
                        fill=(60, 160, 120),
                        width=5
                    )
                    draw_layer.line(
                        [(int(p[0]), int(p[1])) for p in contour],
                        fill=(140, 255, 200),
                        width=2
                    )
                    draw_layer.line(
                        [(int(p[0]), int(p[1])) for p in contour],
                        fill=(220, 255, 240),
                        width=1
                    )

            # 2. 混合到荧光屏缓冲（衰减实现拖尾）
            alpha_blend = min(0.45, 1.0 - cfg["decay"])
            buf = Image.blend(buf, layer, alpha=alpha_blend)

            # 3. 绘制静态 UI 元素
            canvas = buf.copy()
            draw = ImageDraw.Draw(canvas, "RGBA")
            r = int(min(W, H) * 0.42)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=GRID_COL, width=1)
            draw.ellipse([cx-r*0.66, cy-r*0.66, cx+r*0.66, cy+r*0.66], outline=GRID_COL, width=1)
            draw.ellipse([cx-r*0.33, cy-r*0.33, cx+r*0.33, cy+r*0.33], outline=GRID_COL, width=1)

            draw.line([cx-r, cy, cx+r, cy], fill=GRID_COL, width=1)
            draw.line([cx, cy-r, cx, cy+r], fill=GRID_COL, width=1)
            draw.line([cx-r*0.7, cy-r*0.7, cx+r*0.7, cy+r*0.7], fill=(40, 45, 55), width=1)
            draw.line([cx-r*0.7, cy+r*0.7, cx+r*0.7, cy-r*0.7], fill=(40, 45, 55), width=1)

            draw.text((cx-r, cy-r-20), "L", fill=DIM_COL)
            draw.text((cx+r-10, cy-r-20), "R", fill=DIM_COL)
            draw.text((cx-4, cy-r-20), "MONO", fill=DIM_COL)

            bar_w = W * 0.6
            bar_h = 16
            bx = (W - bar_w) / 2
            by = H - 60
            draw.rounded_rectangle([bx, by, bx + bar_w, by + bar_h],
                                   radius=bar_h//2, fill=CORR_BAR_BG)
            zero_x = bx + bar_w / 2
            draw.line([zero_x, by - 5, zero_x, by + bar_h + 5], fill=(100, 105, 115), width=2)

            c = corr[i]
            c = max(-1.0, min(1.0, c))
            val_w = (c + 1.0) / 2.0 * bar_w
            if c >= 0:
                draw.rounded_rectangle([zero_x, by+2, bx + val_w, by+bar_h-2],
                                       radius=(bar_h-4)//2, fill=CORR_POS)
            else:
                draw.rounded_rectangle([bx + val_w, by+2, zero_x, by+bar_h-2],
                                       radius=(bar_h-4)//2, fill=CORR_NEG)

            draw.text((bx, by - 24), "CORRELATION", fill=DIM_COL)
            draw.text((bx + bar_w - 40, by - 24), f"{c:+.2f}", fill=TEXT_COL)

            proc.stdin.write(canvas.tobytes("raw", "RGB"))
            if i % report == 0:
                print(f"\r渲染中… {i*100//n_frames:3d}%", end="", file=sys.stderr, flush=True)
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
    ap = argparse.ArgumentParser(description="立体声相位表 (Vectorscope / Goniometer)")
    ap.add_argument("input", help="输入音频文件（mp3 / wav / flac / m4a …）")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")
    ap.add_argument("--fps", type=int, default=30, help="视频帧率")
    ap.add_argument("--width", type=int, default=720, help="画面宽度（px）")
    ap.add_argument("--height", type=int, default=1280, help="画面高度（px）")
    ap.add_argument("--window", type=float, default=30.0,
                    help="分析窗口 ms (越小越灵敏，越大越稳定)")
    ap.add_argument("--decay", type=float, default=0.82,
                    help="荧光衰减系数 (0.5=快速消失，0.9=长拖尾)")
    ap.add_argument("--scale", type=float, default=0.38,
                    help="波形缩放比例")
    a = ap.parse_args()
    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_phase.mp4"
    print("[1/3] 读取音频（强制立体声）", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s", file=sys.stderr)
    if data.shape[1] < 2:
        print("      警告：原曲为单声道，相位表将显示为一条直线。", file=sys.stderr)
    print("[2/3] 计算利萨茹点集与相关系数", file=sys.stderr)
    vectors, corr = compute_vectorscope(data, sr, a.fps, a.window)
    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height,
               decay=a.decay, scale=a.scale)
    print(f"[3/3] 渲染 {vectors.shape[0]} 帧 → {out_path}", file=sys.stderr)
    render(vectors, corr, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NoiseGateMeter.py —— 噪声门指示灯（含迟滞 + 保持 + 阈值标尺）

原理:
  信号 > 开阈值 → 灯亮（门开）
  信号 < 关阈值 → 灯灭（门关）
  中间加 hold_ms 保持，防止信号抖动让灯狂闪

用法:
    python3 NoiseGateMeter.py 歌曲.mp3 -o out.mp4
    python3 NoiseGateMeter.py 歌曲.mp3 --open -30 --close -36 --hold 120
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

BG = (14, 15, 20)
DIAL = (30, 32, 40)
DIM_COL = (60, 65, 75)
TEXT_COL = (200, 210, 220)
GREEN_ON = (80, 240, 130)
GREEN_GLOW = (80, 240, 130, 90)
RED_OFF = (60, 62, 70)
THRESH_HI = (255, 220, 90)
THRESH_LO = (255, 130, 60)


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
# RMS 包络
# ==========================================================================
def compute_rms(data, sr, fps, window_ms=30.0):
    n, ch = data.shape
    mono = data.mean(axis=1) if ch > 1 else data[:, 0]

    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    win = max(1, int(sr * window_ms / 1000.0))
    half = win // 2

    sq = mono.astype(np.float64) ** 2
    csum = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(sq, out=csum[1:])

    centers = ((np.arange(n_frames) + 0.5) * hop).astype(np.int64)
    a = np.clip(centers - half, 0, n - 1)
    b = np.clip(centers + half, 1, n)
    b = np.maximum(b, a + 1)

    rms = np.sqrt(np.maximum((csum[b] - csum[a]) / (b - a), 0))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return db.astype(np.float32)


# ==========================================================================
# 噪声门状态机：开/关 + 迟滞 + 保持
# ==========================================================================
def compute_gate_state(db, fps, open_db=-30.0, close_db=-36.0, hold_ms=120.0):
    """
    返回 (state, since_switch)
      state: bool 数组，True = 门开
      since_switch: 每帧距离上次切换状态多少秒（用来做呼吸效果）
    """
    n = len(db)
    state = np.zeros(n, dtype=bool)
    since = np.zeros(n, dtype=np.float32)
    hold_frames = max(1, int(fps * hold_ms / 1000.0))

    opened = False
    last_switch = -10**9

    for i in range(n):
        v = float(db[i])
        if not opened and v > open_db:
            if i - last_switch > 2:  # 防止第一帧就狂切
                opened = True
                last_switch = i
        elif opened and v < close_db:
            if i - last_switch > hold_frames:
                opened = False
                last_switch = i
        state[i] = opened
        since[i] = (i - last_switch) / float(fps)
    return state, since


# ==========================================================================
# 绘制
# ==========================================================================
def draw_gate_panel(canvas, x, y, w, h,
                    db_now, is_open, since_switch,
                    open_db=-30.0, close_db=-36.0,
                    label="NOISE GATE"):
    """
    一块控制面板：
      - 左上：标签
      - 中上：大圆指示灯
      - 下方：阈值标尺（开/关两条线 + 当前指针）
      - 右上：OPEN / CLOSED 状态字
    """
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 面板
    radius = max(4, h // 20)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius,
                           fill=DIAL, outline=(60, 65, 75, 220), width=2)

    # 标签
    draw.text((x + 14, y + 12), label, fill=TEXT_COL)

    # ---- 指示灯 ----
    cx = x + w // 2
    cy = y + int(h * 0.36)
    r = int(h * 0.17)

    if is_open:
        # 开：绿色 + 光晕 + 呼吸
        breath = 0.85 + 0.15 * math.sin(since_switch * 6.0)
        glow_r = int(r * (1.8 + 0.2 * breath))
        draw.ellipse([cx - glow_r, cy - glow_r,
                      cx + glow_r, cy + glow_r],
                     fill=GREEN_GLOW)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(int(GREEN_ON[0] * breath),
                           int(GREEN_ON[1] * breath),
                           int(GREEN_ON[2] * breath)))
        # 高光
        draw.ellipse([cx - r // 2, cy - r // 2,
                      cx + r // 4, cy + r // 4],
                     fill=(255, 255, 255, 120))
        state_txt = "OPEN"
        state_col = GREEN_ON
    else:
        # 关：暗灰
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED_OFF)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=(90, 95, 105), width=2)
        state_txt = "CLOSED"
        state_col = (160, 170, 180)

    # 状态字
    bbox = draw.textbbox((0, 0), state_txt)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw / 2, cy + r + 10), state_txt, fill=state_col)

    # ---- 阈值标尺 ----
    bar_x0 = x + 20
    bar_x1 = x + w - 20
    bar_y = y + h - 40
    bar_h = 10

    # dB 范围映射：假设标尺显示 -60 ~ 0
    db_min, db_max = -60.0, 0.0

    def db_to_x(db):
        t = (db - db_min) / (db_max - db_min)
        t = max(0.0, min(1.0, t))
        return int(bar_x0 + t * (bar_x1 - bar_x0))

    # 底槽
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h],
                   fill=(25, 27, 33))

    # 色块区段
    for k in range(bar_x0, bar_x1, 3):
        t = (k - bar_x0) / max(bar_x1 - bar_x0, 1)
        if t < 0.4:
            c = (60, 90, 70)
        elif t < 0.7:
            c = (90, 130, 80)
        else:
            c = (130, 90, 70)
        draw.rectangle([k, bar_y, k + 2, bar_y + bar_h], fill=c)

    # 当前值指针
    px = db_to_x(float(db_now))
    px = max(bar_x0, min(bar_x1 - 3, px))
    draw.rectangle([px, bar_y - 4, px + 3, bar_y + bar_h + 4],
                   fill=(255, 255, 255))

    # 开阈值线（黄）
    ox = db_to_x(open_db)
    draw.line([ox, bar_y - 6, ox, bar_y + bar_h + 6],
              fill=THRESH_HI, width=2)

    # 关阈值线（橙）
    cxx = db_to_x(close_db)
    draw.line([cxx, bar_y - 4, cxx, bar_y + bar_h + 4],
              fill=THRESH_LO, width=1)

    # 刻度文字
    for d in (-60, -40, -30, -20, 0):
        px = db_to_x(d)
        draw.text((px - 10, bar_y + bar_h + 6), f"{d}",
                  fill=(140, 145, 155))

    # 数值
    draw.text((x + 20, bar_y - 22), f"{float(db_now):6.1f} dB",
              fill=TEXT_COL)


# ==========================================================================
# 渲染
# ==========================================================================
def render(db, state, since, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n = len(db)

    pw = int(W * 0.7)
    ph = int(H * 0.55)
    px = (W - pw) // 2
    py = (H - ph) // 2

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
            draw_gate_panel(canvas, px, py, pw, ph,
                            float(db[i]),
                            bool(state[i]),
                            float(since[i]),
                            open_db=cfg["open_db"],
                            close_db=cfg["close_db"])
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
    ap = argparse.ArgumentParser(
        description="噪声门指示灯（迟滞 + 保持 + 阈值标尺）")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--open", dest="open_db", type=float, default=-30.0,
                    help="开阈值 dB")
    ap.add_argument("--close", dest="close_db", type=float, default=-36.0,
                    help="关阈值 dB，要比开阈值低 3~6 dB")
    ap.add_argument("--hold", dest="hold_ms", type=float, default=120.0,
                    help="保持时间 ms，防止抖动")
    ap.add_argument("--window", type=float, default=30.0,
                    help="RMS 计算窗口 ms")

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    if a.close_db >= a.open_db:
        raise SystemExit("--close 必须小于 --open")

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_gate.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data) / sr:.2f}s",
          file=sys.stderr)

    print("[2/3] 计算 RMS 和门状态", file=sys.stderr)
    db = compute_rms(data, sr, a.fps, a.window)
    state, since = compute_gate_state(db, a.fps,
                                      a.open_db, a.close_db, a.hold_ms)
    open_ratio = float(state.mean()) * 100.0
    print(f"      门开占比: {open_ratio:.1f}%", file=sys.stderr)

    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height,
               open_db=a.open_db, close_db=a.close_db)

    print(f"[3/3] 渲染 {len(db)} 帧 → {out_path}", file=sys.stderr)
    render(db, state, since, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

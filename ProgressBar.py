#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProgressBar.py —— 纯播放进度条（独立视频）

特点:
  - 一条横条，从 0% 涨到 100%
  - 已播色块 + 底槽 + 拖拽圆点
  - 当前时间 / 总时间显示（可选）
  - 剩余时间倒计时（可选）
  - 圆角端点、渐变色

用法:
    python3 ProgressBar.py 歌曲.mp3 -o out.mp4
    python3 ProgressBar.py 歌曲.mp3 --thickness 24 --show-countdown
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (8, 9, 14)
RAIL = (35, 38, 46)
TEXT_COL = (210, 215, 225)
DIM_COL = (140, 148, 160)
KNOB_COL = (255, 255, 255)

# 进度条颜色渐变（左 → 右）
GRADIENT_STOPS = [
    (0.00, (60, 220, 140)),     # 绿
    (0.45, (90, 200, 255)),     # 蓝
    (0.70, (200, 180, 80)),     # 黄
    (0.88, (255, 130, 60)),     # 橙
    (1.00, (255, 60, 50)),      # 红
]


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
# 字体
# ==========================================================================
def load_font(path, size):
    if path and os.path.isfile(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    for p in (
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/system/fonts/NotoSansSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ==========================================================================
# 工具
# ==========================================================================
def format_time(sec):
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = int(sec - m * 60)
    if m > 0:
        return f"{m}:{s:02d}"
    return f"{s}s"


def color_at(t):
    """按比例取渐变色"""
    for i in range(len(GRADIENT_STOPS) - 1):
        a = GRADIENT_STOPS[i]
        b = GRADIENT_STOPS[i + 1]
        if t <= b[0]:
            k = (t - a[0]) / max(b[0] - a[0], 1e-6)
            return tuple(int(a[1][j] + (b[1][j] - a[1][j]) * k)
                         for j in range(3))
    return GRADIENT_STOPS[-1][1]


# ==========================================================================
# 绘制
# ==========================================================================
def draw_progress_bar(canvas, cx, cy, w, h,
                      cur_sec, total_sec,
                      font=None,
                      show_time=True,
                      show_countdown=False,
                      knob=True,
                      track_border=True):
    draw = ImageDraw.Draw(canvas, "RGBA")
    x0 = cx - w // 2
    x1 = cx + w // 2
    y0 = cy - h // 2
    y1 = cy + h // 2

    radius = h // 2

    # 底槽
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=RAIL)

    # 底槽边框
    if track_border:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                               outline=(70, 76, 88), width=1)

    # 已播色块（按渐变逐像素上色）
    frac = 0.0
    if total_sec > 0:
        frac = max(0.0, min(1.0, cur_sec / total_sec))
    fill_w = int(w * frac)
    if fill_w > 0:
        for px in range(0, fill_w, 2):
            t = px / max(w, 1)
            col = color_at(t)
            draw.rectangle([x0 + px, y0 + 1,
                            x0 + min(px + 2, fill_w), y1 - 1],
                           fill=col)

    # 圆角端点修正（把左右两端的圆角重新盖上）
    if fill_w > 0:
        r = radius
        # 左端圆角保留
        mask_l = Image.new("L", (r * 2, r * 2), 0)
        md = ImageDraw.Draw(mask_l)
        md.ellipse([0, 0, r * 2, r * 2], fill=255)
        canvas.paste(
            Image.new("RGB", (r * 2, r * 2),
                      color_at(0)),
            (x0, cy - r),
            mask_l)

    # 拖拽圆点
    if knob:
        kx = x0 + fill_w
        kx = max(x0 + radius, min(x1 - radius, kx))
        kr = radius + 2
        draw.ellipse([kx - kr, cy - kr, kx + kr, cy + kr],
                     fill=(20, 22, 28))
        draw.ellipse([kx - kr + 2, cy - kr + 2,
                      kx + kr - 2, cy + kr - 2],
                     fill=KNOB_COL)

    # 时间文字
    if font is not None and show_time:
        # 已播时间（左）
        left_txt = format_time(cur_sec)
        bbox = draw.textbbox((0, 0), left_txt, font=font)
        draw.text((x0, y1 + 12), left_txt, fill=TEXT_COL, font=font)

        # 总时间（右）
        right_txt = format_time(total_sec)
        bbox = draw.textbbox((0, 0), right_txt, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x1 - tw, y1 + 12), right_txt, fill=DIM_COL, font=font)

    if font is not None and show_countdown:
        # 剩余时间居中显示
        remain = max(0.0, total_sec - cur_sec)
        c_txt = f"-{format_time(remain)}"
        bbox = draw.textbbox((0, 0), c_txt, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw / 2, y1 + 12), c_txt,
                  fill=(200, 160, 90), font=font)


# ==========================================================================
# 渲染
# ==========================================================================
def render(total_sec, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames = max(1, int(math.ceil(total_sec * fps)))

    font = load_font(cfg["font"], cfg["font_size"]) if cfg["show_time"] \
        or cfg["show_countdown"] else None

    pw = int(W * cfg["width_ratio"])
    ph = cfg["thickness"]
    px = W // 2
    py = int(H * cfg["y_ratio"])

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
            canvas = Image.new("RGB", (W, H), BG)
            cur_sec = i / float(fps)

            draw_progress_bar(canvas, px, py, pw, ph,
                              cur_sec, total_sec,
                              font=font,
                              show_time=cfg["show_time"],
                              show_countdown=cfg["show_countdown"],
                              knob=cfg["knob"],
                              track_border=True)

            proc.stdin.write(canvas.tobytes("raw", "RGB"))
            if i % report == 0:
                print(f"\r渲染中… {i * 100 // n_frames:3d}%",
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
    ap = argparse.ArgumentParser(description="纯播放进度条")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--thickness", type=int, default=14,
                    help="进度条厚度 px")
    ap.add_argument("--width-ratio", type=float, default=0.8,
                    help="进度条宽度占画面比例")
    ap.add_argument("--y-ratio", type=float, default=0.5,
                    help="纵向位置 0~1")

    ap.add_argument("--font", default=None, help="中文字体路径")
    ap.add_argument("--font-size", type=int, default=22)

    ap.add_argument("--show-time", dest="show_time",
                    action="store_true", help="显示时间文字（默认开）")
    ap.add_argument("--no-time", dest="show_time",
                    action="store_false")
    ap.set_defaults(show_time=True)

    ap.add_argument("--show-countdown", dest="show_countdown",
                    action="store_true", help="显示剩余时间倒计时")
    ap.set_defaults(show_countdown=False)

    ap.add_argument("--no-knob", dest="knob",
                    action="store_false", help="不显示拖拽圆点")
    ap.set_defaults(knob=True)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_progress.mp4"

    print("[1/2] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    total_sec = len(data) / sr
    print(f"      {sr} Hz, {data.shape[1]}ch, {total_sec:.2f}s",
          file=sys.stderr)

    cfg = dict(input=a.input, fps=a.fps,
               width=a.width, height=a.height,
               thickness=a.thickness,
               width_ratio=a.width_ratio,
               y_ratio=a.y_ratio,
               font=a.font,
               font_size=a.font_size,
               show_time=a.show_time,
               show_countdown=a.show_countdown,
               knob=a.knob)

    print(f"[2/2] 渲染 {int(total_sec * a.fps)} 帧 → {out_path}",
          file=sys.stderr)
    render(total_sec, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

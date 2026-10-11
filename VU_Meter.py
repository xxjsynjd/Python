#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audiometer.py —— 音频 → 竖条式电平表视频（含原声）

用法示例:
    python audiometer.py song.mp3 -o meter.mp4
    python audiometer.py song.flac -o meter.mp4 --release 0.5 --release-db 20
    python audiometer.py song.wav  -o meter.mp4 --fps 60 --width 720 --height 1280
    python audiometer.py song.mp3  -o meter.mp4 --db-min -60 --no-peak-hold

依赖:
    pip install numpy pillow soundfile
    ffmpeg / ffprobe 在 PATH 中
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG_COLOR = (13, 14, 18)
PEAK_COLOR = (255, 255, 255)

# 电平条从下到上的颜色 (dB, (r, g, b))
COLOR_STOPS = [
    (-60.0, (16, 70, 45)),
    (-40.0, (20, 160, 80)),
    (-22.0, (90, 215, 70)),
    (-12.0, (190, 225, 55)),
    (-6.0, (250, 205, 40)),
    (-2.0, (255, 135, 25)),
    (0.0, (255, 70, 45)),
    (6.0, (255, 45, 45)),
]


# --------------------------------------------------------------------------
# 音频读取
# --------------------------------------------------------------------------
def _decode_with_ffmpeg(path):
    """用 ffmpeg 解码成 float32，返回 (data[n, ch], sample_rate)"""
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

    data = None
    sr = None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        data = None

    if data is None:
        if not (FFMPEG and FFPROBE):
            raise SystemExit(
                "读取音频失败：请 `pip install soundfile`，或确保 ffmpeg/ffprobe 在 PATH 中")
        data, sr = _decode_with_ffmpeg(path)

    data = np.ascontiguousarray(data, dtype=np.float32)
    return data, int(sr)


# --------------------------------------------------------------------------
# 包络 & 指针弹道
# --------------------------------------------------------------------------
def compute_envelope(data, sr, fps, window_ms):
    """按帧计算 RMS 包络（用滑动窗口，帧中心对齐）"""
    n, ch = data.shape
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))

    win = max(1, int(sr * window_ms / 1000.0))
    half = win // 2

    sq = data.astype(np.float64) ** 2
    csum = np.zeros((n + 1, ch), dtype=np.float64)
    csum[1:] = np.cumsum(sq, axis=0)

    centers = ((np.arange(n_frames) + 0.5) * hop).astype(np.int64)
    a = np.clip(centers - half, 0, n - 1)
    b = np.clip(centers + half, 1, n)
    b = np.maximum(b, a + 1)
    b = np.minimum(b, n)

    total = csum[b] - csum[a]
    count = (b - a).astype(np.float64)[:, None]
    rms = np.sqrt(np.maximum(total / count, 0.0))
    return rms, n_frames


def apply_ballistics(db, fps, release_db_per_sec, peak_release_db_per_sec=None):
    """
    指针弹道：
      - 上升：瞬时跟随（听觉上更接近真实电平表）
      - 下降：按 release_db_per_sec 线性回落（dB/s）—— 这就是“指针回落”

    返回 (pointer_db, peak_db)
    """
    n, ch = db.shape
    out = np.empty_like(db)
    peaks = np.empty_like(db)

    cur = np.full(ch, -120.0, dtype=np.float64)
    pk = np.full(ch, -120.0, dtype=np.float64)

    rel_step = release_db_per_sec / fps
    pk_step = (peak_release_db_per_sec if peak_release_db_per_sec
               else release_db_per_sec) / fps

    for i in range(n):
        tgt = db[i]
        # 回落（下降受限），上升瞬时
        cur = np.maximum(tgt, cur - rel_step)
        out[i] = cur
        # 峰值保持：跟随最大值，然后按自己的速度回落
        pk = np.maximum(tgt, pk - pk_step)
        peaks[i] = pk

    return out, peaks


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def build_bar_texture(bar_h, bar_w, db_min, db_max, seg_h):
    """预生成电平条的亮 / 暗两张贴图"""
    ys = np.arange(bar_h, dtype=np.float64)
    frac = (bar_h - 1 - ys) / max(bar_h - 1, 1)      # 0=底部, 1=顶部
    db = db_min + frac * (db_max - db_min)

    xp = np.array([c[0] for c in COLOR_STOPS], dtype=np.float64)
    arr = np.empty((bar_h, 3), dtype=np.float64)
    for k in range(3):
        fp = np.array([c[1][k] for c in COLOR_STOPS], dtype=np.float64)
        arr[:, k] = np.interp(db, xp, fp)

    arr = np.repeat(arr[:, None, :], bar_w, axis=1)   # (bar_h, bar_w, 3)

    # 分段效果（像 LED 灯条）
    if seg_h and seg_h > 1:
        rows = np.arange(bar_h)
        gap = ((bar_h - 1 - rows) % seg_h) == 0
        arr[gap, :, :] *= 0.22

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    bright = Image.fromarray(arr, "RGB")
    dim = Image.fromarray((arr.astype(np.float32) * 0.09).astype(np.uint8), "RGB")
    return bright, dim


def render_video(norm, peak_norm, cfg, out_path):
    W, H = cfg["width"], cfg["height"]
    n_ch = norm.shape[1]
    fps = cfg["fps"]

    margin_x = max(28, int(W * 0.075))
    margin_top = max(40, int(H * 0.06))
    margin_bot = max(50, int(H * 0.075))
    gap = max(10, int(W * 0.028))

    avail_w = W - 2 * margin_x - gap * (n_ch - 1)
    bar_w = max(6, avail_w // n_ch)
    bar_top = margin_top
    bar_h = H - margin_top - margin_bot
    if bar_h < 10:
        raise SystemExit("画布太小，请调大 --height")

    bright, dim = build_bar_texture(bar_h, bar_w, cfg["db_min"], cfg["db_max"],
                                    cfg["seg"])

    W = (W // 2) * 2
    H = (H // 2) * 2

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
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = norm.shape[0]
    report = max(1, total // 100)

    try:
        for i in range(total):
            canvas = Image.new("RGB", (W, H), BG_COLOR)

            for c in range(n_ch):
                x = margin_x + c * (bar_w + gap)

                # 底座（暗条）
                canvas.paste(dim, (x, bar_top))

                # 点亮部分
                fill = int(round(float(norm[i, c]) * bar_h))
                fill = max(0, min(bar_h, fill))
                if fill > 0:
                    region = bright.crop((0, bar_h - fill, bar_w, bar_h))
                    canvas.paste(region, (x, bar_top + bar_h - fill))

                # 峰值保持线
                if cfg["peak_hold"]:
                    pv = float(peak_norm[i, c])
                    if pv > 0.001:
                        py = bar_top + bar_h - int(round(pv * bar_h))
                        py = max(bar_top, min(bar_top + bar_h - 3, py))
                        canvas.paste(PEAK_COLOR, (x, py, x + bar_w, py + 3))

            proc.stdin.write(canvas.tobytes("raw", "RGB"))

            if i % report == 0:
                pct = i * 100 // total
                print(f"\r渲染中… {pct:3d}%", end="", file=sys.stderr, flush=True)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()

    print("\r渲染完成 ✔", file=sys.stderr)
    return proc.returncode


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="把音频文件转成竖条式电平表视频（混入原声）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("input", help="输入音频文件（mp3 / wav / flac / m4a …）")
    ap.add_argument("-o", "--output", default=None,
                    help="输出视频文件（默认 输入名_meter.mp4）")

    ap.add_argument("--fps", type=int, default=60, help="视频帧率")
    ap.add_argument("--width", type=int, default=720, help="画面宽度")
    ap.add_argument("--height", type=int, default=1080, help="画面高度")

    ap.add_argument("--db-min", type=float, default=-54.0, help="表盘最低电平 (dB)")
    ap.add_argument("--db-max", type=float, default=6.0, help="表盘最高电平 (dB)")

    # —— 指针回落参数 ——
    ap.add_argument("--release", type=float, default=0.5,
                    help="指针回落时间（秒）：指针回落 --release-db 分贝所需的时间")
    ap.add_argument("--release-db", type=float, default=20.0,
                    help="配合 --release 使用的回落分贝量")

    ap.add_argument("--window", type=float, default=35.0,
                    help="RMS 计算窗口（毫秒），越大越平滑")
    ap.add_argument("--seg", type=int, default=8,
                    help="电平条分段高度（像素），0 表示不分段")

    ap.add_argument("--peak-hold", dest="peak_hold", action="store_true",
                    help="开启峰值保持指示线（默认开启）")
    ap.add_argument("--no-peak-hold", dest="peak_hold", action="store_false",
                    help="关闭峰值保持指示线")
    ap.set_defaults(peak_hold=True)

    ap.add_argument("--peak-release", type=float, default=1.2,
                    help="峰值指示线回落 --release-db 分贝所需时间（秒）")

    args = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装并加入 PATH")

    if args.release <= 0:
        raise SystemExit("--release 必须大于 0")
    if args.db_max <= args.db_min:
        raise SystemExit("--db-max 必须大于 --db-min")

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = base + "_meter.mp4"

    # 指针回落速度 (dB/s)
    release_rate = args.release_db / args.release
    peak_rate = args.release_db / args.peak_release if args.peak_release > 0 else release_rate

    print(f"[1/4] 读取音频: {args.input}", file=sys.stderr)
    data, sr = load_audio(args.input)
    dur = len(data) / sr
    print(f"      采样率 {sr} Hz, 声道 {data.shape[1]}, 时长 {dur:.2f}s", file=sys.stderr)

    print(f"[2/4] 计算包络与弹道（回落 {release_rate:.1f} dB/s）", file=sys.stderr)
    rms, n_frames = compute_envelope(data, sr, args.fps, args.window)

    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    db = np.clip(db, -100.0, 30.0)

    ptr_db, peak_db = apply_ballistics(db, args.fps, release_rate, peak_rate)

    span = args.db_max - args.db_min
    norm = np.clip((ptr_db - args.db_min) / span, 0.0, 1.0).astype(np.float32)
    peak_norm = np.clip((peak_db - args.db_min) / span, 0.0, 1.0).astype(np.float32)

    cfg = {
        "input": args.input,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "db_min": args.db_min,
        "db_max": args.db_max,
        "seg": args.seg,
        "peak_hold": args.peak_hold,
    }

    print(f"[3/4] 渲染 {n_frames} 帧 @ {args.fps}fps → {args.width}x{args.height}",
          file=sys.stderr)
    rc = render_video(norm, peak_norm, cfg, out_path)

    if rc != 0:
        raise SystemExit(f"ffmpeg 编码失败（退出码 {rc}）")

    print(f"[4/4] 输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oscilloscope.py —— 时域波形示波器（采样点轮廓折线，非离散散点）

特性:
  - 每帧取一个时域窗口，采样点用折线连成连续轮廓线
  - 颜色支持十六进制输入（#RRGGBB / 0xRRGGBB，兼容十进制数值）；所有显示开关默认关闭，未指定的数值参数使用内置默认值
  - 支持导入 JSON 预设文件（命令行显式参数优先于预设）
  - 可开启峰谷值显示：峰值 / 谷值数值 + 峰、谷标记点（峰圈贴峰值线上方，谷圈贴谷值线下方）
  - 可开启峰谷值线段：峰值 / 谷值水平参考线（可选保持回落，保持时长可调；线性滑动平滑）
  - 可开启时间模糊淡出模式（Audacious 示波器余晖拖尾）：残影边淡出边高斯模糊扩散，
    可调拖尾保持时间（或衰减系数）与模糊半径

用法示例:
    python3 Oscilloscope.py 歌曲.mp3 -o scope.mp4
    python3 Oscilloscope.py 歌曲.mp3 --color #00ffb4 --show-peaks --peak-lines
    python3 Oscilloscope.py 歌曲.mp3 --preset scope_preset.json
    python3 Oscilloscope.py 歌曲.mp3 --trail --trail-hold 0.6 --blur 1.6
    python3 Oscilloscope.py 歌曲.mp3 --show-peaks --peak-lines --peak-hold --peak-hold-sec 1.2 --peak-ramp 0.4
    python3 Oscilloscope.py 歌曲.mp3 --window 25 --points 480 --line-width 2

依赖:
    pip install numpy pillow soundfile
    ffmpeg / ffprobe 在 PATH 中
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# 默认配色（RGB）
BG_COLOR = (8, 10, 14)
GRID_COL = (28, 33, 42)
TRACE_COL = (0, 255, 180)      # 波形轮廓线（荧光青）
PEAK_COL = (255, 96, 84)       # 峰值（暖红）
TROUGH_COL = (96, 168, 255)    # 谷值（冷蓝）
TEXT_COL = (190, 200, 215)
DIM_COL = (95, 100, 112)

# 预设文件允许读取的键（其余忽略，防止预设覆盖危险项）
PRESET_KEYS = {
    "color", "bg_color", "line_width", "window", "points", "scale",
    "show_peaks", "peak_lines", "peak_hold", "peak_hold_sec", "grid", "font",
    "trail", "decay", "peak_ramp", "trail_hold", "blur",
}


# --------------------------------------------------------------------------
# 颜色 / 字体
# --------------------------------------------------------------------------
def parse_color(value, default):
    """把十六进制（#RRGGBB / 0xRRGGBB）或十进制颜色值解析成 (r, g, b)。"""
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        if s.lower().startswith("0x"):
            v = int(s, 16)
        elif s.startswith("#"):
            v = int(s[1:], 16)
        else:
            v = int(s)
    else:
        v = int(value)
    v = v & 0xFFFFFF
    return (v >> 16, (v >> 8) & 0xFF, v & 0xFF)


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


# --------------------------------------------------------------------------
# 音频读取
# --------------------------------------------------------------------------
def _decode_ffmpeg(path):
    """用 ffmpeg 解码成 float32，返回 (data[n, ch], sample_rate)"""
    p = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    parts = p.stdout.strip().splitlines()[0].split(",")
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

    data, sr = None, None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        data = None

    if data is None:
        if not (FFMPEG and FFPROBE):
            raise SystemExit(
                "读取音频失败：请 `pip install soundfile`，或确保 ffmpeg/ffprobe 在 PATH 中")
        data, sr = _decode_ffmpeg(path)

    data = np.ascontiguousarray(data, dtype=np.float32)
    return data, int(sr)


# --------------------------------------------------------------------------
# 时域波形计算
# --------------------------------------------------------------------------
def compute_wave(data, sr, fps, window_ms, n_points):
    """逐帧截取时域窗口，均匀抽取 n_points 个采样点。

    返回 waves(n_frames, n_points)、peaks(n_frames)、troughs(n_frames)、帧数。
    """
    n = len(data)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    win = max(64, int(sr * window_ms / 1000.0))

    mono = data.mean(axis=1).astype(np.float32)

    waves = np.zeros((n_frames, n_points), dtype=np.float32)
    peaks = np.zeros(n_frames, dtype=np.float32)
    troughs = np.zeros(n_frames, dtype=np.float32)

    idx = np.linspace(0, win - 1, n_points).astype(np.int64)

    for i in range(n_frames):
        c = int((i + 0.5) * hop)
        a = c - win // 2
        b = a + win

        seg = np.zeros(win, dtype=np.float32)
        if a < 0:
            seg[-a:min(b, n)] = mono[:max(0, min(b, n))]
        elif b > n:
            seg[:n - a] = mono[a:]
        else:
            seg[:] = mono[a:b]

        waves[i] = seg[idx]
        peaks[i] = float(seg.max()) if len(seg) else 0.0
        troughs[i] = float(seg.min()) if len(seg) else 0.0

        if i % 500 == 0:
            print(f"\r  波形计算 {i * 100 // n_frames:3d}%",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return waves, peaks, troughs, n_frames


def apply_hold(peaks, troughs, fps, hold_sec):
    """峰谷值保持弹道：上升瞬时跟随，下降按 hold_sec 缓慢回落。

    峰值只向上保持，谷值只向下保持。返回 (peak_hold, trough_hold)。
    """
    n = len(peaks)
    out_p = np.zeros(n, dtype=np.float64)
    out_t = np.zeros(n, dtype=np.float64)

    cur_p, cur_t = 0.0, 0.0
    rel_p = 1.0 / max(hold_sec, 1e-3) / fps   # 每秒回落幅度，按归一化线性
    rel_t = rel_p

    for i in range(n):
        tp, tt = float(peaks[i]), float(troughs[i])
        cur_p = max(tp, cur_p - rel_p)
        cur_t = min(tt, cur_t + rel_t)
        out_p[i] = cur_p
        out_t[i] = cur_t
    return out_p.astype(np.float32), out_t.astype(np.float32)


def apply_ramp(peaks, troughs, fps, ramp_sec):
    """峰谷值线性滑动：输出值在 ramp_sec 秒内线性趋近目标值，双向平滑。

    每帧最大变化幅度受斜率限制，避免峰谷数值跳变。
    返回 (peak_ramp, trough_ramp)。
    """
    n = len(peaks)
    out_p = np.zeros(n, dtype=np.float64)
    out_t = np.zeros(n, dtype=np.float64)

    cur_p, cur_t = 0.0, 0.0
    rate = 1.0 / max(ramp_sec, 1e-3) / fps   # 每帧允许的最大线性变化（归一化）

    for i in range(n):
        tp, tt = float(peaks[i]), float(troughs[i])
        dp = tp - cur_p
        dt = tt - cur_t
        cur_p += max(-rate, min(rate, dp))
        cur_t += max(-rate, min(rate, dt))
        out_p[i] = cur_p
        out_t[i] = cur_t
    return out_p.astype(np.float32), out_t.astype(np.float32)


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def render(waves, peaks, troughs, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames = waves.shape[0]

    margin_x = max(24, int(W * 0.06))
    top_pad = max(56, int(H * 0.07))
    bot_pad = max(56, int(H * 0.07))
    cx = W // 2
    cy = H // 2
    amp_h = (H - top_pad - bot_pad) / 2.0
    scale = cfg["scale"]

    trace_col = cfg["color"]
    bg_col = cfg["bg_color"]
    line_w = cfg["line_width"]

    font = load_font(cfg.get("font"), max(16, int(H * 0.022)))

    trail = bool(cfg.get("trail", False))
    decay = float(cfg.get("decay", 0.85))
    trail_hold = float(cfg.get("trail_hold", 0.0))
    if trail_hold > 1e-3:
        # 由拖尾保持时间（秒）按指数衰减算每帧残留系数
        residual = math.exp(-(1.0 / fps) / trail_hold)
    else:
        residual = decay
    residual = min(0.98, max(0.5, residual))
    blur_radius = float(cfg.get("blur", 1.2))
    bg_img = Image.new("RGB", (W, H), bg_col)
    # 余晖缓冲：仅时间模糊淡出模式下持久保留历史波形
    buf = Image.new("RGB", (W, H), bg_col)

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
            seg = waves[i]
            n_pts = len(seg)
            xs = np.linspace(margin_x, W - margin_x, n_pts)
            ys = cy - seg * scale * amp_h

            # —— 峰谷水平位置（瞬时或保持值；是否画线由开关决定）——
            pk = float(peaks[i])
            tr = float(troughs[i])
            yp = int(cy - pk * scale * amp_h) if pk > 1e-3 else cy
            yt = int(cy - tr * scale * amp_h) if tr < -1e-3 else cy
            yp = max(top_pad, min(H - bot_pad, yp))
            yt = max(top_pad, min(H - bot_pad, yt))

            # —— 余晖层：峰谷线段 + 当前轮廓折线（时间模糊淡出作用于此层）——
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dl = ImageDraw.Draw(layer)
            if cfg["peak_lines"]:
                if pk > 1e-3:
                    dl.line([margin_x, yp, W - margin_x, yp],
                            fill=PEAK_COL, width=1)
                if tr < -1e-3:
                    dl.line([margin_x, yt, W - margin_x, yt],
                            fill=TROUGH_COL, width=1)

            points = list(zip(xs.tolist(), ys.tolist()))
            if len(points) >= 2:
                dl.line(points, fill=trace_col, width=line_w)

            if trail:
                # 旧余晖先高斯模糊扩散，再按残留系数向背景淡出，最后叠加当前轮廓
                blurred = buf.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                faded = Image.blend(blurred, bg_img, alpha=1.0 - residual)
                buf = Image.alpha_composite(
                    faded.convert("RGBA"), layer).convert("RGB")
                canvas = buf.copy()
            else:
                canvas = Image.new("RGB", (W, H), bg_col)
                canvas.paste(layer, (0, 0), layer)

            draw = ImageDraw.Draw(canvas, "RGBA")

            # —— 网格与中线（静态，不参与余晖）——
            if cfg["grid"]:
                for frac in (0.25, 0.5, 0.75):
                    gy = int(top_pad + frac * (H - top_pad - bot_pad))
                    draw.line([margin_x, gy, W - margin_x, gy],
                              fill=GRID_COL, width=1)
                draw.line([margin_x, cy, W - margin_x, cy],
                          fill=(48, 54, 66), width=1)

            # —— 峰谷值显示：数值文本 + 峰/谷标记点 ——
            # 峰值标记贴在峰值水平线上方临近，谷值标记贴在谷值水平线下方临近
            if cfg["show_peaks"]:
                k_pk = int(np.argmax(seg))
                k_tr = int(np.argmin(seg))
                xp_pk = xs[k_pk]
                xp_tr = xs[k_tr]
                r = max(3, line_w + 1)
                gap = r + 3
                y_pk_mark = max(top_pad, yp - gap)
                y_tr_mark = min(H - bot_pad, yt + gap)
                draw.ellipse([xp_pk - r, y_pk_mark - r, xp_pk + r, y_pk_mark + r],
                             outline=PEAK_COL, width=2)
                draw.ellipse([xp_tr - r, y_tr_mark - r, xp_tr + r, y_tr_mark + r],
                             outline=TROUGH_COL, width=2)

                draw.text((margin_x, top_pad - int(H * 0.045)),
                          f"PEAK  {pk:+.2f}", fill=PEAK_COL, font=font)
                draw.text((W - margin_x - 150, top_pad - int(H * 0.045)),
                          f"TROUGH {tr:+.2f}", fill=TROUGH_COL, font=font)

            proc.stdin.write(canvas.convert("RGB").tobytes("raw", "RGB"))

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
    return proc.returncode


# --------------------------------------------------------------------------
# 预设合并
# --------------------------------------------------------------------------
def load_preset(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise SystemExit(f"找不到预设文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit("预设文件必须是 JSON 对象")
    return {k: v for k, v in data.items() if k in PRESET_KEYS}


def main():
    ap = argparse.ArgumentParser(
        description="时域波形示波器（采样点轮廓折线）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("input", help="输入音频文件（mp3 / wav / flac / m4a …）")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)

    ap.add_argument("--preset", default=None,
                    help="导入 JSON 预设文件；命令行显式参数优先于预设")
    ap.add_argument("--color", default=None,
                    help="波形轮廓线颜色，十六进制（#RRGGBB / 0xRRGGBB，如 #00ffb4），兼容十进制数值")
    ap.add_argument("--bg-color", default=None,
                    help="背景色，十六进制（#RRGGBB / 0xRRGGBB，如 #080a0e），兼容十进制数值")
    ap.add_argument("--line-width", type=int, default=None,
                    help="波形轮廓线宽（像素），默认 2")
    ap.add_argument("--window", type=float, default=None,
                    help="时域分析窗口（毫秒），越小越灵敏，默认 30.0")
    ap.add_argument("--points", type=int, default=None,
                    help="每帧折线采样点数（轮廓线分辨率），默认 400")
    ap.add_argument("--scale", type=float, default=None,
                    help="波形振幅缩放比例，默认 1.0")
    ap.add_argument("--font", default=None, help="可选字体文件路径")

    ap.add_argument("--show-peaks", dest="show_peaks", action="store_true",
                    help="开启峰谷值显示（数值 + 峰/谷标记点）")
    ap.add_argument("--no-show-peaks", dest="show_peaks", action="store_false",
                    help="关闭峰谷值显示")
    ap.set_defaults(show_peaks=None)

    ap.add_argument("--peak-lines", dest="peak_lines", action="store_true",
                    help="开启峰谷值水平线段")
    ap.add_argument("--no-peak-lines", dest="peak_lines", action="store_false",
                    help="关闭峰谷值水平线段")
    ap.set_defaults(peak_lines=None)

    ap.add_argument("--peak-hold", dest="peak_hold", action="store_true",
                    help="峰谷线段保持并缓慢回落（配合 --peak-lines）")
    ap.add_argument("--no-peak-hold", dest="peak_hold", action="store_false",
                    help="峰谷线段使用瞬时值，不保持")
    ap.set_defaults(peak_hold=None)

    ap.add_argument("--peak-hold-sec", type=float, default=None,
                    help="峰谷值保持时长（秒，配合 --peak-hold）；峰值线 / 谷值线从保持极值逐渐向当前电平靠拢回落的时间，越大回落越慢")

    ap.add_argument("--peak-ramp", type=float, default=None,
                    help="峰谷值线性滑动时间（秒，0=关闭，默认 0）；峰谷读数在该时长内线性趋近目标值，避免跳变")

    ap.add_argument("--grid", dest="grid", action="store_true",
                    help="开启背景网格")
    ap.add_argument("--no-grid", dest="grid", action="store_false",
                    help="关闭背景网格")
    ap.set_defaults(grid=None)

    ap.add_argument("--trail", dest="trail", action="store_true",
                    help="开启时间模糊淡出模式（Audacious 示波器余晖拖尾）")
    ap.add_argument("--no-trail", dest="trail", action="store_false",
                    help="关闭余晖拖尾，每帧只显示当前轮廓")
    ap.set_defaults(trail=None)

    ap.add_argument("--decay", type=float, default=None,
                    help="余晖衰减系数 0.5~0.98（越大拖尾越长，默认 0.85；未给 --trail-hold 时生效）")
    ap.add_argument("--trail-hold", type=float, default=None,
                    help="拖尾淡入淡出保持时间（秒，0=关闭改用 --decay，默认 0）；余晖在该时长内逐渐淡出并模糊")
    ap.add_argument("--blur", type=float, default=None,
                    help="余晖高斯模糊半径（像素，默认 1.2），越大残影越扩散（配合 --trail）")

    args = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装并加入 PATH")

    # —— 预设为基线，命令行显式参数覆盖预设 ——
    preset = load_preset(args.preset)

    def pick(key, cli_val, default):
        if cli_val is not None:
            return cli_val
        if key in preset:
            return preset[key]
        return default

    window_ms = pick("window", args.window, 30.0)
    n_points = pick("points", args.points, 400)
    line_width = pick("line_width", args.line_width, 2)
    scale = pick("scale", args.scale, 1.0)
    show_peaks = bool(pick("show_peaks", args.show_peaks, False))
    peak_lines = bool(pick("peak_lines", args.peak_lines, False))
    peak_hold = bool(pick("peak_hold", args.peak_hold, False))
    peak_hold_sec = float(pick("peak_hold_sec", args.peak_hold_sec, 1.2))
    peak_ramp = float(pick("peak_ramp", args.peak_ramp, 0.0))
    grid = bool(pick("grid", args.grid, False))
    trail = bool(pick("trail", args.trail, False))
    decay = float(pick("decay", args.decay, 0.85))
    trail_hold = float(pick("trail_hold", args.trail_hold, 0.0))
    blur = float(pick("blur", args.blur, 1.2))
    font_path = pick("font", args.font, None)

    color = parse_color(pick("color", args.color, None), TRACE_COL)
    bg_color = parse_color(pick("bg_color", args.bg_color, None), BG_COLOR)

    if window_ms <= 0:
        raise SystemExit("--window 必须大于 0")
    if n_points < 8:
        raise SystemExit("--points 不能小于 8")
    if line_width < 1:
        raise SystemExit("--line-width 不能小于 1")
    if not (0.5 <= decay <= 0.98):
        raise SystemExit("--decay 必须在 0.5~0.98 之间")
    if peak_ramp < 0:
        raise SystemExit("--peak-ramp 不能小于 0")
    if peak_hold_sec <= 0:
        raise SystemExit("--peak-hold-sec 必须大于 0")
    if trail_hold < 0:
        raise SystemExit("--trail-hold 不能小于 0")
    if blur < 0:
        raise SystemExit("--blur 不能小于 0")

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = base + "_scope.mp4"

    print("[1/3] 读取音频", file=sys.stderr)
    data, sr = load_audio(args.input)
    dur = len(data) / sr
    print(f"      {sr} Hz, {data.shape[1]}ch, {dur:.2f}s", file=sys.stderr)

    print("[2/3] 计算时域波形", file=sys.stderr)
    waves, peaks, troughs, n_frames = compute_wave(
        data, sr, args.fps, window_ms, n_points)

    if peak_ramp > 1e-3:
        peaks, troughs = apply_ramp(peaks, troughs, args.fps, ramp_sec=peak_ramp)

    if peak_hold:
        peaks, troughs = apply_hold(peaks, troughs, args.fps,
                                    hold_sec=peak_hold_sec)

    cfg = dict(
        input=args.input, fps=args.fps, width=args.width, height=args.height,
        color=color, bg_color=bg_color, line_width=line_width,
        scale=scale, show_peaks=show_peaks, peak_lines=peak_lines,
        peak_hold=peak_hold, peak_hold_sec=peak_hold_sec,
        peak_ramp=peak_ramp, grid=grid, font=font_path,
        trail=trail, decay=decay, trail_hold=trail_hold, blur=blur,
    )

    print(f"[3/3] 渲染 {n_frames} 帧 @ {args.fps}fps → "
          f"{args.width}x{args.height}", file=sys.stderr)
    rc = render(waves, peaks, troughs, cfg, out_path)

    if rc != 0:
        raise SystemExit(f"ffmpeg 编码失败（退出码 {rc}）")
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

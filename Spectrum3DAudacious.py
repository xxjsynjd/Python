#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum3DAudacious.py —— Audacious 风格 3D 频谱（第八版）

新增:
  - --hue-offset：红色起始角度（0.375=左上角红）
  - --flip-freq：翻转频谱方向（低频放右边）
  - --hue-reverse：色环反向（反转色相环方向，不翻转频率轴）
  - --gradient：渐变类型 angular 角度(默认) / linear 线性 / radial 径向 /
    reflected 对称 / diamond 菱形
  - --grid：显示底盘网格线（默认关闭，不加即关闭）
  - 纵深透明度渐变蒙版：-p Point 点序号 -a Alpha 透明度(0~1) -l Location
    纵深位置(0~1) -c Center 相邻点插值中点(0~1)；可成组重复，按位置排序线性插值；
    --preset-type 可套用内置预设 white-to-black（近端白实→远端黑透），手动控制点优先
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
GRID_COL = (26, 30, 40)


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


def project(x, y, z, W, H, yaw, cfg):
    wx = x * cfg["x_span"]
    wy = (y - 0.5) * cfg["depth"]
    wz = z

    yaw_total = yaw + math.radians(cfg["yaw_offset"])
    ca = math.cos(yaw_total)
    sa = math.sin(yaw_total)
    rx = wx * ca - wy * sa
    ry = wx * sa + wy * ca
    rz = wz

    pitch = math.radians(cfg["pitch"])
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    screen_y = -rz * cp - ry * sp
    depth = ry * cp - rz * sp

    persp = 1.0 / (1.0 + depth * cfg["persp"])

    sx = W * 0.5 + rx * W * cfg["scale_x"] * persp
    sy = H * cfg["y_offset"] + screen_y * H * cfg["scale_y"] * persp

    return sx, sy, depth


def _grad_t(gtype, x, yn):
    """按 PS 渐变类型把平面坐标映射到渐变进度 t∈[0,1]。

    x ∈ [-1,1]（频段方向），yn ∈ [-1,1]（纵深方向，已居中归一）。
    """
    if gtype == "linear":
        # 线性渐变：沿频段方向从左到右
        return (x + 1.0) * 0.5
    if gtype == "radial":
        # 径向渐变：由中心向外圆扩散
        r = math.sqrt(x * x + yn * yn) / math.sqrt(2.0)
        return min(1.0, max(0.0, r))
    if gtype == "reflected":
        # 对称渐变：以中线为轴向两侧镜像
        return min(1.0, max(0.0, abs(x)))
    if gtype == "diamond":
        # 菱形渐变：由中心沿曼哈顿距离向外菱形扩散
        d = (abs(x) + abs(yn)) * 0.5
        return min(1.0, max(0.0, d))
    # angular（角度渐变，默认）：绕中心顺时针扫掠
    theta = math.atan2(yn, x)
    return theta / (2 * math.pi)


def gradient_color(gtype, x, y, hue_offset=0.5, hue_reverse=False):
    yn = (y - 0.5) * 2.0
    t = _grad_t(gtype, x, yn)
    if hue_reverse:
        t = -t
    hue = (t + hue_offset) % 1.0
    return hue_to_rgb(hue)


def build_mask(points):
    """组装纵深透明度蒙版控制点并按位置排序。

    points: 列表，每项 (idx, alpha, loc, center)。
    返回 [(loc, alpha, center), ...]，按 loc 升序；空列表表示不启用蒙版。
    """
    pts = []
    for _idx, alpha, loc, center in points:
        loc = min(1.0, max(0.0, loc))
        alpha = min(1.0, max(0.0, alpha))
        center = min(1.0, max(0.0, center))
        pts.append((loc, alpha, center))
    pts.sort(key=lambda x: x[0])
    return pts


# 内置透明度蒙版预设：每项为 (点序号, 透明度, 纵深位置, 插值中点)
# 白=不透明(1.0)，黑=全透明(0.0)；方向由预设名约定。
MASK_PRESETS = {
    "white-to-black": [(1, 1.0, 0.0, 0.5), (2, 0.0, 1.0, 0.5)],
}


def mask_alpha(t, pts):
    """按纵深位置 t∈[0,1] 采样蒙版透明度。

    相邻控制点之间以左点 center 作为插值中点（PS 色标中点语义），
    两段分别线性插值；pts 为空时返回 1.0（全不透明）。
    """
    if not pts:
        return 1.0
    t = min(1.0, max(0.0, t))
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        l1, a1, c1 = pts[i]
        l2, a2, _ = pts[i + 1]
        if l1 <= t <= l2:
            span = l2 - l1
            if span <= 1e-9:
                return a2
            mid = l1 + c1 * span
            if t <= mid:
                k = 0.5 * (t - l1) / (mid - l1) if mid > l1 else 1.0
            else:
                k = 0.5 + 0.5 * (t - mid) / (l2 - mid) if l2 > mid else 1.0
            k = min(1.0, max(0.0, k))
            return a1 + (a2 - a1) * k
    return pts[-1][1]


def blend(fg, alpha, bg=BG):
    """前景色按 alpha 与背景色混合，返回 RGB 元组。"""
    if alpha >= 1.0:
        return fg
    if alpha <= 0.0:
        return bg
    return tuple(int(bg[k] * (1.0 - alpha) + fg[k] * alpha)
                 for k in range(3))


def render(spec_db, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames, bands = spec_db.shape
    db_min, db_max = cfg["db_min"], cfg["db_max"]

    norm = np.clip((spec_db - db_min) / (db_max - db_min), 0, 1)
    norm = norm ** 0.7

    rows = cfg["rows"]

    if cfg.get("flip_freq", False):
        bar_xs = np.linspace(1.0, -1.0, bands)
    else:
        bar_xs = np.linspace(-1.0, 1.0, bands)

    bar_w = abs((bar_xs[1] - bar_xs[0]) * 0.42) if bands > 1 else 0.05

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

            # —— 底盘网格线（地面 z=0，随视角一同投影）——
            if cfg.get("grid", False):
                gx = np.linspace(-1.0, 1.0, 11)
                gy = np.linspace(0.0, 1.0, 9)
                for gxi in gx:
                    p0 = project(gxi, 0.0, 0.0, W, H, yaw, cfg)
                    p1 = project(gxi, 1.0, 0.0, W, H, yaw, cfg)
                    draw.line([p0[:2], p1[:2]], fill=GRID_COL, width=1)
                for gyi in gy:
                    p0 = project(-1.0, gyi, 0.0, W, H, yaw, cfg)
                    p1 = project(1.0, gyi, 0.0, W, H, yaw, cfg)
                    draw.line([p0[:2], p1[:2]], fill=GRID_COL, width=1)

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

                    col = gradient_color(cfg.get("gradient", "angular"),
                                         x_c, t_pos, cfg["hue_offset"],
                                         cfg.get("hue_reverse", False))

                    prims.append({
                        "depth": depth,
                        "t": t_pos,
                        "bottom": [p_bl[:2], p_br[:2], p_fr[:2], p_fl[:2]],
                        "top": [top_bl[:2], top_br[:2],
                                top_fr[:2], top_fl[:2]],
                        "col": col,
                    })

            prims.sort(key=lambda p: -p["depth"])

            mask_pts = cfg.get("mask", [])

            for p in prims:
                bot = p["bottom"]
                top = p["top"]
                col = p["col"]
                alpha = mask_alpha(p["t"], mask_pts)

                side_shades = [0.55, 0.78, 0.65, 0.5]
                for k in range(4):
                    k2 = (k + 1) % 4
                    face = [bot[k], bot[k2], top[k2], top[k]]
                    draw.polygon(face, fill=blend(shade(col, side_shades[k]), alpha))

                draw.polygon(top, fill=blend(shade(col, 1.18), alpha))

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


def main():
    ap = argparse.ArgumentParser(
        description="Audacious gl-spectrum 风格（第八版）")
    ap.add_argument("input", help="输入音频文件（mp3 / wav / flac / m4a …）")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件（默认按音频名加 _audacious.mp4）")
    ap.add_argument("--fps", type=int, default=60, help="视频帧率")
    ap.add_argument("--width", type=int, default=1280, help="画面宽度（px）")
    ap.add_argument("--height", type=int, default=720, help="画面高度（px）")

    ap.add_argument("--bars", type=int, default=28, help="横向频段柱条数量")
    ap.add_argument("--rows", type=int, default=36, help="纵深拖尾排数（前后层数）")
    ap.add_argument("--fft-size", type=int, default=4096, help="FFT 窗大小（点数，自动取 2 的幂）")
    ap.add_argument("--fmin", type=float, default=40.0, help="分析最低频率（Hz）")
    ap.add_argument("--fmax", type=float, default=14000.0, help="分析最高频率（Hz）")
    ap.add_argument("--db-min", type=float, default=-85.0, help="频谱归一化下限电平（dB）")
    ap.add_argument("--db-max", type=float, default=-5.0, help="频谱归一化上限电平（dB）")

    ap.add_argument("--hue-offset", type=float, default=0.5,
                    help="色相起始偏移（0.375=左上角红，0.5=正上方红）")
    ap.add_argument("--flip-freq", dest="flip_freq",
                    action="store_true", help="翻转频谱（低频放右边）")
    ap.add_argument("--hue-reverse", dest="hue_reverse",
                    action="store_true",
                    help="色环反向（反转色相环方向，不翻转频率轴）")

    ap.add_argument("--gradient", default="angular",
                    choices=["angular", "linear", "radial", "reflected", "diamond"],
                    help="PS 渐变类型：angular=角度扫掠(默认)，linear=线性，"
                         "radial=径向，reflected=对称，diamond=菱形")

    ap.add_argument("--grid", dest="grid", action="store_true",
                    help="显示底盘网格线")
    ap.set_defaults(grid=False)

    ap.add_argument("-p", "--mask-point", dest="mask_p", type=int,
                    action="append", default=[],
                    help="Point 蒙版控制点序号（对标 PS 渐变编辑器上方不透明度色标）；"
                         "与 -a/-l/-c 成组重复使用，"
                         "如 -p 1 -a 1.0 -l 0.0 -c 0.5 -p 2 -a 0.15 -l 1.0 -c 0.5")
    ap.add_argument("-a", "--mask-alpha", dest="mask_a", type=float,
                    action="append", default=[],
                    help="Alpha 该控制点透明度 0~1（0=全透明/黑，1=不透明/白）")
    ap.add_argument("-l", "--mask-loc", dest="mask_l", type=float,
                    action="append", default=[],
                    help="Location 控制点纵深位置 0~1（0=最近排，1=最远排）")
    ap.add_argument("-c", "--mask-center", dest="mask_c", type=float,
                    action="append", default=[],
                    help="Center 相邻两控制点之间插值中点 0~1（PS 色标间菱形中点，默认 0.5）")
    ap.add_argument("--preset-type", dest="preset_type", default="none",
                    choices=["none", "white-to-black"],
                    help="内置透明度蒙版预设：white-to-black=近端白(不透明)→远端黑(全透明)；"
                         "none=不套用(默认)。手动 -p/-a/-l/-c 优先于预设")

    ap.add_argument("--osc-period", type=float, default=15.0,
                    help="摆动周期（秒），越大越慢")
    ap.add_argument("--osc-angle", type=float, default=120.0,
                    help="摆动幅度（度）")

    ap.add_argument("--pitch", type=float, default=35.0,
                    help="俯视角（度）。0=侧视，90=纯俯视")
    ap.add_argument("--yaw-offset", type=float, default=20.0,
                    help="初始偏航角（度）")

    ap.add_argument("--persp", type=float, default=0.10, help="透视强度（越大近大远小越明显）")
    ap.add_argument("--x-span", type=float, default=1.0, help="横向展开跨度系数")
    ap.add_argument("--depth", type=float, default=1.6, help="纵深深度系数")
    ap.add_argument("--scale-x", type=float, default=0.42, help="横向屏幕缩放系数")
    ap.add_argument("--scale-y", type=float, default=0.32, help="纵向屏幕缩放系数")
    ap.add_argument("--y-offset", type=float, default=0.68, help="画面纵向偏移（0~1，越低越靠下）")

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")

    mp, ma, ml, mc = a.mask_p, a.mask_a, a.mask_l, a.mask_c
    n = len(mp)
    if not (len(ma) == n and len(ml) == n):
        raise SystemExit("蒙版参数需按 -p/-a/-l 成组配齐（-c 可省略，默认 0.5）")
    if n and len(mc) < n:
        mc = mc + [0.5] * (n - len(mc))
    manual_pts = build_mask(list(zip(mp, ma, ml, mc[:n])))
    if manual_pts:
        mask_pts = manual_pts
    elif a.preset_type != "none":
        mask_pts = build_mask(MASK_PRESETS[a.preset_type])
    else:
        mask_pts = []
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
        hue_offset=a.hue_offset, flip_freq=a.flip_freq,
        hue_reverse=a.hue_reverse, gradient=a.gradient, grid=a.grid,
        mask=mask_pts,
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

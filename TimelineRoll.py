#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimelineRoll.py —— 滚动时间轴 + 波形（剪映风格，全套）

功能:
  - 游标固定，波形 + 刻度一起往左滑
  - 一屏显示多少秒由 --window 控制
  - 两端渐隐、已播/未播分色
  - --auto-vocals 用 Demucs 分离人声
  - --beats 叠加 BPM 小节线 + 拍号

用法:
    python3 TimelineRoll.py 歌曲.mp3 -o out.mp4
    python3 TimelineRoll.py 歌曲.mp3 --window 8 --beats
    python3 TimelineRoll.py 歌曲.mp3 --auto-vocals --beats
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

BG = (8, 9, 14)
PANEL_BG = (16, 17, 21)
PANEL_BORDER = (40, 220, 120, 200)
WAVE_PAST = (255, 120, 90, 235)
WAVE_FUTURE = (90, 200, 255, 230)
RAIL = (55, 60, 70)
TICK_MAJOR = (180, 185, 195, 235)
TICK_MINOR = (95, 100, 110, 175)
TEXT_COL = (210, 215, 225)
CURSOR_COL = (255, 90, 70)
BAR_GREEN = (40, 220, 120)
BEAT_GREEN = (40, 160, 90)
BPM_COL = (255, 220, 90)


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
# Demucs 分离
# ==========================================================================
def find_or_create_vocals(audio_path, cache_dir=".vocals_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_path = os.path.join(cache_dir, base + "_vocals.wav")

    if os.path.isfile(cache_path):
        print(f"      缓存命中: {cache_path}", file=sys.stderr)
        return cache_path

    demucs_bin = shutil.which("demucs")
    if not demucs_bin:
        print("      未找到 demucs，跳过分离", file=sys.stderr)
        return None

    print("      正在分离人声...", file=sys.stderr)
    tmp_out = os.path.join(cache_dir, "_tmp_" + base)
    os.makedirs(tmp_out, exist_ok=True)

    cmd = [demucs_bin, "--two-stems", "vocals",
           "-o", tmp_out, "--float32",
           "-n", "htdemucs", "-d", "cpu", audio_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=7200)
    except subprocess.TimeoutExpired:
        print("      分离超时", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    if p.returncode != 0:
        print(f"      分离失败: {p.stderr.strip()[-300:]}",
              file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    vocals_src = None
    inst_src = None
    for root, _, files in os.walk(tmp_out):
        for fn in files:
            if fn == "vocals.wav":
                vocals_src = os.path.join(root, fn)
            elif fn == "no_vocals.wav":
                inst_src = os.path.join(root, fn)

    if not vocals_src:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    shutil.move(vocals_src, cache_path)
    if inst_src:
        try:
            shutil.move(inst_src,
                        os.path.join(cache_dir,
                                     base + "_instrumental.wav"))
        except Exception:
            pass
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"      人声已缓存: {cache_path}", file=sys.stderr)
    return cache_path


# ==========================================================================
# 波形包络
# ==========================================================================
def build_waveform_envelope(mono, sr, points_per_sec):
    n = len(mono)
    if n == 0:
        return np.zeros(2, dtype=np.float32)

    dur = n / sr
    n_points = max(2, int(dur * points_per_sec) + 2)
    idx = np.linspace(0, n, n_points + 1).astype(np.int64)

    peaks = np.zeros(n_points, dtype=np.float32)
    for i in range(n_points):
        a = idx[i]
        b = max(idx[i] + 1, idx[i + 1])
        seg = mono[a:b]
        if len(seg):
            peaks[i] = np.max(np.abs(seg))

    k = np.ones(3, dtype=np.float32) / 3.0
    peaks = np.convolve(peaks, k, mode="same")
    m = peaks.max()
    if m > 1e-9:
        peaks = peaks / m
    return np.clip(peaks, 0, 1)


# ==========================================================================
# BPM
# ==========================================================================
def detect_bpm(mono, sr, fps, bpm_min=60.0, bpm_max=200.0):
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(len(mono) / hop)))
    energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a = int(i * hop)
        b = int((i + 1) * hop)
        seg = mono[a:b]
        if len(seg):
            energy[i] = np.sum(seg ** 2)
    onset = np.diff(energy, prepend=energy[0])
    onset = np.maximum(onset, 0)
    onset = onset - onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag_min = max(1, int(fps * 60.0 / bpm_max))
    lag_max = min(len(ac) - 1, int(fps * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return 120.0
    best_lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1]))
    bpm = 60.0 * fps / best_lag
    while bpm < bpm_min:
        bpm *= 2
    while bpm > bpm_max:
        bpm /= 2
    return float(bpm)


# ==========================================================================
# 工具
# ==========================================================================
def format_time(sec):
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = sec - m * 60
    if m > 0:
        return f"{m}:{s:05.2f}"
    return f"{s:.2f}s"


def pick_step(px_per_sec, min_px=70):
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
                  60, 120, 300, 600, 1200, 1800]
    for c in candidates:
        if c * px_per_sec >= min_px:
            return c
    return candidates[-1]


# ==========================================================================
# 绘制
# ==========================================================================
def draw_timeline_wave(canvas, cx, cy, w, h,
                       cur_sec, total_sec,
                       wave_env, wave_points_per_sec,
                       window_sec=10.0,
                       wave_ratio=0.62,
                       bpm=0.0, beat_offset=0.0,
                       beats_per_bar=4, beat_unit=4):
    draw = ImageDraw.Draw(canvas, "RGBA")

    x0 = cx - w // 2
    x1 = cx + w // 2
    y0 = cy - h // 2
    y1 = cy + h // 2

    radius = max(4, h // 12)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                           fill=PANEL_BG,
                           outline=PANEL_BORDER, width=2)

    wave_h = int(h * wave_ratio)
    tl_h = h - wave_h
    wave_y0 = y0
    wave_y1 = y0 + wave_h
    tl_y0 = wave_y1
    tl_y1 = y1

    px_per_sec = w / max(window_sec, 0.1)
    cursor_x = cx

    half_sec = window_sec / 2.0
    t_left = max(0.0, cur_sec - half_sec)
    t_right = cur_sec + half_sec

    # ---------- 波形 ----------
    if wave_env is not None and len(wave_env) > 2:
        mid_y = (wave_y0 + wave_y1) // 2
        amp = (wave_y1 - wave_y0) * 0.42
        env_n = len(wave_env)

        cols = x1 - x0
        xs = np.arange(x0, x1)
        ts = t_left + (xs - x0) / px_per_sec

        idx_f = ts * wave_points_per_sec
        idx_i = np.clip(idx_f.astype(np.int64), 0, env_n - 1)
        idx_j = np.clip(idx_i + 1, 0, env_n - 1)
        frac = np.clip(idx_f - idx_i, 0, 1)
        vals = wave_env[idx_i] * (1 - frac) + wave_env[idx_j] * frac

        ys_top = (mid_y - vals * amp).astype(np.int32)
        ys_bot = (mid_y + vals * amp).astype(np.int32)

        for px, yt, yb in zip(xs, ys_top, ys_bot):
            if yt == yb:
                continue
            col = WAVE_PAST if px <= cursor_x else WAVE_FUTURE
            draw.line([(int(px), int(yt)), (int(px), int(yb))],
                      fill=col, width=1)

        draw.line([(x0 + 2, mid_y), (x1 - 2, mid_y)],
                  fill=RAIL, width=1)

        # 左右渐隐
        fade_w = max(8, w // 14)
        for k in range(fade_w):
            a = int(210 * (1 - k / fade_w))
            draw.line([(x0 + k, wave_y0 + 2), (x0 + k, wave_y1 - 1)],
                      fill=(PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], a))
            draw.line([(x1 - 1 - k, wave_y0 + 2),
                       (x1 - 1 - k, wave_y1 - 1)],
                      fill=(PANEL_BG[0], PANEL_BG[1], PANEL_BG[2], a))

    # ---------- 刻度 ----------
    step = pick_step(px_per_sec, min_px=70)
    sub_step = step / 5.0

    t = math.floor(t_left / step) * step
    while t <= t_right:
        x = cursor_x + (t - cur_sec) * px_per_sec
        if x0 + 6 <= x <= x1 - 6:
            draw.line([(x, tl_y1 - int(tl_h * 0.55)), (x, tl_y1 - 3)],
                      fill=TICK_MAJOR, width=2)
            lab = format_time(t)
            bbox = draw.textbbox((0, 0), lab)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw / 2, tl_y0 + 3), lab, fill=TEXT_COL)
        for k in range(1, 5):
            ts2 = t + sub_step * k
            if ts2 > t_right:
                break
            xs = cursor_x + (ts2 - cur_sec) * px_per_sec
            if x0 + 3 <= xs <= x1 - 3:
                draw.line([(xs, tl_y1 - int(tl_h * 0.28)),
                           (xs, tl_y1 - 3)],
                          fill=TICK_MINOR, width=1)
        t += step

    # ---------- 底部进度条 ----------
    bar_h = max(2, tl_h // 16)
    bar_y = tl_y1 - bar_h - 1
    draw.rectangle([x0 + 3, bar_y, x1 - 3, bar_y + bar_h], fill=RAIL)
    if cur_sec > 0:
        played = min(cursor_x, x1 - 3)
        if played > x0 + 3:
            draw.rectangle([x0 + 3, bar_y, played, bar_y + bar_h],
                           fill=CURSOR_COL)

    # ---------- 小节线 ----------
    if bpm > 0:
        sec_per_beat = 60.0 / bpm
        sec_per_bar = sec_per_beat * beats_per_bar

        first_beat = max(0, int(math.ceil(
            (t_left - beat_offset) / sec_per_beat)))
        t = beat_offset + first_beat * sec_per_beat
        beat_i = first_beat
        while t <= t_right + 1e-6:
            x = cursor_x + (t - cur_sec) * px_per_sec
            if x0 + 1 <= x <= x1 - 1:
                if beat_i % beats_per_bar != 0:
                    draw.line([(x, wave_y0 + 2), (x, wave_y1 - 1)],
                              fill=BEAT_GREEN + (90,), width=1)
            t += sec_per_beat
            beat_i += 1

        first_bar = max(0, int(math.ceil(
            (t_left - beat_offset) / sec_per_bar)))
        t = beat_offset + first_bar * sec_per_bar
        while t <= t_right + 1e-6:
            x = cursor_x + (t - cur_sec) * px_per_sec
            if x0 + 1 <= x <= x1 - 1:
                draw.line([(x, wave_y0 + 2), (x, wave_y1 - 1)],
                          fill=BAR_GREEN + (230,), width=2)
            t += sec_per_bar

    # ---------- 游标 ----------
    draw.line([(cursor_x, wave_y0 + 2), (cursor_x, tl_y1 - 2)],
              fill=CURSOR_COL, width=2)
    tri = 6
    draw.polygon([
        (cursor_x - tri, wave_y0 - tri),
        (cursor_x + tri, wave_y0 - tri),
        (cursor_x, wave_y0 + 2),
    ], fill=CURSOR_COL)

    # ---------- 左上角拍号 ----------
    if bpm > 0:
        sx, sy = x0 + 12, y0 - 72
        sw, sh = 60, 60
        draw.rounded_rectangle([sx, sy, sx + sw, sy + sh],
                               radius=8,
                               fill=(14, 18, 16, 220),
                               outline=BAR_GREEN + (200,), width=2)
        txt = f"{beats_per_bar}/{beat_unit}"
        bbox = draw.textbbox((0, 0), txt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((sx + (sw - tw) / 2 - bbox[0],
                   sy + (sh - th) / 2 - bbox[1]),
                  txt, fill=BAR_GREEN + (255,))

        draw.text((sx + sw + 12, sy + 6), f"{int(bpm)} BPM",
                  fill=BPM_COL)
        draw.text((sx + sw + 12, sy + 32),
                  f"{format_time(cur_sec)} / {format_time(total_sec)}",
                  fill=TEXT_COL)


# ==========================================================================
# 渲染
# ==========================================================================
def render(wave_env, points_per_sec, total_sec, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n_frames = max(1, int(math.ceil(total_sec * fps)))

    tl_w = int(W * cfg["width_ratio"])
    tl_h = cfg["height_px"]
    tl_cx = W // 2
    tl_cy = H - tl_h // 2 - cfg["bottom_margin"]

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

            draw_timeline_wave(
                canvas,
                cx=tl_cx, cy=tl_cy,
                w=tl_w, h=tl_h,
                cur_sec=cur_sec,
                total_sec=total_sec,
                wave_env=wave_env,
                wave_points_per_sec=points_per_sec,
                window_sec=cfg["window_sec"],
                wave_ratio=cfg["wave_ratio"],
                bpm=cfg["bpm"],
                beat_offset=cfg["beat_offset"],
                beats_per_bar=cfg["beats_per_bar"],
                beat_unit=cfg["beat_unit"],
            )

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
    ap = argparse.ArgumentParser(
        description="滚动时间轴 + 波形（剪映风格，全套）")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--window", type=float, default=10.0,
                    help="时间轴一屏显示多少秒")
    ap.add_argument("--height-px", type=int, default=160,
                    help="时间轴区域高度（像素）")
    ap.add_argument("--width-ratio", type=float, default=0.92)
    ap.add_argument("--bottom-margin", type=int, default=40)
    ap.add_argument("--wave-ratio", type=float, default=0.62)
    ap.add_argument("--wave-res", type=float, default=300.0,
                    help="波形包络点/秒")

    ap.add_argument("--auto-vocals", dest="auto_vocals",
                    action="store_true", help="Demucs 分离人声")
    ap.add_argument("--cache-dir", default=".vocals_cache")
    ap.add_argument("--vocals", default=None,
                    help="手动指定人声轨")

    ap.add_argument("--beats", dest="beats", action="store_true",
                    help="叠加 BPM 小节线 + 拍号")
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="手动 BPM，0=自动")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--beat-unit", type=int, default=4)
    ap.add_argument("--beat-offset", type=float, default=0.0)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_timeline.mp4"

    print("[1/4] 准备音频", file=sys.stderr)
    src_path = a.input
    if a.vocals and os.path.isfile(a.vocals):
        src_path = a.vocals
        print(f"      手动人声轨: {src_path}", file=sys.stderr)
    elif a.auto_vocals:
        vp = find_or_create_vocals(a.input, a.cache_dir)
        if vp:
            src_path = vp
            print(f"      使用人声: {vp}", file=sys.stderr)
        else:
            print("      分离失败，退回原曲", file=sys.stderr)

    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s",
          file=sys.stderr)

    # 波形用原曲（保真），BPM 用分离后（更准）
    mono_for_wave = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    print("[2/4] 生成波形包络", file=sys.stderr)
    wave_env = build_waveform_envelope(mono_for_wave, sr, a.wave_res)
    print(f"      包络点数: {len(wave_env)}", file=sys.stderr)

    total_sec = len(data) / sr

    print("[3/4] BPM 检测", file=sys.stderr)
    if a.bpm > 0:
        bpm = a.bpm
    else:
        bpm = detect_bpm(mono_for_wave, sr, a.fps) if a.beats else 0.0
    print(f"      BPM = {bpm:.1f}", file=sys.stderr)

    cfg = dict(
        input=a.input, fps=a.fps,
        width=a.width, height=a.height,
        window_sec=a.window,
        height_px=a.height_px,
        width_ratio=a.width_ratio,
        bottom_margin=a.bottom_margin,
        wave_ratio=a.wave_ratio,
        bpm=bpm, beats_per_bar=a.beats_per_bar,
        beat_unit=a.beat_unit, beat_offset=a.beat_offset,
    )

    print(f"[4/4] 渲染 {int(total_sec * a.fps)} 帧 → {out_path}",
          file=sys.stderr)
    render(wave_env, a.wave_res, total_sec, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

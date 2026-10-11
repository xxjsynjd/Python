#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PitchMeter1.py —— 动态简谱音符计（Demucs 命令行分离 + 音乐分析面板）

功能:
  - 全自动 Demucs 人声分离（--auto-vocals），用命令行调用，不 import demucs
  - 带缓存，第一次分离，第二次直接读 .vocals_cache/
  - BPM 自动检测 + 绿色小节线 / 淡绿拍线
  - 左上角音乐分析面板：拍号 / 小节 / 拍位 / 上下半拍 / 滑音 / 连音 / NEXT IN / HOLD
  - 右上角无人声时显示休止符 0
  - 笔状态：有人声红线，无人声白点

用法:
    python3 PitchMeter1.py "歌.mp3" --auto-vocals
    python3 PitchMeter1.py "歌.mp3" --auto-vocals --width 720 --height 1280 --fps 30
    python3 PitchMeter1.py "歌.mp3" --vocals path/to/vocals.wav
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
BG = (14, 15, 20)
GRID = (40, 44, 54)
LINE_COL = (255, 70, 70, 255)
DOT_COL = (230, 230, 230, 200)
PEN_GLOW = (255, 90, 90, 110)
TEXT_COL = (200, 210, 220)
BAR_GREEN = (40, 220, 120)
BEAT_GREEN = (40, 160, 90)

PANEL_BG = (14, 18, 22, 225)
PANEL_BORDER = (40, 220, 120, 200)
HALF_UP_COL = (90, 220, 255)
HALF_DN_COL = (255, 180, 90)
GLISS_COL = (255, 90, 200)
LEGATO_COL = (120, 220, 255)
NEXT_COL = (255, 220, 90)
HOLD_COL = (120, 255, 180)
DIM_COL = (90, 95, 105)


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
# Demucs 分离（命令行版，不 import demucs）
# ==========================================================================
def find_or_create_vocals(audio_path, cache_dir=".vocals_cache"):
    """
    用 Demucs 命令行分离人声，带缓存。
    不 import demucs，直接用 subprocess 调。
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_path = os.path.join(cache_dir, base + "_vocals.wav")

    if os.path.isfile(cache_path):
        print(f"      缓存命中: {cache_path}", file=sys.stderr)
        return cache_path

    # 找 demucs 命令：优先 `demucs`，没有就用 `python -m demucs.separate`
    demucs_bin = shutil.which("demucs")
    if demucs_bin:
        demucs_cmd = [demucs_bin]
        print(f"      使用命令: {demucs_bin}", file=sys.stderr)
    else:
        demucs_cmd = [sys.executable, "-m", "demucs.separate"]
        print(f"      使用命令: {' '.join(demucs_cmd)}", file=sys.stderr)

    print("      正在用 Demucs 分离人声（第一次会下载模型）...",
          file=sys.stderr)
    tmp_out = os.path.join(cache_dir, "_tmp_" + base)
    os.makedirs(tmp_out, exist_ok=True)

    cmd = demucs_cmd + [
        "--two-stems", "vocals",
        "-o", tmp_out,
        "--float32",
        "-n", "htdemucs",
        "-d", "cpu",
        audio_path,
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=7200)
    except subprocess.TimeoutExpired:
        print("      分离超时", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    if p.returncode != 0:
        print(f"      分离失败（退出码 {p.returncode}）", file=sys.stderr)
        print(f"      stderr: {p.stderr.strip()[-500:]}", file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    # 扫描输出目录，找 vocals.wav
    vocals_src = None
    inst_src = None
    for root, _, files in os.walk(tmp_out):
        for fn in files:
            full = os.path.join(root, fn)
            if fn == "vocals.wav":
                vocals_src = full
            elif fn == "no_vocals.wav":
                inst_src = full

    if not vocals_src:
        print("      没找到 vocals.wav 输出", file=sys.stderr)
        for root, _, files in os.walk(tmp_out):
            for fn in files:
                print(f"        找到: {os.path.join(root, fn)}",
                      file=sys.stderr)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

    shutil.move(vocals_src, cache_path)
    if inst_src and os.path.isfile(inst_src):
        try:
            acc_path = os.path.join(cache_dir, base + "_instrumental.wav")
            shutil.move(inst_src, acc_path)
            print(f"      伴奏已缓存: {acc_path}", file=sys.stderr)
        except Exception:
            pass

    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"      人声已缓存: {cache_path}", file=sys.stderr)
    return cache_path


# ==========================================================================
# BPM 检测
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
# YIN 音高检测
# ==========================================================================
def detect_pitch_from_vocals(mono, sr, fps,
                             fmin=80.0, fmax=1000.0,
                             frame_ms=60.0, threshold=0.08,
                             energy_gate_db=-50.0):
    n = len(mono)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    win = max(128, int(sr * frame_ms / 1000.0))

    tau_min = max(2, int(sr / fmax))
    tau_max = min(win - 1, int(sr / fmin))

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    gate_lin = 10.0 ** (energy_gate_db / 20.0)

    for i in range(n_frames):
        c = int((i + 0.5) * hop)
        a = c - win // 2
        b = a + win
        if a < 0:
            seg = np.zeros(win, dtype=np.float32)
            seg[-a:] = mono[:b]
        elif b > n:
            seg = np.zeros(win, dtype=np.float32)
            seg[:n - a] = mono[a:]
        else:
            seg = mono[a:b].astype(np.float32)

        rms = np.sqrt(np.mean(seg ** 2))
        if rms < gate_lin:
            continue

        seg = seg - seg.mean()

        d = np.zeros(tau_max + 1, dtype=np.float32)
        for tau in range(1, tau_max + 1):
            diff = seg[:-tau] - seg[tau:]
            d[tau] = np.sum(diff * diff)

        cm = np.ones(tau_max + 1, dtype=np.float32)
        running = 0.0
        for tau in range(1, tau_max + 1):
            running += d[tau]
            cm[tau] = d[tau] * tau / max(running, 1e-9)

        tau_est = -1
        for tau in range(tau_min, tau_max):
            if cm[tau] < threshold:
                while tau + 1 < tau_max and cm[tau + 1] < cm[tau]:
                    tau += 1
                tau_est = tau
                break

        if tau_est > 0:
            if 1 <= tau_est < tau_max - 1:
                x0, x1, x2 = cm[tau_est - 1], cm[tau_est], cm[tau_est + 1]
                denom = 2 * (2 * x1 - x2 - x0)
                if abs(denom) > 1e-9:
                    shift = (x2 - x0) / denom
                    tau_est = tau_est + shift
            f = sr / tau_est
            if fmin <= f <= fmax:
                f0[i] = f
                voiced[i] = True

        if i % 200 == 0:
            print(f"\r  音高检测 {i * 100 // n_frames:3d}%",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return f0, voiced


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-6) / 440.0)


def midi_to_name(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#",
             "G", "G#", "A", "A#", "B"]
    mi = int(round(m))
    octave = mi // 12 - 1
    return f"{names[mi % 12]}{octave}"


def smooth_pitch(f0, voiced, fps, hold_ms=150.0):
    n = len(f0)
    hold = max(1, int(fps * hold_ms / 1000.0))
    f0s = f0.copy()
    vs = voiced.copy()
    last = 0.0
    last_i = -10**9
    for i in range(n):
        if voiced[i] and f0[i] > 0:
            last = f0[i]
            last_i = i
        else:
            if i - last_i <= hold:
                f0s[i] = last
                vs[i] = True
    return f0s, vs


# ==========================================================================
# 音乐特征：滑音 / 连音
# ==========================================================================
def analyze_music_features(midi_seq, voiced_seq, fps,
                           gliss_semitones=2.0,
                           gliss_win_sec=0.12,
                           legato_min_sec=0.35,
                           legato_max_jitter=0.35):
    n = len(midi_seq)
    gliss_flags = np.zeros(n, dtype=bool)
    legato_flags = np.zeros(n, dtype=bool)

    if n == 0:
        return gliss_flags, legato_flags

    gliss_win = max(2, int(gliss_win_sec * fps))
    legato_min = max(2, int(legato_min_sec * fps))

    for i in range(gliss_win, n):
        if not (voiced_seq[i] and voiced_seq[i - gliss_win]):
            continue
        delta = midi_seq[i] - midi_seq[i - gliss_win]
        if abs(delta) >= gliss_semitones:
            gliss_flags[i - gliss_win + 1:i + 1] = True

    i = 0
    while i < n:
        if not voiced_seq[i]:
            i += 1
            continue
        j = i
        while j < n and voiced_seq[j]:
            j += 1
        seg_len = j - i
        if seg_len >= legato_min:
            seg = midi_seq[i:j]
            jitter = float(np.std(seg))
            if jitter <= legato_max_jitter:
                legato_flags[i:j] = True
        i = j

    legato_flags &= ~gliss_flags
    return gliss_flags, legato_flags


# ==========================================================================
# 时序：下一个音符 / 当前音符剩余
# ==========================================================================
def compute_timing(voiced_seq, fps):
    n = len(voiced_seq)
    next_in = np.zeros(n, dtype=np.float32)
    hold_left = np.zeros(n, dtype=np.float32)

    next_voiced_idx = -1
    for i in range(n - 1, -1, -1):
        if voiced_seq[i]:
            next_voiced_idx = i
        if next_voiced_idx >= 0:
            next_in[i] = (next_voiced_idx - i) / float(fps)
        else:
            next_in[i] = -1.0

    i = 0
    while i < n:
        if not voiced_seq[i]:
            i += 1
            continue
        j = i
        while j < n and voiced_seq[j]:
            j += 1
        end_idx = j - 1
        for k in range(i, j):
            hold_left[k] = (end_idx - k) / float(fps)
        i = j

    return next_in, hold_left


# ==========================================================================
# 绘制
# ==========================================================================
def draw_pitch_meter(canvas, x, y, w, h,
                     midi_seq, voiced_seq, cur_idx,
                     window_frames,
                     fmin=80.0, fmax=1000.0,
                     line_width=5, dot_size=5,
                     bpm=0.0, beat_offset=0.0,
                     beats_per_bar=4, fps=60):
    draw = ImageDraw.Draw(canvas, "RGBA")

    radius = max(4, h // 16)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=BG)

    midi_lo = hz_to_midi(fmin)
    midi_hi = hz_to_midi(fmax)
    if midi_hi - midi_lo < 1:
        midi_hi = midi_lo + 12

    px_per_frame = w / max(window_frames, 1)

    for m in range(int(midi_lo), int(midi_hi) + 1):
        if m % 12 == 0:
            yy = y + h - (m - midi_lo) / (midi_hi - midi_lo) * h
            draw.line([(x + 2, yy), (x + w - 2, yy)], fill=GRID, width=1)
            draw.text((x + 4, yy + 2), midi_to_name(m), fill=TEXT_COL)

    if bpm > 0:
        sec_per_beat = 60.0 / bpm
        sec_per_bar = sec_per_beat * beats_per_bar
        window_sec = window_frames / float(fps)
        t_now = cur_idx / float(fps)
        t_left = t_now - window_sec

        first_beat = max(0, int(math.ceil(
            (t_left - beat_offset) / sec_per_beat)))
        t = beat_offset + first_beat * sec_per_beat
        beat_i = first_beat
        while t <= t_now + 1e-6:
            px = x + w - (t_now - t) * (w / window_sec)
            if x + 1 <= px <= x + w - 1:
                if beat_i % beats_per_bar != 0:
                    draw.line([(px, y + 2), (px, y + h - 2)],
                              fill=BEAT_GREEN + (90,), width=1)
            t += sec_per_beat
            beat_i += 1

        first_bar = max(0, int(math.ceil(
            (t_left - beat_offset) / sec_per_bar)))
        t = beat_offset + first_bar * sec_per_bar
        while t <= t_now + 1e-6:
            px = x + w - (t_now - t) * (w / window_sec)
            if x + 1 <= px <= x + w - 1:
                draw.line([(px, y + 2), (px, y + h - 2)],
                          fill=BAR_GREEN + (230,), width=2)
            t += sec_per_bar

    start = max(0, cur_idx - window_frames)
    n_vis = cur_idx - start + 1
    if n_vis <= 0:
        return

    frames = np.arange(start, cur_idx + 1)
    xs = x + w - (cur_idx - frames) * px_per_frame
    ys = y + h - (midi_seq[start:cur_idx + 1] - midi_lo) / \
         (midi_hi - midi_lo) * h
    vs = voiced_seq[start:cur_idx + 1]

    for k in range(1, n_vis):
        if vs[k] and vs[k - 1]:
            draw.line([(xs[k - 1], ys[k - 1]), (xs[k], ys[k])],
                      fill=LINE_COL, width=line_width)
            r = line_width // 2
            for xx, yy in ((xs[k - 1], ys[k - 1]), (xs[k], ys[k])):
                draw.ellipse([xx - r, yy - r, xx + r, yy + r],
                             fill=LINE_COL)

    r_dot = dot_size
    for k in range(n_vis):
        if not vs[k]:
            px, py = xs[k], ys[k]
            if x + 2 <= px <= x + w - 2:
                draw.ellipse([px - r_dot, py - r_dot,
                              px + r_dot, py + r_dot],
                             fill=DOT_COL)

    if 0 <= cur_idx < len(midi_seq):
        pen_x = xs[-1]
        pen_y = ys[-1]
        if voiced_seq[cur_idx]:
            glow_r = dot_size * 3
            draw.ellipse([pen_x - glow_r, pen_y - glow_r,
                          pen_x + glow_r, pen_y + glow_r],
                         fill=PEN_GLOW)
            draw.ellipse([pen_x - dot_size, pen_y - dot_size,
                          pen_x + dot_size, pen_y + dot_size],
                         fill=LINE_COL)
        else:
            draw.ellipse([pen_x - dot_size, pen_y - dot_size,
                          pen_x + dot_size, pen_y + dot_size],
                         fill=DOT_COL)
        draw.line([(pen_x, y + 4), (pen_x, y + h - 4)],
                  fill=(255, 255, 255, 30), width=1)

    if 0 <= cur_idx < len(midi_seq) and voiced_seq[cur_idx]:
        name = midi_to_name(midi_seq[cur_idx])
        draw.text((x + w - 42, y + 6), name, fill=LINE_COL)


def _bar_number_and_beat(cur_sec, bpm, beat_offset, beats_per_bar):
    if bpm <= 0:
        return 0, 0, 0.0
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * beats_per_bar
    t = max(0.0, cur_sec - beat_offset)
    bar_idx = int(t // sec_per_bar)
    in_bar = t - bar_idx * sec_per_bar
    beat_idx = int(in_bar // sec_per_beat)
    in_beat = (in_bar - beat_idx * sec_per_beat) / sec_per_beat
    return bar_idx + 1, beat_idx, in_beat


def draw_music_panel(canvas, x, y,
                     beats, unit,
                     bar_num, beat_idx, in_beat,
                     is_gliss, is_legato,
                     has_voice,
                     next_in_sec, hold_left_sec):
    draw = ImageDraw.Draw(canvas, "RGBA")

    W_, H_ = 260, 300
    draw.rounded_rectangle([x, y, x + W_, y + H_],
                           radius=10, fill=PANEL_BG,
                           outline=PANEL_BORDER, width=2)

    draw.text((x + 12, y + 10), f"{beats}/{unit}",
              fill=BAR_GREEN + (255,))
    draw.line([(x + 8, y + 38), (x + W_ - 8, y + 38)],
              fill=(60, 65, 75, 200), width=1)

    draw.text((x + 12, y + 48), "BAR", fill=DIM_COL)
    draw.text((x + 54, y + 48), f"{bar_num:04d}", fill=TEXT_COL + (255,))

    draw.text((x + 12, y + 78), "BEAT", fill=DIM_COL)
    bx = x + 54
    for k in range(beats):
        active = (k == beat_idx)
        col = (255, 255, 255) if active else DIM_COL
        draw.text((bx + k * 22, y + 78), str(k + 1), fill=col)

    up_active = in_beat < 0.5
    up_col = HALF_UP_COL + (255,) if up_active else DIM_COL
    dn_col = HALF_DN_COL + (255,) if not up_active else DIM_COL
    draw.rectangle([x + 12, y + 112, x + 22, y + 118], fill=up_col)
    draw.text((x + 28, y + 106), "上半拍", fill=up_col)
    draw.rectangle([x + 130, y + 112, x + 140, y + 118], fill=dn_col)
    draw.text((x + 146, y + 106), "下半拍", fill=dn_col)

    bx0, by0 = x + 12, y + 140
    bx1, by1 = x + W_ - 12, y + 148
    draw.rectangle([bx0, by0, bx1, by1], fill=(40, 44, 52))
    mid = bx0 + (bx1 - bx0) // 2
    draw.line([(mid, by0), (mid, by1)], fill=(70, 75, 85), width=1)
    cursor_x = bx0 + int((bx1 - bx0) * max(0.0, min(1.0, in_beat)))
    draw.rectangle([bx0, by0, cursor_x, by1],
                   fill=(HALF_UP_COL if up_active else HALF_DN_COL) + (255,))

    ty = y + 166
    g_col = GLISS_COL + (255,) if is_gliss else DIM_COL
    draw.text((x + 12, ty), "滑音", fill=g_col)
    draw.line([(x + 12, ty + 22), (x + 60, ty + 22)],
              fill=g_col, width=2)
    draw.polygon([(x + 60, ty + 22), (x + 54, ty + 18), (x + 54, ty + 26)],
                 fill=g_col)

    l_col = LEGATO_COL + (255,) if is_legato else DIM_COL
    draw.text((x + 130, ty), "连音", fill=l_col)
    draw.arc([x + 128, ty + 14, x + 178, ty + 30], start=200, end=340,
             fill=l_col, width=2)

    ty2 = y + 204
    if has_voice:
        next_txt = "—"
        next_col = DIM_COL
    elif next_in_sec < 0:
        next_txt = "END"
        next_col = DIM_COL
    else:
        next_txt = f"{next_in_sec:4.1f}s"
        next_col = NEXT_COL + (255,)

    draw.text((x + 12, ty2), "NEXT IN", fill=DIM_COL)
    draw.text((x + 130, ty2), next_txt, fill=next_col)

    ty3 = y + 234
    if has_voice:
        hold_txt = f"{hold_left_sec:4.1f}s"
        hold_col = HOLD_COL + (255,)
    else:
        hold_txt = "—"
        hold_col = DIM_COL

    draw.text((x + 12, ty3), "HOLD", fill=DIM_COL)
    draw.text((x + 130, ty3), hold_txt, fill=hold_col)

    if not has_voice:
        draw.rectangle([x + 12, y + 268, x + W_ - 12, y + 288],
                       fill=(40, 20, 22, 220),
                       outline=(255, 70, 70, 200))
        draw.text((x + 20, y + 272), "♪ 0  休止", fill=(255, 90, 90))


def draw_no_voice_badge(canvas, x, y, active):
    if not active:
        return
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = 46, 46
    draw.rounded_rectangle([x, y, x + w, y + h],
                           radius=8, fill=(30, 20, 22, 220),
                           outline=(255, 70, 70, 200), width=2)
    txt = "0"
    bbox = draw.textbbox((0, 0), txt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2 - bbox[0],
               y + (h - th) / 2 - bbox[1]),
              txt, fill=(255, 90, 90, 255))


# ==========================================================================
# 渲染
# ==========================================================================
def render(midi_seq, voiced_seq, gliss_seq, legato_seq,
           next_in_seq, hold_left_seq, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    n = len(midi_seq)

    mw = int(W * cfg["width_ratio"])
    mh = cfg["height_px"]
    mx = (W - mw) // 2
    my = int(H * cfg["y_ratio"])

    window_frames = int(cfg["window_sec"] * fps)

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

            draw_pitch_meter(canvas, mx, my, mw, mh,
                             midi_seq, voiced_seq, i, window_frames,
                             fmin=cfg["fmin"], fmax=cfg["fmax"],
                             line_width=cfg["line_width"],
                             dot_size=cfg["dot_size"],
                             bpm=cfg["bpm"],
                             beat_offset=cfg["beat_offset"],
                             beats_per_bar=cfg["beats_per_bar"],
                             fps=fps)

            cur_sec = i / float(fps)
            bar_num, beat_idx, in_beat = _bar_number_and_beat(
                cur_sec, cfg["bpm"], cfg["beat_offset"],
                cfg["beats_per_bar"])

            draw_music_panel(canvas, 20, 20,
                             beats=cfg["beats_per_bar"],
                             unit=cfg["beat_unit"],
                             bar_num=bar_num,
                             beat_idx=beat_idx,
                             in_beat=in_beat,
                             is_gliss=bool(gliss_seq[i]),
                             is_legato=bool(legato_seq[i]),
                             has_voice=bool(voiced_seq[i]),
                             next_in_sec=float(next_in_seq[i]),
                             hold_left_sec=float(hold_left_seq[i]))

            draw_no_voice_badge(canvas, W - 70, 20,
                                active=not bool(voiced_seq[i]))

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
        description="动态简谱音符计（Demucs 命令行分离 + 音乐分析面板）")
    ap.add_argument("input", help="输入音频文件")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")

    ap.add_argument("--auto-vocals", dest="auto_vocals",
                    action="store_true",
                    help="用 Demucs 自动分离人声（推荐）")
    ap.add_argument("--cache-dir", default=".vocals_cache",
                    help="人声缓存目录")
    ap.add_argument("--vocals", default=None,
                    help="手动指定人声轨（覆盖 --auto-vocals）")

    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--fmin", type=float, default=80.0)
    ap.add_argument("--fmax", type=float, default=1000.0)
    ap.add_argument("--frame-ms", type=float, default=60.0)
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--hold-ms", type=float, default=150.0)
    ap.add_argument("--energy-gate", type=float, default=-50.0)

    ap.add_argument("--width-ratio", type=float, default=0.9)
    ap.add_argument("--height-px", type=int, default=300)
    ap.add_argument("--y-ratio", type=float, default=0.35)
    ap.add_argument("--window-sec", type=float, default=8.0)

    ap.add_argument("--line-width", type=int, default=5)
    ap.add_argument("--dot-size", type=int, default=5)

    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--beat-unit", type=int, default=4)
    ap.add_argument("--beat-offset", type=float, default=0.0)
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="手动 BPM，0=自动检测")

    ap.add_argument("--gliss-semitones", type=float, default=2.0)
    ap.add_argument("--gliss-win", type=float, default=0.12)
    ap.add_argument("--legato-min", type=float, default=0.35)
    ap.add_argument("--legato-jitter", type=float, default=0.35)

    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg，请先安装")

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_pitch.mp4"

    print("[1/5] 准备音频", file=sys.stderr)
    vocal_path = None

    if a.vocals and os.path.isfile(a.vocals):
        vocal_path = a.vocals
        print(f"      手动人声轨: {vocal_path}", file=sys.stderr)
    elif a.auto_vocals:
        vocal_path = find_or_create_vocals(a.input, a.cache_dir)
        if vocal_path:
            print(f"      使用人声: {vocal_path}", file=sys.stderr)
        else:
            print("      分离失败，退回原曲", file=sys.stderr)
    else:
        print("      未开启分离，使用原曲", file=sys.stderr)

    if vocal_path and os.path.isfile(vocal_path):
        vdata, sr = load_audio(vocal_path)
    else:
        vdata, sr = load_audio(a.input)

    mono = vdata.mean(axis=1) if vdata.shape[1] > 1 else vdata[:, 0]

    print("[2/5] BPM 检测", file=sys.stderr)
    if a.bpm > 0:
        bpm = a.bpm
        print(f"      手动 BPM = {bpm:.1f}", file=sys.stderr)
    else:
        bpm = detect_bpm(mono, sr, a.fps)
        print(f"      自动 BPM ≈ {bpm:.1f}", file=sys.stderr)

    print("[3/5] 音高检测（YIN）", file=sys.stderr)
    f0, voiced = detect_pitch_from_vocals(
        mono, sr, a.fps, a.fmin, a.fmax,
        a.frame_ms, a.threshold, a.energy_gate)
    f0, voiced = smooth_pitch(f0, voiced, a.fps, a.hold_ms)

    midi = np.zeros_like(f0)
    mask = voiced & (f0 > 0)
    midi[mask] = hz_to_midi(f0[mask])

    last = hz_to_midi(a.fmin)
    for i in range(len(midi)):
        if mask[i]:
            last = midi[i]
        else:
            midi[i] = last

    print("[4/5] 音乐特征分析（滑音 / 连音 / 时序）", file=sys.stderr)
    gliss, legato = analyze_music_features(
        midi, mask, a.fps,
        gliss_semitones=a.gliss_semitones,
        gliss_win_sec=a.gliss_win,
        legato_min_sec=a.legato_min,
        legato_max_jitter=a.legato_jitter)
    next_in, hold_left = compute_timing(mask, a.fps)
    print(f"      滑音帧: {int(gliss.sum())}  连音帧: {int(legato.sum())}",
          file=sys.stderr)

    cfg = {
        "input": a.input, "fps": a.fps, "width": a.width,
        "height": a.height, "fmin": a.fmin, "fmax": a.fmax,
        "width_ratio": a.width_ratio, "height_px": a.height_px,
        "y_ratio": a.y_ratio, "window_sec": a.window_sec,
        "line_width": a.line_width, "dot_size": a.dot_size,
        "bpm": bpm, "beat_offset": a.beat_offset,
        "beats_per_bar": a.beats_per_bar, "beat_unit": a.beat_unit,
    }

    print(f"[5/5] 渲染 {len(midi)} 帧 → {out_path}", file=sys.stderr)
    render(midi, mask, gliss, legato, next_in, hold_left, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

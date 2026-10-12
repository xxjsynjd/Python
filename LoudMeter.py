#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from MultiTrackRender import MultiTrackRenderer

BG = (16, 17, 21)
DIAL_BG = (30, 32, 40)
TICK_COL = (180, 185, 195)
TEXT_COL = (210, 215, 225)
BAR_COL = (80, 200, 120)
CURSOR_COL = (255, 255, 255)
TARGET_COL = (255, 200, 40)
PEAK_COL = (255, 255, 255)


class LoudMeterState:
    def __init__(self, sr=44100, fps=30, release_sec=0.5, peak_hold_sec=1.5):
        self.sr = sr
        self.fps = fps
        self.rel_step = 1.0 / max(release_sec * fps, 1e-6)
        self.pks_step = 1.0 / max(peak_hold_sec * fps, 1e-6)
        self.win_samples = int(sr * 0.4)
        self.reset()

    def reset(self):
        self.audio_cache = np.zeros(0, dtype=np.float32)
        self.ptr = -60.0
        self.pk = -60.0

    def process_audio_chunk(self, new_audio):
        if len(new_audio) == 0:
            return self.ptr, self.pk
        mono = new_audio.mean(axis=1).astype(np.float32) / 32768.0
        self.audio_cache = np.concatenate([self.audio_cache, mono])
        if len(self.audio_cache) > self.win_samples * 2:
            self.audio_cache = self.audio_cache[-self.win_samples:]
        if len(self.audio_cache) < self.win_samples // 2:
            return self.ptr, self.pk
        mean_sq = np.mean(self.audio_cache ** 2)
        lufs = -0.691 + 10.0 * np.log10(max(mean_sq, 1e-10))
        self.ptr = max(lufs, self.ptr - self.rel_step)
        self.pk = max(lufs, self.pk - self.pks_step)
        return self.ptr, self.pk


def draw_lufs_meter(canvas, x, y, w, h, lufs_now, peak_val,
                    lufs_target=-14.0, db_min=-60.0, db_max=0.0):
    draw = ImageDraw.Draw(canvas, "RGBA")
    def xp(db): return x + int((db - db_min) / (db_max - db_min) * w)

    draw.rectangle([x, y, x + w, y + h], fill=DIAL_BG)
    draw.rectangle([xp(-24), y, xp(-9), y + h], fill=(40, 130, 70))
    draw.rectangle([xp(-9), y, xp(-6), y + h], fill=(170, 140, 40))
    draw.rectangle([xp(-6), y, x + w, y + h], fill=(170, 50, 45))

    for db in (-36, -24, -14, -9, -6):
        px = xp(db)
        draw.line([px, y - 3, px, y + h + 3], fill=TICK_COL, width=1)

    cur_x = max(x, min(x + w, xp(lufs_now)))
    draw.rectangle([x, y, cur_x, y + h], fill=BAR_COL)
    draw.line([cur_x, y - 4, cur_x, y + h + 4], fill=CURSOR_COL, width=2)

    if peak_val is not None:
        px = max(x, min(x + w - 2, xp(peak_val)))
        draw.rectangle([px, y, px + 2, y + h], fill=PEAK_COL)

    tx = xp(lufs_target)
    draw.polygon([(tx, y - 6), (tx - 5, y - 14), (tx + 5, y - 14)], fill=TARGET_COL)
    txt = f"{lufs_now:.1f} LUFS"
    draw.text((x + 6, y + h + 10), txt, fill=TEXT_COL)


def main():
    ap = argparse.ArgumentParser(description="实时响度计 LUFS - 多文件版")
    ap.add_argument("inputs", nargs="+", help="输入音频文件（可多个）")
    ap.add_argument("-o", "--output", default=None, help="输出视频文件")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--release", type=float, default=0.5)
    ap.add_argument("--peak-hold", type=float, default=1.5)
    ap.add_argument("--window", type=float, default=400.0)
    ap.add_argument("--lufs-target", type=float, default=-14.0)
    ap.add_argument("--transition", type=float, default=1.5, help="下一曲过渡时长（秒）")
    ap.add_argument("--cover-intro", type=float, default=0.0, help="封面开场时长（秒）")

    a = ap.parse_args()

    if len(a.inputs) > 1:
        print(f"\n⚠️ 检测到 {len(a.inputs)} 个音频文件。", flush=True)
        ans = input(f"是否合并为一个视频，并添加 {a.transition} 秒的下一曲过渡动画？(y/n): ")
        if ans.lower() != 'y':
            sys.exit(0)

    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.inputs[0]))[0]
        out_path = base + "_multi_lufs.mp4"

    state = LoudMeterState(sr=44100, fps=a.fps,
                           release_sec=a.release, peak_hold_sec=a.peak_hold)

    def render_frame(audio_window, current_time, st):
        if len(audio_window) > 0:
            lufs_now, peak_val = st.process_audio_chunk(audio_window)
        else:
            lufs_now, peak_val = -60.0, -60.0
        W = a.width // 2 * 2
        H = a.height // 2 * 2
        canvas = Image.new("RGB", (W, H), BG)
        mw = int(W * 0.8)
        mh = int(H * 0.08)
        mx = (W - mw) // 2
        my = int(H * 0.5)
        draw_lufs_meter(canvas, mx, my, mw, mh, lufs_now, peak_val,
                        lufs_target=a.lufs_target)
        return canvas

    print(f"[1/3] 初始化流式渲染引擎", flush=True)
    renderer = MultiTrackRenderer(
        a.inputs, out_path,
        fps=a.fps, width=a.width, height=a.height,
        transition_sec=a.transition,
        cover_intro_sec=a.cover_intro,
    )
    print(f"[2/3] 开始渲染 {renderer.total_frames} 帧 → {out_path}", flush=True)
    print(f"[3/3] 渲染中...", flush=True)

    renderer.run(render_frame, state)
    print(f"输出: {out_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerateAll.py —— 一首歌，全套可视化，一条命令跑完

自动依次调用所有可视化脚本，每首歌输出一整套 MP4。
脚本不存在时自动跳过；支持续跑、跳过已存在、限流。

用法:
    GenerateAll.py 歌.mp3
    GenerateAll.py 歌.mp3 -o out_dir
    GenerateAll.py 歌.mp3 --modes spectrum3d,pitch,lyrics
    GenerateAll.py 歌.mp3 --skip-existing
    GenerateAll.py 歌.mp3 --no-auto-vocals
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable

# 每个模式: (脚本名, 输出后缀, 附加参数)
MODES = {
    "spectrum3d": ("Spectrum3DAudacious.py", "_3d.mp4", [
        "--yaw-offset", "180", "--osc-angle", "40",
        "--hue-offset", "0.375", "--flip-freq",
        "--scale-x", "0.336",
    ]),
    "spectrum2d": ("Spectrum_2d.py", "_spec.mp4", []),
    "loud":       ("LoudMeter.py", "_loud.mp4", []),
    "thrust":     ("ThrustMeter.py", "_thrust.mp4", []),
    "vibration":  ("VibrationMeter.py", "_vib.mp4", []),
    "noisegate":  ("NoiseGateMeter.py", "_gate.mp4", []),
    "phase":      ("PhaseMeter.py", "_phase.mp4", []),
    "timeline":   ("TimelineRoll.py", "_timeline.mp4", []),
    "pitch":      ("PitchMeter1.py", "_pitch.mp4", ["--auto-vocals"]),
    "lyrics":     ("LyricsDisplayAI.py", "_lyrics.mp4", [
        "--auto-vocals", "--model", "base", "--lang", "zh",
    ]),
    "progress":   ("ProgressBar.py", "_progress.mp4", []),
    "speaker":    ("SpeakerMeter.py", "_speaker.mp4", []),
    "physics":    ("SpeakerPhysics.py", "_physics.mp4", []),
    "wmp":        ("WMPVisualizer.py", "_wmp.mp4", []),
    "waterfall":  ("WaterfallWMP.py", "_waterfall.mp4", []),
    "waterfall3d":("Waterfall3D.py", "_wf3d.mp4", []),
    "vu":         ("VU_Meter.py", "_vu.mp4", []),
}

DEFAULT_ORDER = list(MODES.keys())

SEARCH_DIRS = [
    os.path.expanduser("~/音频转视频工具"),
    os.path.expanduser("~/tools"),
    os.path.expanduser("~/bin"),
    os.path.expanduser("~"),
]


def find_script(name):
    """找脚本：PATH 优先，其次 home 下的几个常见目录"""
    p = shutil.which(name)
    if p:
        return p
    for d in SEARCH_DIRS:
        cand = os.path.join(d, name)
        if os.path.isfile(cand):
            return cand
    return None


def run_one(script_path, audio_path, out_path, extra_args, timeout=3600):
    cmd = [PY, script_path, audio_path, "-o", out_path] + extra_args
    t0 = time.time()
    try:
        # stderr 直通终端，保留进度条
        p = subprocess.run(cmd, timeout=timeout)
        dt = time.time() - t0
        if p.returncode != 0:
            return False, f"退出码 {p.returncode}", dt
        return True, "", dt
    except subprocess.TimeoutExpired:
        return False, "超时", time.time() - t0
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return False, str(e), time.time() - t0


def main():
    ap = argparse.ArgumentParser(
        description="一首歌，全套可视化，一条命令跑完",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", help="输入音频文件")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="输出目录（默认与音频同目录下的 <歌名>_videos/）")
    ap.add_argument("--modes", default="all",
                    help="要跑的模式，逗号分隔，或 all。可选: " +
                         ",".join(MODES.keys()))
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true",
                    help="已存在的跳过")
    ap.add_argument("--no-auto-vocals", action="store_true",
                    help="禁用 AI 人声分离")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="单个脚本超时秒数")
    a = ap.parse_args()

    if not os.path.isfile(a.input):
        raise SystemExit(f"找不到文件: {a.input}")

    if a.modes.strip().lower() == "all":
        modes = DEFAULT_ORDER
    else:
        modes = [m.strip() for m in a.modes.split(",")
                 if m.strip() in MODES]

    if not modes:
        raise SystemExit("没有有效的模式")

    if a.output_dir is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        a.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(a.input)),
            f"{base}_videos")
    os.makedirs(a.output_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(a.input))[0]

    print("=" * 56)
    print(f"输入   : {a.input}")
    print(f"输出   : {a.output_dir}")
    print(f"模式   : {len(modes)} 个 → {', '.join(modes)}")
    print(f"分辨率 : {a.width}x{a.height} @ {a.fps}fps")
    print("=" * 56)

    t_total = time.time()
    ok = 0
    skipped = 0
    fail = []

    for idx, mode in enumerate(modes, 1):
        script_name, suffix, extra = MODES[mode]
        script_path = find_script(script_name)

        if not script_path:
            print(f"[{idx}/{len(modes)}] {mode:12s} ✘ 脚本未找到 ({script_name})")
            fail.append((mode, "脚本未找到"))
            continue

        out_path = os.path.join(a.output_dir, name + suffix)

        if a.skip_existing and os.path.exists(out_path):
            print(f"[{idx}/{len(modes)}] {mode:12s} ⏭ 已存在，跳过")
            skipped += 1
            continue

        args = list(extra) + [
            "--width", str(a.width),
            "--height", str(a.height),
            "--fps", str(a.fps),
        ]
        if a.no_auto_vocals:
            args = [x for x in args if x != "--auto-vocals"]

        print(f"[{idx}/{len(modes)}] {mode:12s} ▶ 开始")
        sys.stdout.flush()

        success, err, dt = run_one(script_path, a.input, out_path,
                                    args, timeout=a.timeout)

        if success:
            print(f"[{idx}/{len(modes)}] {mode:12s} ✔ 完成 ({dt:.1f}s)")
            ok += 1
        else:
            print(f"[{idx}/{len(modes)}] {mode:12s} ✘ 失败 ({dt:.1f}s): {err}")
            fail.append((mode, err))

    dt_total = time.time() - t_total

    print("=" * 56)
    print(f"总计: {ok} 成功, {skipped} 跳过, {len(fail)} 失败")
    print(f"耗时: {dt_total:.1f} 秒 ({dt_total/60:.1f} 分钟)")
    if fail:
        print("\n失败清单:")
        for m, e in fail:
            print(f"  {m}: {e}")
    print("=" * 56)


if __name__ == "__main__":
    main()

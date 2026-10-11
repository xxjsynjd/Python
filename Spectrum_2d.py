#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 竖条频谱分析仪（对数频率分带，含原声）
用法:
    python spectrum_2d.py song.mp3 -o out.mp4 --bands 64 --release 0.4
"""
import argparse, math, os, shutil, subprocess, sys
import numpy as np
from PIL import Image

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (8, 8, 14)
PEAK = (255, 255, 255)
GRID = (30, 32, 44)
COLOR_STOPS = [
    (-80.0, (10, 30, 80)), (-55.0, (20, 90, 190)), (-35.0, (40, 170, 220)),
    (-22.0, (70, 220, 160)), (-14.0, (200, 230, 60)), (-6.0, (255, 170, 40)),
    (-1.0, (255, 70, 60)),
]


def _decode_ffmpeg(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=sample_rate,channels",
                        "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    parts = p.stdout.strip().splitlines()[0].split(",")
    sr, ch = int(float(parts[0])), int(float(parts[1]))
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path,
                          "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch).copy(), sr


def load_audio(path):
    if not os.path.isfile(path): raise SystemExit(f"找不到文件: {path}")
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        if not (FFMPEG and FFPROBE): raise SystemExit("读音频失败")
        data, sr = _decode_ffmpeg(path)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def spectrum_bands(data, sr, fps, fft_size, bands, fmin, fmax):
    """逐帧 FFT，按对数频率分带 → (n_frames, bands) dB 矩阵"""
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    n = len(mono)
    hop = sr / fps
    n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    # 对数分带边界
    edges = np.geomspace(max(fmin, 20.0), min(fmax, sr / 2 - 1), bands + 1)
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
            seg = np.zeros(fft_size, dtype=np.float32); seg[-a:] = mono[:b]
        elif b > n:
            seg = np.zeros(fft_size, dtype=np.float32); seg[:n - a] = mono[a:]
        else:
            seg = mono[a:b]
        spec = np.abs(np.fft.rfft(seg * win))
        spec = 20 * np.log10(np.maximum(spec / (fft_size / 4), 1e-10))
        for k, idx in enumerate(band_idx):
            out[i, k] = spec[idx].max()
        if i % 200 == 0:
            print(f"\r  频谱 {i*100//n_frames:3d}%", end="", file=sys.stderr)
    print(file=sys.stderr)
    return out


def ballistics_bands(db, fps, rel_db):
    n, b = db.shape
    out = np.empty_like(db)
    cur = np.full(b, -140.0)
    step = rel_db / fps
    for i in range(n):
        cur = np.maximum(db[i], cur - step)
        out[i] = cur
    return out


def make_textures(bar_h, bar_w, db_min, db_max, seg_h):
    ys = np.arange(bar_h, dtype=np.float64)
    frac = (bar_h - 1 - ys) / max(bar_h - 1, 1)
    db = db_min + frac * (db_max - db_min)
    xp = np.array([c[0] for c in COLOR_STOPS])
    arr = np.empty((bar_h, 3), dtype=np.float64)
    for k in range(3):
        fp = np.array([c[1][k] for c in COLOR_STOPS])
        arr[:, k] = np.interp(db, xp, fp)
    arr = np.repeat(arr[:, None, :], bar_w, axis=1)
    if seg_h > 1:
        rows = np.arange(bar_h)
        gap = ((bar_h - 1 - rows) % seg_h) == 0
        arr[gap, :, :] *= 0.25
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB"), \
           Image.fromarray((arr.astype(np.float32) * 0.10).astype(np.uint8), "RGB")


def render(norm, cfg, out_path):
    W, H = cfg["width"] // 2 * 2, cfg["height"] // 2 * 2
    bands = norm.shape[1]
    fps = cfg["fps"]

    mx = max(16, int(W * 0.03))
    mt = max(30, int(H * 0.05))
    mb = max(40, int(H * 0.06))
    gap = 2
    avail = W - 2 * mx - gap * (bands - 1)
    bar_w = max(2, avail // bands)
    bar_h = H - mt - mb
    bright, dim = make_textures(bar_h, bar_w, cfg["db_min"], cfg["db_max"], cfg["seg"])

    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", cfg["input"], "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    total = norm.shape[0]
    report = max(1, total // 100)
    try:
        for i in range(total):
            canvas = Image.new("RGB", (W, H), BG)
            # 参考网格线
            for g in range(1, 5):
                y = mt + bar_h * g // 5
                canvas.paste(GRID, (mx, y, W - mx, y + 1))
            for b in range(bands):
                x = mx + b * (bar_w + gap)
                canvas.paste(dim, (x, mt))
                fill = max(0, min(bar_h, int(round(float(norm[i, b]) * bar_h))))
                if fill > 0:
                    canvas.paste(bright.crop((0, bar_h - fill, bar_w, bar_h)),
                                 (x, mt + bar_h - fill))
            proc.stdin.write(canvas.tobytes("raw", "RGB"))
            if i % report == 0:
                print(f"\r渲染中… {i*100//total:3d}%", end="", file=sys.stderr)
    finally:
        try: proc.stdin.close()
        except Exception: pass
        proc.wait()
    print("\r完成 ✔", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bands", type=int, default=64, help="频段数量（竖条数）")
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--fmin", type=float, default=30.0)
    ap.add_argument("--fmax", type=float, default=16000.0)
    ap.add_argument("--db-min", type=float, default=-80.0)
    ap.add_argument("--db-max", type=float, default=0.0)
    ap.add_argument("--release", type=float, default=0.4)
    ap.add_argument("--release-db", type=float, default=30.0)
    ap.add_argument("--seg", type=int, default=6)
    a = ap.parse_args()

    if not FFMPEG: raise SystemExit("未找到 ffmpeg")
    out = a.output or (os.path.splitext(os.path.basename(a.input))[0] + "_spec.mp4")

    print("[1/4] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    print(f"      {sr} Hz, {data.shape[1]}ch, {len(data)/sr:.2f}s", file=sys.stderr)

    print(f"[2/4] FFT 分带 ({a.bands} bands)", file=sys.stderr)
    spec = spectrum_bands(data, sr, a.fps, a.fft_size, a.bands, a.fmin, a.fmax)

    print("[3/4] 弹道 + 归一化", file=sys.stderr)
    rel = a.release_db / a.release
    spec = ballistics_bands(spec, a.fps, rel)
    span = a.db_max - a.db_min
    norm = np.clip((spec - a.db_min) / span, 0, 1).astype(np.float32)

    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height,
               db_min=a.db_min, db_max=a.db_max, seg=a.seg)
    print("[4/4] 渲染", file=sys.stderr)
    render(norm, cfg, out)
    print(f"输出: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

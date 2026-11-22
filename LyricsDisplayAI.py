#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LyricsDisplayAI.py —— 歌词显示（AI 对齐 + 上滑 + 对齐/排版/排列方向）

功能:
  1. 读歌词（--lrc 优先 > 同目录 .lrc > ID3）
  2. --auto-vocals 用 Demucs 分离人声（带缓存，秒读）
  3. Whisper 对人声轨逐字对齐
  4. 唱到哪弹哪 + 上滑过渡 + 水平对齐 + 排版模式 + 排列方向
  5. 读取不到歌词时不退出：画面以跨平台等宽字体红字显示
     “fatal: Unable to read ID3 tag: No Lyrics”
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

BG = (8, 9, 14)
COL_PAST = (120, 130, 145, 200)
COL_NOW_BASE = (225, 228, 235, 245)
COL_NOW_HL = (255, 80, 80, 255)
COL_FUTURE = (170, 178, 190, 225)
COL_SHADOW = (0, 0, 0, 190)

# 无歌词时画面红字报错文本（等宽字体渲染）
FATAL_TEXT = "fatal: Unable to read ID3 tag: No Lyrics"
COL_FATAL = (255, 56, 56, 255)


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


LRC_TS = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]")
LRC_META = re.compile(r"^\[(ti|ar|al|by|offset):(.*)\]$", re.IGNORECASE)


def parse_lrc(text):
    lines = []
    offset_ms = 0
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = LRC_META.match(raw)
        if m:
            if m.group(1).lower() == "offset":
                try:
                    offset_ms = int(m.group(2).strip())
                except ValueError:
                    pass
            continue
        stamps = list(LRC_TS.finditer(raw))
        if not stamps:
            continue
        content = raw[stamps[-1].end():].strip()
        for st in stamps:
            mm = int(st.group(1))
            ss = int(st.group(2))
            frac = st.group(3) or "0"
            if len(frac) == 1:
                ms = int(frac) * 100
            elif len(frac) == 2:
                ms = int(frac) * 10
            else:
                ms = int(frac[:3])
            t = mm * 60 + ss + ms / 1000.0 + offset_ms / 1000.0
            if content:
                lines.append((max(0.0, t), content))
    lines.sort(key=lambda x: x[0])
    return lines


def read_lrc_file(audio_path):
    base = os.path.splitext(audio_path)[0]
    for ext in (".lrc", ".LRC"):
        p = base + ext
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return parse_lrc(f.read())
            except UnicodeDecodeError:
                with open(p, "r", encoding="gbk", errors="ignore") as f:
                    return parse_lrc(f.read())
    return None


def read_id3_lyrics(audio_path):
    try:
        from mutagen import File as MFile
    except ImportError:
        return None
    try:
        m = MFile(audio_path)
    except Exception:
        return None
    if m is None:
        return None
    candidates = []
    for key in (m.keys() if hasattr(m, "keys") else []):
        if key.startswith("USLT"):
            try:
                candidates.append(str(m[key]))
            except Exception:
                pass
    for field in ("lyrics", "LYRICS", "unsyncedlyrics",
                  "UNSYNCEDLYRICS", "©lyr"):
        if field in m:
            try:
                val = m[field]
                if isinstance(val, list):
                    val = val[0]
                candidates.append(str(val))
            except Exception:
                pass
    for text in candidates:
        lines = parse_lrc(text)
        if lines:
            return lines
    return None


def load_lyrics(audio_path, lrc_path=None):
    if lrc_path and os.path.isfile(lrc_path):
        try:
            with open(lrc_path, "r", encoding="utf-8") as f:
                lines = parse_lrc(f.read())
        except UnicodeDecodeError:
            with open(lrc_path, "r", encoding="gbk", errors="ignore") as f:
                lines = parse_lrc(f.read())
        if lines:
            print(f"  歌词来源: 指定文件 {lrc_path} ({len(lines)} 行)",
                  file=sys.stderr)
            return lines

    lines = read_lrc_file(audio_path)
    if lines:
        print(f"  歌词来源: LRC ({len(lines)} 行)", file=sys.stderr)
        return lines

    lines = read_id3_lyrics(audio_path)
    if lines:
        print(f"  歌词来源: ID3 ({len(lines)} 行)", file=sys.stderr)
        return lines

    print("  未读取到歌词（无指定 .lrc / 同名 .lrc / ID3 歌词）",
          file=sys.stderr)
    return []


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
    print("      正在用 Demucs 分离人声...", file=sys.stderr)
    tmp_out = os.path.join(cache_dir, "_tmp_" + base)
    os.makedirs(tmp_out, exist_ok=True)
    cmd = [demucs_bin, "--two-stems", "vocals",
           "-o", tmp_out, "--float32",
           "-n", "htdemucs", "-d", "cpu", audio_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None
    if p.returncode != 0:
        print(f"      分离失败: {p.stderr.strip()[-200:]}", file=sys.stderr)
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
                        os.path.join(cache_dir, base + "_instrumental.wav"))
        except Exception:
            pass
    shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"      人声已缓存: {cache_path}", file=sys.stderr)
    return cache_path


def align_words_with_whisper(audio_path, lrc_lines, model_size="base",
                             lang="zh"):
    engine = None
    try:
        from faster_whisper import WhisperModel
        print(f"  使用 faster-whisper ({model_size})", file=sys.stderr)
        engine = ("faster", WhisperModel(model_size, device="cpu",
                                          compute_type="float32"))
    except ImportError:
        pass
    if engine is None:
        try:
            import whisper
            print(f"  使用 openai-whisper ({model_size})", file=sys.stderr)
            engine = ("whisper", whisper.load_model(model_size))
        except ImportError:
            print("  未找到 whisper 库，退回均匀分配", file=sys.stderr)
            return None

    print("  AI 对齐中…", file=sys.stderr)
    words = []
    if engine[0] == "faster":
        segments, _ = engine[1].transcribe(
            audio_path, language=lang,
            word_timestamps=True, beam_size=5)
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append((w.start, w.end, w.word.strip()))
    else:
        result = engine[1].transcribe(audio_path, language=lang,
                                       word_timestamps=True)
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append((w["start"], w["end"], w["word"].strip()))

    if not words:
        print("  AI 未返回词时间戳，退回均匀分配", file=sys.stderr)
        return None

    print(f"  AI 识别到 {len(words)} 个词", file=sys.stderr)

    char_times = []
    for line_idx, (line_t, line_text) in enumerate(lrc_lines):
        t_start = line_t
        t_end = lrc_lines[line_idx + 1][0] if line_idx + 1 < len(lrc_lines) \
                else line_t + 4.0
        words_in = [(ws, we, w) for ws, we, w in words
                    if ws >= t_start - 0.3 and ws < t_end + 0.3]
        if not words_in:
            n = len(line_text)
            step = (t_end - t_start) / max(n, 1)
            for k, ch in enumerate(line_text):
                char_times.append((ch, t_start + k * step,
                                   t_start + (k + 1) * step))
            continue

        total_chars = len(line_text)
        total_word_chars = sum(len(w[2]) for w in words_in)
        if total_word_chars == 0:
            total_word_chars = 1

        char_idx = 0
        for ws, we, w in words_in:
            w_len = len(w)
            span = max(1, int(round(w_len * total_chars /
                                     total_word_chars)))
            span = min(span, total_chars - char_idx)
            if span <= 0:
                continue
            for j in range(span):
                if char_idx >= total_chars:
                    break
                ch = line_text[char_idx]
                cs = ws + (we - ws) * j / span
                ce = ws + (we - ws) * (j + 1) / span
                char_times.append((ch, cs, ce))
                char_idx += 1

        if char_idx < total_chars:
            n_rest = total_chars - char_idx
            step = max(0.05, (t_end - t_start) / max(total_chars, 1))
            for j in range(n_rest):
                ch = line_text[char_idx + j]
                cs = t_end - (n_rest - j) * step
                ce = cs + step
                char_times.append((ch, cs, ce))

    return char_times


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


def load_mono_font(size):
    """跨平台加载等宽字体（Android Termux / Linux / Windows / macOS）。

    找不到等宽字体时回退到通用字体，再退回 PIL 默认字体。
    """
    candidates = (
        # Android / Termux
        "/system/fonts/DroidSansMono.ttf",
        "/data/data/com.termux/files/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        # Windows
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        # macOS
        "/System/Library/Fonts/Menlo.ttc",
    )
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return load_font(None, size)


def is_punct(ch):
    return ch in "，。！？、；：,.!?;:—… \t\n\"'（）()《》【】"


def char_bounce(cur_sec, t_start, max_scale=1.35, duration=0.18):
    dt = cur_sec - t_start
    if dt < 0 or dt >= duration or duration <= 1e-6:
        return 1.0, 0.0
    p = dt / duration
    rise = 0.55
    if p < rise:
        e = p / rise
        amp = math.sin(e * math.pi / 2)
    else:
        e = (p - rise) / (1 - rise)
        amp = math.cos(e * math.pi / 2)
    return 1.0 + (max_scale - 1.0) * amp, -8.0 * amp


def _text_bbox(font, text):
    return font.getbbox(text)


def _char_w(font, ch):
    bbox = font.getbbox(ch)
    return bbox[2] - bbox[0]


def _wrap_idx_h(font, text, max_w):
    """横排按宽度贪心折行，返回每行对应的原字符索引列表。
    英文/数字单词整体不拆，优先在空格处断行；单词自身超宽才逐字硬拆。"""
    n = len(text)
    if not max_w or max_w <= 0:
        return [list(range(n))]
    spans = [m.span() for m in re.finditer(
        r"[A-Za-z0-9]+(?:[.\-'][A-Za-z0-9]+)*|\s|.", text)]
    lines_idx, cur_idx, cur_w = [], [], 0
    for a, b in spans:
        idx = list(range(a, b))
        tw = sum(_char_w(font, text[k]) for k in idx)
        tok = text[a:b]
        is_word = bool(tok) and not tok.isspace() and re.match(r"[A-Za-z0-9]", tok[0])
        if cur_idx and cur_w + tw > max_w:
            lines_idx.append(cur_idx)
            cur_idx, cur_w = [], 0
        if is_word and tw > max_w:
            # 单词比可用宽度还宽：逐字硬拆，保证不横向溢出
            for k in idx:
                cw = _char_w(font, text[k])
                if cur_idx and cur_w + cw > max_w:
                    lines_idx.append(cur_idx)
                    cur_idx, cur_w = [k], cw
                else:
                    cur_idx.append(k)
                    cur_w += cw
        else:
            cur_idx += idx
            cur_w += tw
    if cur_idx:
        lines_idx.append(cur_idx)
    return [sl for sl in lines_idx if sl]


def wrap_text_h(font, text, max_w):
    """横排按最大宽度贪心折行，返回行文本列表；max_w 为空/<=0 时不折行。"""
    if not text:
        return []
    if not max_w or max_w <= 0:
        return [text]
    return [text[sl[0]:sl[-1] + 1].rstrip()
            for sl in _wrap_idx_h(font, text, max_w)]


def _draw_text_static(canvas, x, y, text, font, col, shadow_col):
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((x + 2, y + 2), text, font=font, fill=shadow_col)
    draw.text((x, y), text, font=font, fill=col)


def draw_line_h(canvas, cx, y, text, font, col, shadow_col, align, max_w=None):
    """横排静态句；超出 max_w 时自动折成多行，垂直居中堆叠。"""
    draw = ImageDraw.Draw(canvas, "RGBA")
    lines = wrap_text_h(font, text, max_w)
    lh = int(font.size * 1.35)
    total_h = lh * len(lines)
    for j, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if align == "center":
            x = cx - tw / 2 - bbox[0]
        elif align == "left":
            x = cx - bbox[0]
        else:
            x = cx - tw - bbox[0]
        line_y = y - total_h / 2 + j * lh
        yy = line_y + lh / 2 - th / 2 - bbox[1]
        _draw_text_static(canvas, x, yy, line, font, col, shadow_col)


def draw_line_v(canvas, cx, cy, text, font, col, shadow_col, align):
    """竖排一列：从上到下，每字一行（居中/顶/底 对齐）"""
    draw = ImageDraw.Draw(canvas, "RGBA")
    n = len(text)
    if n == 0:
        return
    ch_h = font.size
    gap = int(ch_h * 1.15)
    total_h = gap * n
    if align == "center":
        y0 = cy - total_h / 2
    elif align == "left":   # 竖排时，"left" = 顶端对齐
        y0 = cy
    else:                    # right = 底部对齐
        y0 = cy - total_h

    for k, ch in enumerate(text):
        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = cx - cw / 2 - bbox[0]
        y = y0 + k * gap - th / 2 - bbox[1]
        _draw_text_static(canvas, x, y, ch, font, col, shadow_col)


def draw_current_h(canvas, cx, y_center, text, font,
                   char_times, cur_sec,
                   base_col, hl_col, shadow_col,
                   bounce_scale=1.35, bounce_dur=0.18,
                   align="center", max_w=None):
    """横排当前句：逐字弹跳；超出 max_w 时自动折成多行。"""
    n = len(text)
    if n == 0:
        return
    widths = [_char_w(font, ch) for ch in text]

    # 按宽度贪心分行，记录每行的原字符索引（英文单词整体不拆）
    if max_w and max_w > 0:
        lines_idx = _wrap_idx_h(font, text, max_w)
    else:
        lines_idx = [list(range(n))]

    row_w = [sum(widths[k] for k in idxs) for idxs in lines_idx]
    char_row, char_prefix = {}, {}
    for j, idxs in enumerate(lines_idx):
        run = 0
        for k in idxs:
            char_row[k] = j
            char_prefix[k] = run
            run += widths[k]

    base_bbox = font.getbbox("测Ag")
    line_h = base_bbox[3] - base_bbox[1]
    lh = int(line_h * 1.3)
    pad = int(line_h * 0.9)
    layer_w = int(max(row_w)) + pad * 2
    layer_h = int(lh * len(lines_idx)) + pad * 2
    layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))

    for k in range(n):
        ch = text[k]
        if k < len(char_times):
            _, cs, ce = char_times[k]
        else:
            cs = ce = 0.0
        bbox = font.getbbox(ch)
        cw = bbox[2] - bbox[0]
        ch_h = bbox[3] - bbox[1]

        j = char_row[k]
        if align == "center":
            row_x = (layer_w - row_w[j]) / 2
        elif align == "left":
            row_x = pad
        else:
            row_x = layer_w - pad - row_w[j]
        char_cx_layer = row_x + char_prefix[k] + cw / 2
        char_cy_layer = pad + lh * (j + 0.5)

        if is_punct(ch):
            sx = sy = 1.0
            dy = 0.0
        else:
            sx, dy = char_bounce(cur_sec, cs, bounce_scale, bounce_dur)
            sy = sx

        is_sung = cur_sec >= cs
        col = hl_col if is_sung else base_col

        pad_c = int(ch_h) + 8
        cw_i = int(cw * sx) + pad_c * 2
        ch_i = int(ch_h * sy) + pad_c * 2
        cimg = Image.new("RGBA", (cw_i, ch_i), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cimg)
        ox = pad_c - bbox[0]
        oy = pad_c - bbox[1]
        cd.text((ox + 2, oy + 2), ch, font=font, fill=shadow_col)
        cd.text((ox, oy), ch, font=font, fill=col)

        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            cimg = cimg.resize((max(1, int(cw_i * sx)),
                                max(1, int(ch_i * sy))),
                               Image.BILINEAR)
        paste_x = char_cx_layer - cimg.width / 2
        paste_y = char_cy_layer - cimg.height / 2 + dy
        layer.alpha_composite(cimg, (int(paste_x), int(paste_y)))

    if align == "center":
        layer_x = cx - layer_w / 2
    elif align == "left":
        layer_x = cx - pad
    else:
        layer_x = cx - layer_w + pad
    canvas.alpha_composite(layer, (int(layer_x),
                                   int(y_center - layer_h / 2)))


def draw_current_v(canvas, cx, cy_center, text, font,
                   char_times, cur_sec,
                   base_col, hl_col, shadow_col,
                   bounce_scale=1.35, bounce_dur=0.18,
                   align="center"):
    """竖排当前句：逐字弹跳，从下到上"""
    n = len(text)
    if n == 0:
        return
    ch_h = font.size
    gap = int(ch_h * 1.15)
    total_h = gap * n
    if align == "center":
        y0 = cy_center - total_h / 2
    elif align == "left":    # 顶端
        y0 = cy_center
    else:                     # 底部
        y0 = cy_center - total_h

    draw = ImageDraw.Draw(canvas, "RGBA")
    for k, ch in enumerate(text):
        if k < len(char_times):
            _, cs, ce = char_times[k]
        else:
            cs = ce = 0.0
        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if is_punct(ch):
            sx = sy = 1.0
            dy = 0.0
        else:
            sx, dy = char_bounce(cur_sec, cs, bounce_scale, bounce_dur)
            sy = sx

        is_sung = cur_sec >= cs
        col = hl_col if is_sung else base_col

        c_x = cx - cw / 2 - bbox[0]
        c_y = y0 + k * gap - th / 2 - bbox[1]

        # 用临时图做缩放
        pad = int(th) + 8
        cimg = Image.new("RGBA", (int(cw * sx) + pad * 2,
                                  int(th * sy) + pad * 2), (0, 0, 0, 0))
        cd = ImageDraw.Draw(cimg)
        ox = pad - bbox[0]
        oy = pad - bbox[1]
        cd.text((ox + 2, oy + 2), ch, font=font, fill=shadow_col)
        cd.text((ox, oy), ch, font=font, fill=col)
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            cimg = cimg.resize((max(1, int(cimg.width * sx)),
                                max(1, int(cimg.height * sy))),
                               Image.BILINEAR)
        canvas.alpha_composite(cimg,
                               (int(c_x - pad + (cimg.width - cw * sx) / 2),
                                int(c_y - pad + dy)))


def draw_lyrics(canvas, cx, cy, lyrics, char_times_map, cur_sec,
                font_cur, font_other, visible_lines=5, line_gap=1.7,
                bounce_scale=1.35, bounce_dur=0.18,
                scroll_threshold=0.85, scroll_ease="smoothstep",
                align="center", layout="stack", arrange="h",
                stagger=60, max_w=None):
    if not lyrics:
        return
    times = [t for t, _ in lyrics]
    n = len(lyrics)

    lo, hi = 0, n - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] <= cur_sec:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1

    line_h = int(font_cur.size * line_gap)
    total_h = line_h * visible_lines
    y_start = cy - total_h // 2

    # ---- 上滑过渡 ----
    scroll_offset = 0
    if layout == "stack" and 0 <= idx < n - 1:
        t_now = times[idx]
        t_next = times[idx + 1]
        span = max(t_next - t_now, 1e-3)
        frac = (cur_sec - t_now) / span
        if frac > scroll_threshold:
            p = (frac - scroll_threshold) / max(1.0 - scroll_threshold, 1e-3)
            p = max(0.0, min(1.0, p))
            if scroll_ease == "smoothstep":
                p = p * p * (3 - 2 * p)
            elif scroll_ease == "smootherstep":
                p = p * p * p * (p * (p * 6 - 15) + 10)
            elif scroll_ease == "easeout":
                p = 1 - (1 - p) ** 3
            scroll_offset = -int(p * line_h)

    # ---- 错开量 ----
    def stagger_dx(i):
        if arrange != "h-stagger":
            return 0
        return stagger if (i % 2 == 0) else -stagger

    def stagger_dy(i):
        if arrange != "v-stagger":
            return 0
        return stagger if (i % 2 == 0) else -stagger

    # ---- 当前句折多行时的纵向避让：把上下紧邻句各推开溢出量的一半 ----
    h_ov = 0.0
    if arrange in ("h", "h-stagger") and 0 <= idx < n and max_w:
        rows_cur = len(wrap_text_h(font_cur, lyrics[idx][1], max_w))
        if rows_cur > 1:
            lh_cur = font_cur.size * 1.3
            pad_cur = font_cur.size * 0.9
            h_cur = lh_cur * rows_cur + pad_cur * 2
            h_ov = max(0.0, (h_cur - line_h) / 2.0)

    for slot in range(visible_lines):
        offset = slot - (visible_lines // 2)
        i = idx + offset
        if i < 0 or i >= n:
            continue

        y = y_start + slot * line_h + scroll_offset + stagger_dy(i)
        if arrange in ("h", "h-stagger"):
            if i < idx:
                y -= int(h_ov)
            elif i > idx:
                y += int(h_ov)
        x = cx + stagger_dx(i)
        text = lyrics[i][1]

        is_current = (i == idx)

        if arrange == "h":
            if is_current and i in char_times_map:
                draw_current_h(canvas, x, y, text, font_cur,
                               char_times_map[i], cur_sec,
                               COL_NOW_BASE, COL_NOW_HL, COL_SHADOW,
                               bounce_scale, bounce_dur, align, max_w)
            else:
                col = COL_PAST if i < idx else COL_FUTURE
                draw_line_h(canvas, x, y, text, font_other,
                            col, COL_SHADOW, align, max_w)
        elif arrange == "h-stagger":
            if is_current and i in char_times_map:
                draw_current_h(canvas, x, y, text, font_cur,
                               char_times_map[i], cur_sec,
                               COL_NOW_BASE, COL_NOW_HL, COL_SHADOW,
                               bounce_scale, bounce_dur, align, max_w)
            else:
                col = COL_PAST if i < idx else COL_FUTURE
                draw_line_h(canvas, x, y, text, font_other,
                            col, COL_SHADOW, align, max_w)
        elif arrange == "v":
            if is_current and i in char_times_map:
                draw_current_v(canvas, x, y, text, font_cur,
                               char_times_map[i], cur_sec,
                               COL_NOW_BASE, COL_NOW_HL, COL_SHADOW,
                               bounce_scale, bounce_dur, align)
            else:
                col = COL_PAST if i < idx else COL_FUTURE
                draw_line_v(canvas, x, y, text, font_other,
                            col, COL_SHADOW, align)
        elif arrange == "v-stagger":
            if is_current and i in char_times_map:
                draw_current_v(canvas, x, y, text, font_cur,
                               char_times_map[i], cur_sec,
                               COL_NOW_BASE, COL_NOW_HL, COL_SHADOW,
                               bounce_scale, bounce_dur, align)
            else:
                col = COL_PAST if i < idx else COL_FUTURE
                draw_line_v(canvas, x, y, text, font_other,
                            col, COL_SHADOW, align)


def render(lyrics, char_times_map, total_sec, cfg, out_path):
    W = cfg["width"] // 2 * 2
    H = cfg["height"] // 2 * 2
    fps = cfg["fps"]
    ratio = cfg.get("width_ratio", 0.8)
    # 可用宽度占视频宽度的比例，限制在 70%~100%，不按很小的固定值折行
    ratio = min(1.0, max(0.7, float(ratio)))
    max_w = int(W * ratio)
    n_frames = max(1, int(math.ceil(total_sec * fps)))
    font_cur = load_font(cfg["font"], cfg["font_size_cur"])
    font_other = load_font(cfg["font"], cfg["font_size_other"])

    # 无歌词报错模式：等宽红字，字号自适应到整行不超出可用宽度
    fatal = bool(cfg.get("fatal_error", False))
    mono_font = None
    if fatal:
        avail_w = int(W * 0.9)
        mono_size = max(16, int(H * 0.045))
        mono_font = load_mono_font(mono_size)
        while mono_size > 12:
            bb = mono_font.getbbox(FATAL_TEXT)
            if bb[2] - bb[0] <= avail_w:
                break
            mono_size -= 2
            mono_font = load_mono_font(mono_size)

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
            base = Image.new("RGB", (W, H), BG)
            canvas = base.convert("RGBA")
            cur_sec = i / float(fps)
            if fatal:
                d = ImageDraw.Draw(canvas)
                bb = mono_font.getbbox(FATAL_TEXT)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                tx = (W - tw) // 2 - bb[0]
                ty = (H - th) // 2 - bb[1]
                d.text((tx, ty), FATAL_TEXT, font=mono_font, fill=COL_FATAL)
            else:
                draw_lyrics(canvas, W // 2, int(H * cfg["y_ratio"]),
                            lyrics, char_times_map, cur_sec,
                            font_cur, font_other,
                            visible_lines=cfg["visible_lines"],
                            line_gap=cfg["line_gap"],
                            bounce_scale=cfg["bounce_scale"],
                            bounce_dur=cfg["bounce_dur"],
                            scroll_threshold=cfg["scroll_threshold"],
                            scroll_ease=cfg["scroll_ease"],
                            align=cfg["align"],
                            layout=cfg["layout"],
                            arrange=cfg["arrange"],
                            stagger=cfg["stagger"],
                            max_w=max_w)
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


def main():
    ap = argparse.ArgumentParser(description="歌词显示（AI 对齐 + 全排版）")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--model", default="base",
                    help="whisper 模型: tiny / base / small / medium")
    ap.add_argument("--lang", default="zh", help="语言代码")
    ap.add_argument("--auto-vocals", dest="auto_vocals",
                    action="store_true", help="Demucs 分离人声后再对齐")
    ap.add_argument("--cache-dir", default=".vocals_cache")
    ap.add_argument("--vocals", default=None)
    ap.add_argument("--lrc", default=None,
                    help="手动指定 .lrc 文件路径")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-size-cur", type=int, default=52)
    ap.add_argument("--font-size-other", type=int, default=36)
    ap.add_argument("--visible-lines", type=int, default=5)
    ap.add_argument("--line-gap", type=float, default=1.7)
    ap.add_argument("--y-ratio", type=float, default=0.5)
    ap.add_argument("--bounce-scale", type=float, default=1.35)
    ap.add_argument("--bounce-dur", type=float, default=0.18)
    ap.add_argument("--scroll-threshold", type=float, default=0.85)
    ap.add_argument("--scroll-ease", default="smoothstep",
                    choices=["linear", "smoothstep", "smootherstep", "easeout"])
    ap.add_argument("--align", default="center",
                    choices=["left", "center", "right"],
                    help="水平对齐（或竖排时的顶/中/底对齐）")
    ap.add_argument("--layout", default="stack",
                    choices=["stack", "scroll"],
                    help="stack=上滑过渡；scroll=硬切")
    ap.add_argument("--arrange", default="h",
                    choices=["h", "v", "h-stagger", "v-stagger"],
                    help="排列方式：h=横排；v=竖排；h-stagger=横排错开；v-stagger=竖排错开")
    ap.add_argument("--stagger", type=int, default=60,
                    help="错开模式下的像素偏移量")
    ap.add_argument("--width-ratio", type=float, default=0.8,
                    help="横排可用宽度占视频宽度的比例（0.7~1.0），默认 0.8；"
                         "超出该宽度的句子自动折行")
    a = ap.parse_args()

    if not FFMPEG:
        raise SystemExit("未找到 ffmpeg")
    out_path = a.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(a.input))[0]
        out_path = base + "_lyrics_ai.mp4"

    print("[1/4] 读取音频", file=sys.stderr)
    data, sr = load_audio(a.input)
    total_sec = len(data) / sr
    print(f"      {sr} Hz, {data.shape[1]}ch, {total_sec:.2f}s",
          file=sys.stderr)

    print("[2/4] 读取歌词", file=sys.stderr)
    lyrics = load_lyrics(a.input, a.lrc)
    no_lyrics = len(lyrics) == 0
    if no_lyrics:
        print("      未读取到歌词：画面将以等宽红字报错提示", file=sys.stderr)
    else:
        for t, txt in lyrics[:3]:
            print(f"      [{t:6.2f}s] {txt}", file=sys.stderr)

    char_times_map = {}
    if not no_lyrics:
        align_path = a.input
        if a.vocals and os.path.isfile(a.vocals):
            align_path = a.vocals
            print(f"      人声轨: {a.vocals}", file=sys.stderr)
        elif a.auto_vocals:
            vp = find_or_create_vocals(a.input, a.cache_dir)
            if vp:
                align_path = vp
                print(f"      使用人声: {vp}", file=sys.stderr)
            else:
                print("      分离失败，退回原曲", file=sys.stderr)

        print("[3/4] AI 逐字对齐", file=sys.stderr)
        char_times = align_words_with_whisper(align_path, lyrics,
                                              a.model, a.lang)

        if char_times:
            pos = 0
            for i, (t, txt) in enumerate(lyrics):
                n = len(txt)
                char_times_map[i] = char_times[pos:pos + n]
                pos += n
            print(f"      对齐 {len(char_times)} 个字", file=sys.stderr)
        else:
            for i, (t, txt) in enumerate(lyrics):
                t_end = lyrics[i + 1][0] if i + 1 < len(lyrics) else t + 4.0
                n = len(txt)
                step = (t_end - t) / max(n, 1)
                char_times_map[i] = [(txt[k], t + k * step, t + (k + 1) * step)
                                     for k in range(n)]
            print("      使用均匀分配（无 AI 时间戳）", file=sys.stderr)
    else:
        print("[3/4] 跳过 AI 对齐（无歌词）", file=sys.stderr)

    cfg = dict(
        input=a.input, fps=a.fps, width=a.width, height=a.height,
        font=a.font,
        font_size_cur=a.font_size_cur,
        font_size_other=a.font_size_other,
        visible_lines=a.visible_lines,
        line_gap=a.line_gap,
        y_ratio=a.y_ratio,
        bounce_scale=a.bounce_scale,
        bounce_dur=a.bounce_dur,
        scroll_threshold=a.scroll_threshold,
        scroll_ease=a.scroll_ease,
        align=a.align,
        layout=a.layout,
        arrange=a.arrange,
        stagger=a.stagger,
        width_ratio=a.width_ratio,
        fatal_error=no_lyrics,
    )

    print(f"[4/4] 渲染 {int(total_sec * a.fps)} 帧 → {out_path}",
          file=sys.stderr)
    render(lyrics, char_times_map, total_sec, cfg, out_path)
    print(f"输出: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

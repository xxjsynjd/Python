#!/data/data/com.termux/files/usr/bin/python3
import os
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

# ===================== 配置区（Termux XFCE + Android） =====================
R_LOAD = 8.0
U_REF_DBU = 0.7746
# Digital-7数码管字体查找顺序
FONT_CANDIDATES_DIGITAL = [
    "/usr/share/fonts/Digital-7.ttf",
    "/system/fonts/Digital-7.ttf",
    "/sdcard/Fonts/Digital-7.ttf"
]
# 单位常规字体查找顺序
FONT_CANDIDATES_UNIT = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/system/fonts/Roboto-Regular.ttf",
    "/system/fonts/DroidSans.ttf",
    "/sdcard/Fonts/DejaVuSans.ttf"
]
FONT_SIZE_BG = 72
FONT_SIZE_FG = 72
FONT_SIZE_UNIT = 30

BG_RGB = (136, 136, 136)        # 画布底色 #888888
BG_TEXT_RGB = (110, 110, 110)   # 衬底8888.88暗灰色
FG_TEXT_RGB = (255, 0, 0)       # 上层动态读数红色
UNIT_TEXT_RGB = (255,255,255)   # 单位文字白色

W, H = 1280, 420
FPS = 24
BG_TEXT_CONTENT = "8888.88"
PANEL_COUNT = 3
RIGHT_MARGIN = 50
center_y = H // 2 - 20   # 数字垂直中心
UNIT_OFFSET_Y = 70       # 单位在数字下方偏移距离
TMP_FRAME_DIR = "./_tmp_frames"
# ==========================================================================

def insert_space_before_every_one(s:str) -> str:
    """每一个字符'1'前面插入一个空格"""
    out = []
    for ch in s:
        if ch == '1':
            out.append(' ')
        out.append(ch)
    return ''.join(out)

def fmt_calc(val:float) -> str:
    """格式化，保留两位小数，然后对每一个1前面补空格"""
    raw = f"{val:.2f}"
    return insert_space_before_every_one(raw)

def load_font(candidates, size):
    for fp in candidates:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    # 兜底默认字体
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default(size=size)

def calc_audio_peak(buf, sr, chunk):
    import soundfile as sf
    total_samples = len(buf)
    n_frames = math.ceil(total_samples / (sr / FPS))
    i_buf = np.zeros(n_frames)
    db_buf = np.zeros(n_frames)
    u_buf = np.zeros(n_frames)
    for f_idx in range(n_frames):
        start = int(f_idx * sr / FPS)
        end = min(start + chunk, total_samples)
        frame_data = buf[start:end]
        peak = np.max(np.abs(frame_data))
        u = peak
        a = u / R_LOAD
        dbu = 20 * math.log10(u / U_REF_DBU) if u > 1e-9 else -99.0
        i_buf[f_idx] = a
        db_buf[f_idx] = dbu
        u_buf[f_idx] = u
    return i_buf, db_buf, u_buf

def make_frames(i_buf, db_buf, u_buf, font_bg, font_fg, font_unit):
    os.makedirs(TMP_FRAME_DIR, exist_ok=True)
    n = len(i_buf)
    panel_w = W // PANEL_COUNT
    panel_right_x = [
        1 * panel_w - RIGHT_MARGIN,
        2 * panel_w - RIGHT_MARGIN,
        3 * panel_w - RIGHT_MARGIN,
    ]
    unit_list = ["A", "dBu", "V"]
    for idx in range(n):
        img = Image.new("RGB", (W, H), BG_RGB)
        draw = ImageDraw.Draw(img)
        values = [i_buf[idx], db_buf[idx], u_buf[idx]]
        for panel_idx, val in enumerate(values):
            base_x = panel_right_x[panel_idx]
            text_fg = fmt_calc(val)
            unit_txt = unit_list[panel_idx]

            # ==========【修复】先画衬底底层 8888.88 ==========
            bbox_bg = draw.textbbox((0, 0), BG_TEXT_CONTENT, font=font_bg)
            tw_bg = bbox_bg[2] - bbox_bg[0]
            th_bg = bbox_bg[3] - bbox_bg[1]
            x_bg = base_x - tw_bg
            y_bg = center_y - th_bg // 2
            draw.text((x_bg, y_bg), BG_TEXT_CONTENT, font=font_bg, fill=BG_TEXT_RGB)

            # ==========【后画红色动态数字，浮在上层】 ==========
            bbox_fg = draw.textbbox((0, 0), text_fg, font=font_fg)
            tw_fg = bbox_fg[2] - bbox_fg[0]
            th_fg = bbox_fg[3] - bbox_fg[1]
            x_fg = base_x - tw_fg
            y_fg = center_y - th_fg // 2
            draw.text((x_fg, y_fg), text_fg, font=font_fg, fill=FG_TEXT_RGB)

            # 下方单位文字
            bbox_unit = draw.textbbox((0,0), unit_txt, font=font_unit)
            tw_unit = bbox_unit[2] - bbox_unit[0]
            x_unit = base_x - tw_unit
            y_unit = center_y + UNIT_OFFSET_Y
            draw.text((x_unit, y_unit), unit_txt, font=font_unit, fill=UNIT_TEXT_RGB)

        out_png = os.path.join(TMP_FRAME_DIR, f"f_{idx:06d}.png")
        img.save(out_png)
        print(f"\r生成帧 {idx+1}/{n}", end="")
    print("\n帧序列生成完毕")

def render_ffmpeg(audio_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(TMP_FRAME_DIR, "f_%06d.png"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_path
    ]
    print("开始ffmpeg合成视频...")
    subprocess.run(cmd, check=True)

def clean_tmp():
    for f in os.listdir(TMP_FRAME_DIR):
        os.remove(os.path.join(TMP_FRAME_DIR, f))
    os.rmdir(TMP_FRAME_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="音频文件路径")
    args = parser.parse_args()
    audio_path = args.audio
    if not os.path.isfile(audio_path):
        print("音频文件不存在！")
        return
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_file = f"{base_name}_level_panel.mp4"

    print("加载字体...")
    font_bg = load_font(FONT_CANDIDATES_DIGITAL, FONT_SIZE_BG)
    font_fg = load_font(FONT_CANDIDATES_DIGITAL, FONT_SIZE_FG)
    font_unit = load_font(FONT_CANDIDATES_UNIT, FONT_SIZE_UNIT)

    import soundfile as sf
    print("读取音频...")
    buf, sr = sf.read(audio_path)
    if len(buf.shape) > 1:
        buf = buf[:,0]
    chunk = int(sr / FPS)
    print("计算电平...")
    i_buf, db_buf, u_buf = calc_audio_peak(buf, sr, chunk)
    print("渲染画面帧...")
    make_frames(i_buf, db_buf, u_buf, font_bg, font_fg, font_unit)
    render_ffmpeg(audio_path, out_file)
    clean_tmp()
    print(f"✅ 完成，输出文件：{out_file}")

if __name__ == "__main__":
    main()
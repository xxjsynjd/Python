#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

def find_chinese_font():
    """在 Termux 中自动寻找可用的中文字体"""
    possible_paths = [
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/system/fonts/NotoSansSC-Regular.otf",
        "/system/fonts/DroidSansFallbackFull.ttf",
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def format_path(filepath, max_len=30):
    """如果路径太长，截取末尾并加上省略号"""
    if len(filepath) <= max_len:
        return filepath
    return "..." + filepath[-(max_len-3):]

def main():
    parser = argparse.ArgumentParser(description="状态栏插件 - 混音版")
    parser.add_argument("audio", help="输入音频文件路径 (同时也是混音源)")
    parser.add_argument("-o", "--output", help="输出视频文件路径")
    parser.add_argument("--width", type=int, default=1280, help="视频宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720, help="视频高度 (默认: 720)")
    parser.add_argument("--fps", type=int, default=30, help="帧率 (默认: 30)")
    parser.add_argument("--author", type=str, default="xxjsynjd", help="源码作者名")
    parser.add_argument("--font", type=str, help="中文字体路径")
    
    args = parser.parse_args()

    # 1. 读取音频获取时长
    try:
        info = sf.info(args.audio)
        total_seconds = info.duration
    except Exception as e:
        print(f"❌ 无法读取音频信息: {e}")
        sys.exit(1)

    print(f"[1/3] 读取音频")
    print(f"      {info.samplerate} Hz, {info.channels}ch, {total_seconds:.2f}s")

    # 2. 配置字体
    font_path = args.font if args.font else find_chinese_font()
    bar_height = int(args.height * 0.08)
    
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, int(bar_height * 0.45))
            small_font = ImageFont.truetype(font_path, int(bar_height * 0.35))
        else:
            font = ImageFont.load_default()
            small_font = font
    except:
        font = ImageFont.load_default()
        small_font = font

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        output_path = f"{base_name}_status.mp4"

    # ================= 3. 配置 FFmpeg (核心混音逻辑) =================
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{args.width}x{args.height}',
        '-pix_fmt', 'rgb24',
        '-r', str(args.fps),
        '-i', '-',
        '-i', args.audio, 
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest', 
        output_path
    ]
    
    try:
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("❌ 找不到 ffmpeg，请先运行 pkg install ffmpeg")
        sys.exit(1)

    # 4. 渲染循环
    total_frames = int(total_seconds * args.fps)
    print(f"[2/3] 配置渲染参数")
    print(f"      分辨率: {args.width}x{args.height} @ {args.fps}fps, 总计 {total_frames} 帧")
    print(f"[3/3] 渲染中...")

    display_path = format_path(os.path.basename(args.audio), max_len=35)
    img = Image.new('RGB', (args.width, args.height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for frame_idx in range(total_frames):
        draw.rectangle([(0, 0), (args.width, args.height - bar_height)], fill=(0, 0, 0))
        
        current_sec = frame_idx / args.fps
        remaining_sec = total_seconds - current_sec
        if remaining_sec < 0:
            remaining_sec = 0.0

        time_str = f"{current_sec:.2f}s  -{remaining_sec:.2f}s  {total_seconds:.2f}s"

        bar_y_start = args.height - bar_height
        draw.rectangle([(0, bar_y_start), (args.width, args.height)], fill=(15, 15, 15))

        text_y = bar_y_start + (bar_height - int(bar_height * 0.45)) // 2

        draw.text((15, text_y), args.author, font=font, fill=(180, 180, 180))
        author_width = draw.textlength(args.author, font=font)
        draw.text((15 + author_width + 20, text_y + 2), display_path, font=small_font, fill=(120, 120, 120))
        time_width = draw.textlength(time_str, font=font)
        draw.text((args.width - time_width - 15, text_y), time_str, font=font, fill=(255, 255, 255))

        process.stdin.write(img.tobytes())
        
        if frame_idx % args.fps == 0:
            percent = int((frame_idx / total_frames) * 100)
            print(f"渲染中… {percent:3d}%", end='\r')

    process.stdin.close()
    process.wait()
    print(f"\n渲染完成 ✔ (已混入原曲音频)")
    print(f"输出: {output_path}")

if __name__ == "__main__":
    main()

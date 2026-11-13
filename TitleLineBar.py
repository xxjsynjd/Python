#!/usr/bin/env python3
# ==============================================================================
# 组件名: 信息技术疑难解答播放器标题栏 (全曲混流+中文字体修复版)
# ==============================================================================

import os
import sys
import subprocess
import json
import argparse

try:
    import pygame
except ImportError:
    print("[*] 正在自动安装 pygame...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pygame"], check=True)
    import pygame

class InfoTechTroubleshootingPlayerTitleBar:
    PLAYER_VERSION = "v1.0.2"
    APP_NAME = "XinXiJiShuYiNanJieDa Media Player"

    def __init__(self, truncate_len=20):
        self.truncate_len = truncate_len
        self._meta_cache = None

    def _format_time(self, seconds):
        if seconds is None or seconds < 0:
            return "00:00"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _truncate_text(self, text, max_length=None):
        if not text: return "Unknown"
        limit = max_length if max_length else self.truncate_len
        return text[:limit] + "..." if len(text) > limit else text

    def get_metadata(self, audio_path):
        if self._meta_cache: return self._meta_cache
        metadata = {"path": audio_path, "title": "", "artist": "", "album": "", "cover": "", "duration": 0.0}
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            metadata["duration"] = float(data.get("format", {}).get("duration", 0))
            tags = {k.lower(): v for k, v in data.get("format", {}).get("tags", {}).items()}
            metadata["title"] = tags.get("title", os.path.basename(audio_path).split('.')[0])
            metadata["artist"] = tags.get("artist", "Unknown Artist")
            metadata["album"] = tags.get("album", "Unknown Album")
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video" and stream.get("disposition", {}).get("attached_pic") == 1:
                    metadata["cover"] = "Embedded_Cover.jpg"
                    break
        except Exception:
            metadata["title"] = os.path.basename(audio_path).split('.')[0]
        self._meta_cache = metadata
        return metadata

    def generate_title(self, audio_path, current_time=0.0):
        meta = self.get_metadata(audio_path)
        duration = meta["duration"]
        remaining = max(0, duration - current_time)
        info_part = (
            f"[{self._truncate_text(meta['path'], 30)}] - "
            f"[{self._truncate_text(meta['title'])}] - "
            f"[{self._truncate_text(meta['artist'], 15)}] - "
            f"[{self._truncate_text(meta['album'], 15)}] - "
            f"[{self._truncate_text(meta['cover'], 15)}]"
        )
        time_part = (
            f"{self.APP_NAME} {self.PLAYER_VERSION} - "
            f"[{self._format_time(current_time)}] - "
            f"[{self._format_time(remaining)}] - "
            f"[{self._format_time(duration)}]"
        )
        return f"{info_part}\n{time_part}"

# ==============================================================================
# 🚀 核心修复：自动寻找中文字体
# ==============================================================================
def find_chinese_font():
    """在安卓系统中寻找支持中文的字体"""
    possible_fonts = [
        "/system/fonts/NotoSansCJK-Regular.ttc", # 安卓现代默认中文字体
        "/system/fonts/DroidSansFallback.ttf",   # 较老安卓的字体
        "/system/fonts/NotoSansSC-Regular.otf",  # 思源黑体
        "/system/fonts/MiSans-Regular.ttf",      # 小米字体
        "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf" # Termux自带(无中文但可防崩溃)
    ]
    for path in possible_fonts:
        if os.path.exists(path):
            return path
    return None

def render_video(audio_path, output_video, font_size=36, font_path=None, width=1920, height=1080, fps=30):
    title_bar = InfoTechTroubleshootingPlayerTitleBar()
    meta = title_bar.get_metadata(audio_path)
    duration = meta["duration"] + 1.0 
    
    pygame.init()
    screen = pygame.Surface((width, height))
    
    # 【关键修复】加载中文字体
    if not font_path:
        font_path = find_chinese_font()
        
    if font_path and os.path.exists(font_path):
        print(f"[*] 成功加载字体: {font_path}")
        font = pygame.font.Font(font_path, font_size)
    else:
        print("[!] 警告：未找到中文字体，中文可能显示为方框。")
        print("[!] 你可以使用 -f 参数手动指定字体路径。")
        font = pygame.font.Font(None, font_size) 

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-i", audio_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_video
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames = int(duration * fps)
    print(f"[*] 正在渲染全曲视频: {output_video}")
    print(f"[*] 时长: {duration:.2f} 秒 | 总帧数: {total_frames} 帧")

    try:
        for frame_idx in range(total_frames):
            current_time = frame_idx / fps
            if frame_idx % fps == 0:
                print(f"\r  -> 渲染进度: {int(current_time)} / {int(duration)} 秒", end="", flush=True)

            text_content = title_bar.generate_title(audio_path, current_time)
            lines = text_content.split("\n")
            
            screen.fill((30, 30, 30))
            margin_x, margin_y = 50, 50
            line_spacing = int(font_size * 1.5)
            
            for i, line in enumerate(lines):
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (margin_x, margin_y + (i * line_spacing)))
                
            frame_data = pygame.image.tostring(screen, "RGB")
            process.stdin.write(frame_data)
            
    except BrokenPipeError:
        print("\n[!] FFmpeg 管道意外断开。")
    finally:
        process.stdin.close()
        process.wait()
        pygame.quit()
        print(f"\n[✔] 渲染混流完成！保存为: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="信息技术疑难解答播放器标题栏 (全曲混流)")
    parser.add_argument("-i", "--input", required=True, help="音频文件路径")
    parser.add_argument("-o", "--output", help="自定义输出视频路径")
    parser.add_argument("-fs", "--font-size", type=int, default=36, help="视频字体大小")
    # 新增：手动指定字体
    parser.add_argument("-f", "--font", help="手动指定中文字体路径 (.ttf/.ttc)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] 错误：找不到文件 {args.input}")
        sys.exit(1)

    output_path = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0] + ".mp4"
    render_video(args.input, output_path, font_size=args.font_size, font_path=args.font)

if __name__ == "__main__":
    main()

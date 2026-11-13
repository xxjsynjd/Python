#!/usr/bin/env python3
# ==============================================================================
# 组件名: 信息技术疑难解答播放器标题栏 (像素级自适应截断版)
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

    def _truncate_text_pixels(self, text, font, max_width):
        """
        【核心算法】基于实际像素宽度进行截断
        :param text: 原始文本
        :param font: pygame.font.Font 对象
        :param max_width: 允许的最大像素宽度
        """
        if not text: return "Unknown"
        
        # 如果整体宽度没超，直接返回
        if font.size(text)[0] <= max_width:
            return text
        
        ellipsis = "..."
        ellipsis_width = font.size(ellipsis)[0]
        
        # 二分查找：找到能塞下文本 + 省略号的最大前缀
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if font.size(text[:mid])[0] + ellipsis_width <= max_width:
                left = mid
            else:
                right = mid - 1
                
        return text[:left] + ellipsis

    def get_metadata(self, audio_path):
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
        return metadata

    def generate_title(self, audio_path, current_time=0.0, font=None, max_width=None):
        meta = self.get_metadata(audio_path)
        duration = meta["duration"]
        remaining = max(0, duration - current_time)

        # 1. 拼接未经截断的超长原始字符串
        raw_info = (
            f"[{meta['path']}] - "
            f"[{meta['title']}] - "
            f"[{meta['artist']}] - "
            f"[{meta['album']}] - "
            f"[{meta['cover']}]"
        )

        # 2. 根据是否传入字体和宽度，决定用哪种截断方式
        if font and max_width:
            # 视频渲染模式：像素级自适应截断
            info_part = self._truncate_text_pixels(raw_info, font, max_width)
        else:
            # 终端纯文本预览模式：简单的字符数截断 (兜底)
            info_part = raw_info[:80] + "..." if len(raw_info) > 80 else raw_info

        # 3. 组装第二行（时间状态）
        time_part = (
            f"{self.APP_NAME} {self.PLAYER_VERSION} - "
            f"[{self._format_time(current_time)}] - "
            f"[{self._format_time(remaining)}] - "
            f"[{self._format_time(duration)}]"
        )
        return f"{info_part}\n{time_part}"

def find_chinese_font():
    possible_fonts = [
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/system/fonts/DroidSansFallback.ttf",
        "/system/fonts/NotoSansSC-Regular.otf",
        "/system/fonts/MiSans-Regular.ttf",
    ]
    for path in possible_fonts:
        if os.path.exists(path): return path
    return None

def render_video(audio_path, output_video, font_size, width, height, font_path=None, fps=30):
    title_bar = InfoTechTroubleshootingPlayerTitleBar()
    duration = title_bar.get_metadata(audio_path)["duration"] + 1.0
    
    pygame.init()
    screen = pygame.Surface((width, height))
    
    if not font_path:
        font_path = find_chinese_font()
        
    if font_path and os.path.exists(font_path):
        print(f"[*] 成功加载字体: {font_path}")
        font = pygame.font.Font(font_path, font_size)
    else:
        print("[!] 警告：未找到中文字体，中文可能显示为方框。")
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
    print(f"[*] 正在渲染视频: {output_video}")
    print(f"[*] 分辨率: {width}x{height} | 字体: {font_size}px | 时长: {duration:.2f} 秒")

    # 【关键计算】留出左右边距后，单行文本可用的最大像素宽度
    margin_x = 50
    max_text_width = width - (margin_x * 2)

    try:
        for frame_idx in range(total_frames):
            current_time = frame_idx / fps
            if frame_idx % fps == 0:
                print(f"\r  -> 渲染进度: {int(current_time)} / {int(duration)} 秒", end="", flush=True)

            # 传入 font 和 max_text_width，触发像素级截断
            text_content = title_bar.generate_title(audio_path, current_time, font=font, max_width=max_text_width)
            lines = text_content.split("\n")
            
            screen.fill((30, 30, 30))
            margin_y = 50
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
    parser = argparse.ArgumentParser(description="信息技术疑难解答播放器标题栏")
    parser.add_argument("-i", "--input", required=True, help="音频文件路径")
    parser.add_argument("-o", "--output", help="自定义输出视频路径")
    parser.add_argument("-r", "--resolution", default="1920x1080", help="视频分辨率 (宽x高)")
    parser.add_argument("-fs", "--font-size", type=int, default=36, help="视频字体大小(px)")
    parser.add_argument("-f", "--font", help="手动指定中文字体路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] 错误：找不到文件 {args.input}")
        sys.exit(1)

    try:
        width, height = map(int, args.resolution.lower().split('x'))
    except ValueError:
        print("[!] 错误：分辨率格式错误，应为 '宽x高'，例如 '1280x720'")
        sys.exit(1)

    output_path = args.output if args.output else os.path.splitext(os.path.basename(args.input))[0] + ".mp4"
    
    render_video(
        audio_path=args.input, 
        output_video=output_path, 
        font_size=args.font_size,
        width=width,
        height=height,
        font_path=args.font
    )

if __name__ == "__main__":
    main()

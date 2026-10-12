import os
import sys
import shutil
import hashlib
import subprocess
import numpy as np
import tempfile
from PIL import Image, ImageDraw, ImageFont

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
MPV = shutil.which("mpv")

FIXED_CACHE_DIR = os.path.expanduser("~/Python/.fixed_cache")


def _get_fixed_path(src_path):
    os.makedirs(FIXED_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(os.path.abspath(src_path).encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(FIXED_CACHE_DIR, f"{base}_{key}.fixed.wav")


def extract_cover_image(audio_path, out_dir):
    try:
        out_path = os.path.join(out_dir, "cover.jpg")
        result = subprocess.run([
            FFMPEG, "-y", "-loglevel", "quiet",
            "-i", audio_path,
            "-an", "-vcodec", "copy",
            out_path
        ], stderr=subprocess.DEVNULL)
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path
    except Exception:
        pass
    return None


def make_vinyl_cover(cover_path, size, angle_deg):
    S = size
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = int(S * 0.02)
    draw.ellipse([margin, margin, S - margin, S - margin], fill=(18, 18, 22))
    for r in range(int(S * 0.48), int(S * 0.30), -max(2, S // 80)):
        draw.ellipse([S // 2 - r, S // 2 - r, S // 2 + r, S // 2 + r],
                     outline=(45, 45, 55), width=1)
    cover_r = int(S * 0.30)
    if cover_path and os.path.exists(cover_path):
        try:
            cover = Image.open(cover_path).convert("RGB")
            w, h = cover.size
            side = min(w, h)
            cover = cover.crop(((w - side) // 2, (h - side) // 2,
                                (w + side) // 2, (h + side) // 2))
            cover = cover.resize((cover_r * 2, cover_r * 2), Image.Resampling.LANCZOS)
            cover = cover.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=False)
            mask = Image.new("L", (cover_r * 2, cover_r * 2), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, cover_r * 2, cover_r * 2], fill=255)
            img.paste(cover, (S // 2 - cover_r, S // 2 - cover_r), mask)
        except Exception:
            pass
    else:
        cover = Image.new("RGB", (cover_r * 2, cover_r * 2), (60, 80, 160))
        mask = Image.new("L", (cover_r * 2, cover_r * 2), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, cover_r * 2, cover_r * 2], fill=255)
        img.paste(cover, (S // 2 - cover_r, S // 2 - cover_r), mask)
    hole_r = int(S * 0.03)
    draw.ellipse([S // 2 - hole_r, S // 2 - hole_r,
                  S // 2 + hole_r, S // 2 + hole_r], fill=(220, 220, 230))
    return img


def load_font(size):
    candidates = [
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/system/fonts/NotoSansSC-Regular.otf",
        "/system/fonts/DroidSansFallbackFull.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


class MultiTrackRenderer:
    def __init__(self, input_files, output_path, fps=30, width=1280, height=720,
                 transition_sec=1.5, cover_intro_sec=0.0):
        self.input_files = input_files
        self.output_path = output_path
        self.fps = fps
        self.width = width // 2 * 2
        self.height = height // 2 * 2
        self.transition_frames = int(transition_sec * fps)
        self.cover_intro_frames = int(cover_intro_sec * fps)

        self.song_durations = []
        self.song_frames = []
        self.total_frames = 0

        print("正在探测音频时长...", flush=True)
        valid_files = []
        for f in input_files:
            p = subprocess.run([
                FFPROBE, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", f
            ], capture_output=True, text=True)
            duration_str = p.stdout.strip()
            if not duration_str:
                print(f"⚠️ 跳过无效文件: {os.path.basename(f)}", flush=True)
                continue
            try:
                duration = float(duration_str)
                if duration < 1.0:
                    continue
            except ValueError:
                continue
            valid_files.append(f)
            self.song_durations.append(duration)
            self.song_frames.append(int(duration * self.fps))
            self.total_frames += int(duration * self.fps)

        self.input_files = valid_files
        if not self.input_files:
            raise SystemExit("❌ 没有找到有效的音频文件")

    def render_transition(self, canvas_prev, canvas_next, progress):
        W, H = self.width, self.height
        final_canvas = Image.new("RGB", (W, H), (16, 17, 21))
        scale_prev = 1.0 - progress
        if scale_prev > 0.01:
            new_w, new_h = max(1, int(W * scale_prev)), max(1, int(H * scale_prev))
            resized = canvas_prev.resize((new_w, new_h), Image.Resampling.LANCZOS)
            offset_x = int(-W * progress + (W - new_w) / 2)
            offset_y = int(H * progress * 0.5 + (H - new_h) / 2)
            resized = Image.eval(resized, lambda x: int(x * (1.0 - progress)))
            final_canvas.paste(resized, (offset_x, offset_y))
        scale_next = progress
        if scale_next > 0.01:
            new_w, new_h = max(1, int(W * scale_next)), max(1, int(H * scale_next))
            resized = canvas_next.resize((new_w, new_h), Image.Resampling.LANCZOS)
            offset_x = int(W * (1.0 - progress) + (W - new_w) / 2)
            offset_y = int(H * (1.0 - progress) * 0.5 + (H - new_h) / 2)
            resized = Image.eval(resized, lambda x: int(x * progress))
            final_canvas.paste(resized, (offset_x, offset_y))
        return final_canvas

    def _use_fixed(self, path, wav_path):
        fp = _get_fixed_path(path)
        if os.path.exists(fp) and os.path.getsize(fp) > 1024:
            print(f"    ✔ 使用缓存: {os.path.basename(fp)}", flush=True)
            shutil.copy(fp, wav_path)
            return True
        return False

    def _try_ffmpeg(self, path, wav_path):
        result = subprocess.run([
            FFMPEG, "-y", "-loglevel", "quiet",
            "-f", "flac",
            "-analyzeduration", "0", "-probesize", "32",
            "-i", path,
            "-map", "0:a:0?",
            "-vn",
            "-ar", "44100", "-ac", "2",
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            wav_path
        ], stderr=subprocess.DEVNULL)
        return (result.returncode == 0
                and os.path.exists(wav_path)
                and os.path.getsize(wav_path) > 1024)

    def _try_mpv(self, path, wav_path):
        if not MPV:
            return False
        raw_wav = wav_path + ".mpv.wav"
        if os.path.exists(raw_wav):
            os.unlink(raw_wav)
        try:
            result = subprocess.run([
                MPV, "--no-video",
                "--ao=pcm",
                f"--ao-pcm-file={raw_wav}",
                path
            ], stderr=subprocess.DEVNULL, timeout=3600)
        except subprocess.TimeoutExpired:
            return False
        if not (result.returncode == 0 and os.path.exists(raw_wav)
                and os.path.getsize(raw_wav) > 1024):
            try:
                os.unlink(raw_wav)
            except Exception:
                pass
            return False

        verify = subprocess.run([
            FFMPEG, "-y", "-loglevel", "quiet",
            "-i", raw_wav,
            "-ar", "44100", "-ac", "2",
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            wav_path
        ], stderr=subprocess.DEVNULL)

        try:
            os.unlink(raw_wav)
        except Exception:
            pass

        ok = (verify.returncode == 0
              and os.path.exists(wav_path)
              and os.path.getsize(wav_path) > 1024)

        if ok:
            fp = _get_fixed_path(path)
            try:
                shutil.copy(wav_path, fp)
                print(f"    ✔ 已缓存修复结果: {os.path.basename(fp)}", flush=True)
            except Exception:
                pass

        return ok

    def _silence(self, duration, wav_path):
        subprocess.run([
            FFMPEG, "-y", "-loglevel", "quiet",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{duration:.3f}",
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            wav_path
        ], check=True)

    def _build_clean_wav(self):
        tmp_dir = tempfile.mkdtemp(prefix="lufs_wav_")
        wav_list = []
        print(f"正在逐文件统一转码为 16-bit WAV...", flush=True)
        for idx, path in enumerate(self.input_files):
            wav_path = os.path.join(tmp_dir, f"{idx:03d}.wav")
            expected_dur = self.song_durations[idx]
            print(f"  [{idx+1}/{len(self.input_files)}] {os.path.basename(path)} ({expected_dur:.1f}s)", flush=True)

            if self._use_fixed(path, wav_path):
                wav_list.append(wav_path)
                continue
            if self._try_ffmpeg(path, wav_path):
                wav_list.append(wav_path)
                continue

            print(f"    ⚠️ FFmpeg 解析失败，尝试 MPV 宽容解码...", flush=True)
            if self._try_mpv(path, wav_path):
                print(f"    ✔ MPV 解码成功", flush=True)
                wav_list.append(wav_path)
                continue

            print(f"    ⚠️ MPV 也失败，用 {expected_dur:.1f}s 静音补齐", flush=True)
            silent_path = os.path.join(tmp_dir, f"{idx:03d}_silent.wav")
            self._silence(expected_dur, silent_path)
            wav_list.append(silent_path)

        if not wav_list:
            raise SystemExit("❌ 所有文件都转换失败")

        merged_wav = os.path.join(tmp_dir, "merged.wav")
        list_txt = os.path.join(tmp_dir, "list.txt")
        with open(list_txt, "w") as f:
            for w in wav_list:
                f.write(f"file '{w}'\n")

        print(f"正在拼接 {len(wav_list)} 个 WAV 文件...", flush=True)
        subprocess.run([
            FFMPEG, "-y", "-loglevel", "quiet",
            "-f", "concat", "-safe", "0", "-i", list_txt,
            "-ar", "44100", "-ac", "2",
            "-sample_fmt", "s16",
            "-c:a", "pcm_s16le",
            merged_wav
        ], check=True)

        for w in wav_list:
            try:
                os.unlink(w)
            except Exception:
                pass
        try:
            os.unlink(list_txt)
        except Exception:
            pass

        print(f"音频统一转码完成: {merged_wav}", flush=True)
        return merged_wav

    def run(self, render_frame_func, state):
        merged_wav = self._build_clean_wav()

        print("正在读取 WAV 数据...", flush=True)
        raw = subprocess.run(
            [FFMPEG, "-v", "error", "-i", merged_wav,
             "-f", "s16le", "-ar", "44100", "-ac", "2", "-"],
            capture_output=True, check=True
        ).stdout
        audio_all = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).copy()
        print(f"WAV 读取完成，共 {len(audio_all)} 样本", flush=True)

        cmd = [
            FFMPEG, "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps), "-i", "pipe:0",
            "-i", merged_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", self.output_path
        ]

        print("[诊断] 正在启动最终 FFmpeg...", flush=True)
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=None, close_fds=True
        )
        print("[诊断] FFmpeg 已启动，开始逐帧推流", flush=True)

        cover_dir = tempfile.mkdtemp(prefix="cover_")
        cover_paths = []
        for f in self.input_files:
            cp = extract_cover_image(f, cover_dir)
            cover_paths.append(cp)

        current_song_idx = 0
        state.reset()

        global_frame = 0
        last_percent = -1
        total_frames = self.total_frames + self.cover_intro_frames
        total_samples = len(audio_all)
        W = self.width
        H = self.height

        try:
            while global_frame < total_frames:
                if global_frame < self.cover_intro_frames:
                    progress = global_frame / max(1, self.cover_intro_frames)
                    angle = global_frame * 1.5
                    canvas = Image.new("RGB", (W, H), (10, 10, 14))
                    vinyl = make_vinyl_cover(
                        cover_paths[0] if cover_paths else None,
                        int(min(W, H) * 0.7), angle)
                    if progress > 0.8:
                        k = (progress - 0.8) / 0.2
                        scale = 1.0 - 0.4 * k
                        alpha = 1.0 - k
                    else:
                        scale = 1.0
                        alpha = 1.0
                    vsize = int(vinyl.width * scale)
                    if vsize > 1:
                        vinyl = vinyl.resize((vsize, vsize), Image.Resampling.LANCZOS)
                        x = (W - vsize) // 2
                        y = (H - vsize) // 2 - int(H * 0.05)
                        if alpha < 1.0:
                            rgb = Image.eval(vinyl.convert("RGB"), lambda v: int(v * alpha))
                            canvas.paste(rgb, (x, y))
                        else:
                            canvas.paste(vinyl, (x, y), vinyl)
                    font_big = load_font(int(H * 0.05))
                    title = os.path.splitext(os.path.basename(self.input_files[0]))[0]
                    draw = ImageDraw.Draw(canvas)
                    tw = draw.textlength(title, font=font_big)
                    draw.text(((W - tw) / 2, H * 0.80), title,
                              font=font_big, fill=(230, 230, 240))
                    percent = int(global_frame * 100 / total_frames)
                    if percent != last_percent:
                        print(f"\r▶ 封面 {percent:3d}%  [{global_frame}/{total_frames} 帧]",
                              end="", flush=True)
                        last_percent = percent
                    process.stdin.write(canvas.tobytes("raw", "RGB"))
                    global_frame += 1
                    continue

                real_frame = global_frame - self.cover_intro_frames
                current_sec = real_frame / self.fps
                percent = int(global_frame * 100 / total_frames)
                if percent != last_percent:
                    print(f"\r▶ 渲染中… {percent:3d}%  [{global_frame}/{total_frames} 帧]",
                          end="", flush=True)
                    last_percent = percent

                song_idx = 0
                accumulated_time = 0
                for idx, dur in enumerate(self.song_durations):
                    if current_sec >= accumulated_time and current_sec < accumulated_time + dur:
                        song_idx = idx
                        break
                    accumulated_time += dur
                local_sec = current_sec - accumulated_time

                if song_idx != current_song_idx:
                    print(f"\n⏭️ 切换至: {os.path.basename(self.input_files[song_idx])}", flush=True)
                    current_song_idx = song_idx
                    state.reset()

                target_sample = min(int(current_sec * 44100), total_samples)
                window_samples = int(0.4 * 44100)
                start_sample = max(0, target_sample - window_samples)
                window_chunk = audio_all[start_sample:target_sample]
                if len(window_chunk) == 0:
                    window_chunk = np.zeros((0, 2), dtype=np.int16)

                canvas_cur = render_frame_func(window_chunk, local_sec, state)

                is_transition = False
                if song_idx < len(self.input_files) - 1:
                    song_total_frames = self.song_frames[song_idx]
                    local_frame = int(local_sec * self.fps)
                    trans_start = song_total_frames - self.transition_frames // 2
                    trans_end = song_total_frames + self.transition_frames // 2
                    if local_frame >= trans_start and local_frame < trans_end:
                        is_transition = True
                        progress = (local_frame - trans_start) / self.transition_frames

                if is_transition:
                    from LoudMeter import LoudMeterState
                    temp_state = LoudMeterState(sr=44100, fps=self.fps)
                    next_start = int(accumulated_time * 44100)
                    next_chunk = audio_all[next_start:next_start + int(0.4 * 44100)]
                    canvas_next = render_frame_func(next_chunk, 0.0, temp_state)
                    final_canvas = self.render_transition(canvas_cur, canvas_next, progress)
                else:
                    final_canvas = canvas_cur

                process.stdin.write(final_canvas.tobytes("raw", "RGB"))
                global_frame += 1

        finally:
            if process.stdin:
                process.stdin.close()
            process.wait()
            try:
                tmp_dir = os.path.dirname(merged_wav)
                if os.path.exists(merged_wav):
                    os.unlink(merged_wav)
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                if os.path.exists(cover_dir):
                    shutil.rmtree(cover_dir, ignore_errors=True)
            except Exception:
                pass
            print(f"\r▶ 渲染完成 ✔ 100%  [{total_frames}/{total_frames} 帧]", flush=True)
            print(f"最终文件: {self.output_path}", flush=True)

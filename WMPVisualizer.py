#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import colorsys
import math
import os
import random
import shutil
import subprocess
import sys
import json
import inspect
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
try:
    import librosa
except ImportError:
    librosa = None

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
BG = (5, 6, 12)
TEXT_COL = (200, 210, 220)
BEAT_COL = (40, 220, 120)
BPM_COL = (255, 220, 90)

# ==========================================================================
# 基础父类(必须放在最前面)
# ==========================================================================
class Scene:
    name = "scene"
    def __init__(self, W, H, ctx, params=None):
        self.W = W
        self.H = H
        self.ctx = ctx
        self.params = params

    def render(self, i, intensity):
        raise NotImplementedError

# ==========================================================================
# 新增：WMP Tolerance漩涡粒子场景(逆向行为复刻)
# ==========================================================================
class WmpToleranceScene(Scene):
    name = "WMP‑Tolerance漩涡粒子"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.cx = W / 2.0
        self.cy = H / 2.0
        self.max_particles = 600
        # [修复]先定义调色板！！
        self.palette = [
            (22, 124, 255),
            (84, 204, 255),
            (186, 104, 255),
            (54, 224, 224),
            (224, 94, 184),
        ]
        # 残影缓存，浮点实现拖尾
        self.trail_buf = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.trail_decay = 0.87

        # 粒子结构体列表(放到palette之后)
        self.particles = []
        for _ in range(self.max_particles):
            self.particles.append(self._make_particle())

    def _make_particle(self):
        """生成/重置单个粒子"""
        angle = random.random() * math.pi * 2.0
        radius = random.uniform(25, 90)
        return {
            "angle": angle,
            "radius": radius,
            "speed_r": random.uniform(0.25, 1.7),
            "speed_rot": random.uniform(-0.024, 0.024),
            "life": 1.0,
            "decay": random.uniform(0.0022, 0.0042),
            "color": random.choice(self.palette),
            "size": random.uniform(1.0, 2.8)
        }

    def render(self, i, intensity):
        bass = float(self.ctx["bass"][i])
        rms_val = float(self.ctx["rms"][i])

        # 新建黑色画布RGBA
        canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # 更新所有粒子
        for p in self.particles:
            # 低音驱动径向爆发速度
            p["speed_r"] += bass * 0.065
            p["angle"] += p["speed_rot"] + bass * 0.011

            p["radius"] += p["speed_r"]
            p["speed_r"] *= 0.984
            p["life"] -= p["decay"]

            # 死亡重生
            if p["life"] <= 0 or p["radius"] > max(self.W, self.H):
                p.update(self._make_particle())
                continue

            x = self.cx + math.cos(p["angle"]) * p["radius"]
            y = self.cy + math.sin(p["angle"]) * p["radius"]
            r, g, b = p["color"]
            alpha = int(255 * p["life"])
            s = max(1, int(p["size"] * p["life"]))
            draw.ellipse(
                [x - s, y - s, x + s, y + s],
                fill=(r, g, b, alpha)
            )

        # 中心微弱光晕
        draw.ellipse(
            [self.cx - 32, self.cy - 32, self.cx + 32, self.cy + 32],
            fill=(42, 92, 182, 65)
        )

        # 残影混合逻辑
        arr = np.array(canvas, dtype=np.float32)[:, :, :3]
        self.trail_buf *= self.trail_decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)

        out_img = Image.fromarray(self.trail_buf.astype(np.uint8), mode="RGB")
        return out_img

# ==========================================================================
# 新增：逆向微软WMP的3D水波纹引擎(Ambience)
# ==========================================================================
class WmpClassicAmbienceScene(Scene):
    name = "WMP经典水波"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.theta = 0.0
        self.render_w = 320
        self.render_h = 240
        self.cx = self.render_w // 2
        self.cy = self.render_h // 2

        # 注意：这里的残影缓存是小尺寸的(240x320)
        self.trail_buf = np.zeros((self.render_h, self.render_w, 3), dtype=np.float32)
        self.decay = 0.88

    def render(self, i, intensity):
        bass = float(self.ctx["bass"][i])
        self.theta += 0.05 + bass * 0.15
        pulse = (math.sin(self.theta) + 1.0) / 2.0

        small_canvas = Image.new("RGB", (self.render_w, self.render_h), (0, 0, 0))
        draw = ImageDraw.Draw(small_canvas, "RGBA")

        base_r = int(30 + 80 * pulse + 100 * bass)
        hue = (self.theta * 0.1) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.8)

        for offset in range(0, 40, 4):
            alpha = max(0, 255 - offset * 6)
            outline_col = (int(r * 255), int(g * 255), int(b * 255), alpha)
            rr = base_r - offset
            if rr > 0:
                draw.ellipse([self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr],
                             outline=outline_col, width=2)

        for _ in range(int(bass * 20)):
            px = self.cx + random.randint(-base_r, base_r)
            py = self.cy + random.randint(-base_r, base_r)
            draw.point((px, py), fill=(255, 255, 255, 200))
        # 【核心修复】：在320x240的小画布上叠加残影
        arr = np.asarray(small_canvas, np.float32)
        self.trail_buf *= self.decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)

        # 把叠加好残影的小画布转回Image
        small_final = Image.fromarray(self.trail_buf.astype(np.uint8))

        # 然后暴力拉伸到1280x720
        big_canvas = small_final.resize((self.W, self.H), Image.BILINEAR)

        # 最后加上高斯模糊
        final_img = big_canvas.filter(ImageFilter.GaussianBlur(radius=4))

        return final_img

# ==========================================================================
# 原始Scene类
# ==========================================================================
class RadialLineVortexScene(Scene):
    name = "放射漩涡线条"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.cx = W / 2
        self.cy = H / 2
        self.bands = 120
        self.rot = 0.0
        self.trail_buf = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.decay = 0.86

    def render(self, i, intensity):
        spec = self.ctx["spec"][i]
        canvas = Image.new("RGBA", (self.W, self.H), (0,0,0,0))
        draw = ImageDraw.Draw(canvas)
        self.rot += 0.018

        for k in range(self.bands):
            ang = self.rot + (k / self.bands) * 2 * math.pi
            amp = float(spec[k % len(spec)])
            length = amp * 280
            x2 = self.cx + math.cos(ang) * length
            y2 = self.cy + math.sin(ang) * length
            draw.line([(self.cx, self.cy), (x2, y2)],
                      fill=(120,200,255,int(amp*180)), width=2)

        arr = np.array(canvas, dtype=np.float32)[:,:,:3]
        self.trail_buf *= self.decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)
        return Image.fromarray(self.trail_buf.astype(np.uint8), "RGB")

class WmpExplodeRippleScene(Scene):
    name = "WMP 爆炸裂痕放射波纹-左右错位版"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.W = W
        self.H = H
        self.cx = W / 2
        self.cy = H / 2
        self.rays = 240
        self.trail_buf = np.zeros((H, W, 3), dtype=np.float32)
        self.decay = 0.87
        self.scale_smooth = 1.0
        self.rot = 0.0

        # 左右声道各自的圆心，与画面中心错开
        self.left_offset = -0.065
        self.right_offset = 0.065

    def render(self, i, intensity):
        spec_l = self.ctx["spec_left"][i]
        spec_r = self.ctx["spec_right"][i]
        bass_l = self.ctx["bass_left"][i]
        bass_r = self.ctx["bass_right"][i]
        rms_mix = self.ctx["rms"][i]

        # 整体缓慢自转 + 鼓点冲击
        rot_speed = 0.008
        self.rot += rot_speed * (1 + ((bass_l + bass_r) / 2) * 0.55)

        # 整体缩放，鼓点一响向外炸开
        target_scale = 1.0 + rms_mix * 0.28 + ((bass_l + bass_r) / 2) * 0.36
        self.scale_smooth = self.scale_smooth * 0.91 + target_scale * 0.09

        # 左右声道各自独立圆心，错开位置
        cx_l = self.cx + self.left_offset * min(self.W, self.H)
        cx_r = self.cx + self.right_offset * min(self.W, self.H)

        canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # ========== 左声道：左侧圆心的完整 360° 裂痕描边 ==========
        for rid in range(self.rays):
            base_ang = (rid / self.rays) * 2 * math.pi
            ang = base_ang + self.rot
            amp = spec_l[rid % len(spec_l)]
            base_radius = 95 + amp * 145
            r = base_radius * self.scale_smooth

            x = cx_l + math.cos(ang) * r
            y = self.cy + math.sin(ang) * r

            # 用短线连接相邻点，形成连续裂纹轮廓
            if rid > 0:
                draw.line([prev_x, prev_y, x, y],
                          fill=(10, 175, 32, 215), width=2)
            prev_x, prev_y = x, y

        # ========== 右声道：右侧圆心的完整 360° 裂痕描边 ==========
        for rid in range(self.rays):
            base_ang = (rid / self.rays) * 2 * math.pi
            ang = base_ang + self.rot
            amp = spec_r[rid % len(spec_r)]
            base_radius = 95 + amp * 145
            r = base_radius * self.scale_smooth

            x = cx_r + math.cos(ang) * r
            y = self.cy + math.sin(ang) * r

            if rid > 0:
                draw.line([prev_x_r, prev_y_r, x, y],
                          fill=(14, 188, 40, 205), width=2)
            prev_x_r, prev_y_r = x, y

        arr = np.array(canvas, dtype=np.float32)[:, :, :3]
        self.trail_buf *= self.decay
        self.trail_buf += arr
        self.trail_buf = np.clip(self.trail_buf, 0, 255)

        return Image.fromarray(self.trail_buf.astype(np.uint8))

import cv2

class WmpSCurveScene(Scene):
    name = "WMP S‑螺旋粒子原版逆向复刻"
    def __init__(self, W, H, ctx, params=None):
        super().__init__(W, H, ctx, params)
        self.W = W
        self.H = H
        # 粒子出生逻辑源点：右下角（逻辑原点，粒子从这里生成再飞出去）
        self.spawn_x = self.W * 0.92
        self.spawn_y = self.H * 0.92

        # 左右声道两套完全独立粒子池
        self.particles_L = []
        self.particles_R = []

        # 每帧生成新粒子数量，控制生长速度；初始只有很少点，慢慢变多
        self.spawn_per_frame = 3

    def _spawn_particle(self, amp):
        """生成单个新粒子，出生在右下角spawn点"""
        return {
            "x": self.spawn_x,
            "y": self.spawn_y,
            "ang": 0.0,
            "r": 0.0,
            "speed_r": 2.2 + amp * 4.5,      # 径向向外速度，受频谱幅值
            "speed_rot": 0.022 + amp * 0.03, # 自转角速度
            "life": 1.0,
            "decay": 0.0075,
            "amp": amp
        }

    def _update_particles(self, part_list, bass_val):
        """更新一整套粒子：旋转、径向扩张、生命衰减；过滤死亡粒子"""
        new_list = []
        for p in part_list:
            # 低音扰动速度：鼓点来临时旋转/向外加速
            p["speed_r"] += bass_val * 1.4
            p["speed_rot"] += bass_val * 0.016

            # 更新角度与半径
            p["ang"] += p["speed_rot"]
            p["r"] += p["speed_r"]

            # 实际世界坐标
            p["x"] = self.spawn_x + math.cos(p["ang"]) * p["r"]
            p["y"] = self.spawn_y + math.sin(p["ang"]) * p["r"]

            # 摩擦力阻尼，速度衰减
            p["speed_r"] *= 0.975
            p["speed_rot"] *= 0.978

            # 生命周期衰减
            p["life"] -= p["decay"]
            if p["life"] > 0.0:
                new_list.append(p)
        return new_list

    def _draw_particle_stroke(self, draw, part_list):
        """相邻粒子连线，形成螺旋S曲线；粒子life控制线条透明度、粗细"""
        if len(part_list) < 2:
            return
        prev = part_list[0]
        for p in part_list[1:]:
            alpha = int(255 * p["life"])
            w = max(1, int(1.2 + p["amp"] * 2.8))
            draw.line(
                [(prev["x"], prev["y"]), (p["x"], p["y"])],
                fill=(255, 255, 255, alpha),
                width=w
            )
            prev = p

    def render(self, i, intensity):
        spec_l = self.ctx["spec_left"][i]
        spec_r = self.ctx["spec_right"][i]
        bass_l = self.ctx["bass_left"][i]
        bass_r = self.ctx["bass_right"][i]
        bass_avg = (bass_l + bass_r) * 0.5

        canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # ========== 生成新粒子：左右声道独立采样频谱幅值 ==========
        for _ in range(self.spawn_per_frame):
            idx_l = random.randint(0, len(spec_l)-1)
            amp_l = float(spec_l[idx_l])
            self.particles_L.append(self._spawn_particle(amp_l))

            idx_r = random.randint(0, len(spec_r)-1)
            amp_r = float(spec_r[idx_r])
            self.particles_R.append(self._spawn_particle(amp_r))

        # ========== 更新粒子物理状态，剔除死亡粒子 ==========
        self.particles_L = self._update_particles(self.particles_L, bass_l)
        self.particles_R = self._update_particles(self.particles_R, bass_r)

        # ========== 相邻粒子连线绘制S螺旋曲线 ==========
        self._draw_particle_stroke(draw, self.particles_L)
        self._draw_particle_stroke(draw, self.particles_R)

        return canvas.convert("RGB")

# -------------------【下面这一堆建议直接全部删除，不需要】-------------------
# class VortexScene(Scene):
#     def render(self,i,intensity):
#         return Image.new("RGB",(self.W,self.H),(0,0,0))
#
# class ParticlesScene(Scene):
#     def render(self,i,intensity):
#         return Image.new("RGB",(self.W,self.H),(0,0,0))
#
# class RadialBarsScene(Scene):
#     def render(self,i,intensity):
#         return Image.new("RGB",(self.W,self.H),(0,0,0))
#
# class BarsScene(Scene):
#     name = "竖条"
#     def render(self, i, intensity):
#         return Image.new("RGB", (self.W, self.H), (0,0,0))

class BarsScene(Scene):
    name = "竖条"
    def render(self, i, intensity): return Image.new("RGB", (self.W, self.H), (0,0,0))

# ========== 更新场景列表，把Tolerance加入 ==========
SCENE_CLASSES = [
    WmpToleranceScene,
    WmpClassicAmbienceScene,
    WmpSCurveScene,
    WmpExplodeRippleScene
]

# ==========================================================================
# 音频模块&渲染
# ==========================================================================
def _decode_ffmpeg(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    line = p.stdout.strip().splitlines()[0]; parts = line.split(",")
    sr = int(float(parts[0])); ch = int(float(parts[1]))
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch).copy(), sr

def load_audio(path):
    if not os.path.isfile(path): raise SystemExit(f"找不到文件: {path}")
    return _decode_ffmpeg(path)

def smooth1d(x, k):
    y = np.empty_like(x); cur = 0.0
    for i in range(len(x)): cur += k * (x[i] - cur); y[i] = cur
    return y

def analyze_legacy(stereo, sr, fps, fft_size=4096, bands=128, fmin=30.0, fmax=16000.0):
    n = len(stereo)
    hop = sr / float(fps)
    n_frames = max(1, int(math.ceil(n / hop)))
    fft_size = 1 << int(round(math.log2(fft_size)))
    win = np.hanning(fft_size).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    edges = np.geomspace(max(fmin, 20.0), min(fmax, sr / 2 - 1), bands + 1)

    band_idx = []
    for k in range(bands):
        lo, hi = edges[k], edges[k + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2))])
        band_idx.append(idx)

    spec_left = np.zeros((n_frames, bands), dtype=np.float32)
    spec_right = np.zeros((n_frames, bands), dtype=np.float32)
    rms_left = np.zeros(n_frames, dtype=np.float32)
    rms_right = np.zeros(n_frames, dtype=np.float32)

    pad = fft_size // 2
    win_len = int(sr * 0.035)

    for i in range(n_frames):
        c = int((i + 0.5) * hop)
        a = c - pad
        b = a + fft_size

        seg_l = np.zeros(fft_size, dtype=np.float32)
        seg_r = np.zeros(fft_size, dtype=np.float32)
        if a < 0:
            seg_l[-a:] = stereo[:b, 0]
            seg_r[-a:] = stereo[:b, 1]
        elif b > n:
            seg_l[:n - a] = stereo[a:n, 0]
            seg_r[:n - a] = stereo[a:n, 1]
        else:
            seg_l[:] = stereo[a:b, 0]
            seg_r[:] = stereo[a:b, 1]

        s_l = np.abs(np.fft.rfft(seg_l * win))
        s_r = np.abs(np.fft.rfft(seg_r * win))
        s_l = 20.0 * np.log10(np.maximum(s_l / (fft_size / 4), 1e-10))
        s_r = 20.0 * np.log10(np.maximum(s_r / (fft_size / 4), 1e-10))

        for k, idx in enumerate(band_idx):
            spec_left[i, k] = s_l[idx].max()
            spec_right[i, k] = s_r[idx].max()

        a2 = max(0, c - win_len // 2)
        b2 = min(n, c + win_len // 2)
        if b2 > a2:
            rms_left[i] = np.sqrt(np.mean(stereo[a2:b2, 0] ** 2))
            rms_right[i] = np.sqrt(np.mean(stereo[a2:b2, 1] ** 2))

    norm_left = np.clip((spec_left + 90.0) / 80.0, 0, 1)
    norm_right = np.clip((spec_right + 90.0) / 80.0, 0, 1)

    rms_db_l = 20 * np.log10(np.maximum(rms_left, 1e-10))
    rms_db_r = 20 * np.log10(np.maximum(rms_right, 1e-10))
    rms_norm_l = np.clip((rms_db_l + 60.0) / 60.0, 0, 1)
    rms_norm_r = np.clip((rms_db_r + 60.0) / 60.0, 0, 1)

    nb = max(1, bands // 8)
    bass_l = norm_left[:, :nb].mean(axis=1)
    bass_r = norm_right[:, :nb].mean(axis=1)

    # 混合声道，兼容原来旧场景
    spec_mix = (norm_left + norm_right) / 2.0
    rms_mix = (rms_norm_l + rms_norm_r) / 2.0
    bass_mix = (bass_l + bass_r) / 2.0
    treble_mix = (norm_left[:, -nb:].mean(axis=1) + norm_right[:, -nb:].mean(axis=1)) / 2.0

    rms_s = smooth1d(rms_mix, 0.15)
    novelty = np.abs(np.diff(rms_s, prepend=rms_s[0]))
    novelty = smooth1d(novelty, 0.25)
    novelty = novelty / (novelty.max() + 1e-9)

    return dict(
        spec=spec_mix,
        rms=rms_mix,
        bass=bass_mix,
        treble=treble_mix,
        novelty=novelty,
        spec_left=norm_left,
        spec_right=norm_right,
        bass_left=bass_l,
        bass_right=bass_r,
        rms_left=rms_norm_l,
        rms_right=rms_norm_r,
        n_frames=n_frames,
        bands=bands,
        sr=sr,
        fps=fps
    )


def build_timeline(ctx, min_dur, max_dur, scene_cls_list=None, seed=0):
    rng = np.random.default_rng(seed); fps = ctx["fps"]; n = ctx["n_frames"]; nov = ctx["novelty"]
    min_f = int(min_dur * fps); max_f = int(max_dur * fps); timeline = []; i = 0; last_scene = -1
    if scene_cls_list is None: scene_cls_list = SCENE_CLASSES
    scene_idx = int(rng.integers(0, len(scene_cls_list)))
    while i < n:
        end_min = min(n, i + min_f); end_max = min(n, i + max_f)
        if end_max <= end_min: timeline.append((i, scene_idx)); break
        window = nov[end_min:end_max]; peak = int(np.argmax(window)) + end_min if len(window) > 0 else end_max
        timeline.append((i, scene_idx)); i = peak
        choices = [k for k in range(len(scene_cls_list)) if k != last_scene]; scene_idx = int(rng.choice(choices)); last_scene = scene_idx
    return timeline

def render(ctx, timeline, cfg, scene_list, out_path):
    W = cfg["width"] // 2 * 2; H = cfg["height"] // 2 * 2; fps = ctx["fps"]; n = ctx["n_frames"]
    fade_frames = int(cfg["fade"] * fps)
    scenes = [cls(W, H, ctx) for cls in scene_list] if inspect.isclass(scene_list[0]) else scene_list
    scene_of = np.zeros(n, dtype=np.int32)
    for k, (start, sidx) in enumerate(timeline):
        end = timeline[k + 1][0] if (k + 1 < len(timeline)) else n
        scene_of[start:end] = sidx
    intensity = np.clip(0.5 * ctx["rms"] + 0.5 * ctx["novelty"], 0, 1); intensity = smooth1d(intensity, 0.2)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", cfg["input"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE); report = max(1, n // 100); prev_scene = -1; prev_frame = None; fade_left = 0
    try:
        for i in range(n):
            sidx = int(scene_of[i]); cur = scenes[sidx].render(i, float(intensity[i]))
            if sidx != prev_scene: fade_left = fade_frames; prev_scene = sidx
            if fade_left > 0 and prev_frame is not None:
                a = 1.0 - fade_left / max(fade_frames, 1); cur = Image.blend(prev_frame, cur, a); fade_left -= 1
            proc.stdin.write(cur.tobytes("raw", "RGB")); prev_frame = cur
            if i % report == 0: print(f"\r  渲染 {i * 100 // n:3d}%", end="", file=sys.stderr, flush=True)
    finally:
        try: proc.stdin.close()
        except: pass
        proc.wait()
    print("\r渲染完成 ✔", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--fps", type=int, default=60); ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-dur", type=float, default=6.0); ap.add_argument("--max-dur", type=float, default=14.0); ap.add_argument("--fade", type=float, default=0.6); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, sr = load_audio(a.input)
    ctx = analyze_legacy(data, sr, a.fps)
    out_path = a.output or "output_wmp.mp4"
    cfg = dict(input=a.input, fps=a.fps, width=a.width, height=a.height, fade=a.fade)

    timeline = build_timeline(ctx, a.min_dur, a.max_dur, seed=a.seed)
    render(ctx, timeline, cfg, SCENE_CLASSES, out_path)

if __name__ == "__main__":
    main()


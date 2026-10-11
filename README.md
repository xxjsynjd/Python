# Python 音频可视化工具集

在 Android 平板上用 Termux + Python 手搓的一套音频可视化工具。
不依赖 GPU，不依赖 X11，不用电脑，手机就能跑。

输入一首歌，输出一整套可视化的 MP4。

---

## 包含的工具

### 核心：音频转视频

| 脚本 | 功能 | 风格 |
|---|---|---|
| `Spectrum3DAudacious.py` | 3D 频谱墙 | Audacious gl-spectrum 风格 |
| `Spectrum_2d.py` | 2D 频谱竖条 | 经典 FFT 频谱 |
| `WaterfallWMP.py` | 瀑布图 | Windows Media Player XP 风格 |
| `Waterfall3D.py` | 3D 瀑布 | 立体声频谱墙 |
| `VU_Meter.py` | 电平表 | 带回落 + 峰值保持 |
| `LoudMeter.py` | 响度计 | BS.1770 LUFS |
| `ThrustMeter.py` | 推力计 | 低频冲击感 |
| `VibrationMeter.py` | 震力计 | 低频震动波纹 |
| `NoiseGateMeter.py` | 噪声门 | 开关指示灯 |
| `PhaseMeter.py` | 相位表 | 立体声利萨茹图形 |
| `SpeakerMeter.py` | 低音炮 + 高音喇叭 | 分频驱动双喇叭 |
| `SpeakerPhysics.py` | 纸盆物理仪 | 二阶微分方程 + 相空间 |
| `ProgressBar.py` | 播放进度条 | 纯横条 |
| `TimelineRoll.py` | 滚动时间轴 | 波形 + 刻度 + BPM 小节线 |
| `PitchMeter1.py` | 音符计 | YIN 音高检测 + 滑音连音 |
| `WMPVisualizer.py` | 随机场景可视化 | 瀑布 / 漩涡 / 粒子 / 圆环 / 竖条 |

### AI 歌词逐字对齐

| 脚本 | 功能 |
|---|---|
| `LyricsDisplayAI.py` | 读取 LRC，用 Whisper 逐字对齐，唱到哪弹哪 |
| `GenerateAll.py` | 一首歌，一键跑全套 |

---

## 安装

### Termux（Android）

```bash
pkg install python ffmpeg
pip install numpy pillow soundfile
附加：AI 歌词对齐

```bash
pip install faster-whisper ctranslate2 tokenizers huggingface-hub
```

注意：Termux 上 ctranslate2 需要自己编译（源码里带 pthread_setaffinity_np 的坑，需要手动 patch）。
不想折腾的话，LyricsDisplayAI.py 会自动降级为整句均分模式。

附加：AI 人声分离（Demucs）

```bash
pip install demucs
```

第一次跑会自动下载模型（约 80 MB），缓存到 .vocals_cache/，之后秒读。

---

用法

单个工具

```bash
# 3D 频谱
Spectrum3DAudacious.py 歌.mp3 -o out.mp4

# 音符计（带 AI 人声分离）
PitchMeter1.py 歌.mp3 -o out.mp4 --auto-vocals

# 歌词逐字弹跳
LyricsDisplayAI.py 歌.mp3 -o out.mp4 --auto-vocals --model base --lang zh
```

一键全套

```bash
GenerateAll.py 歌.mp3
```

输出到 歌_videos/ 目录，17 个仪表全在里面。

只跑其中几个

```bash
GenerateAll.py 歌.mp3 --modes spectrum3d,pitch,lyrics
```

---

常用参数
参数 | 说明 | 默认
--width / --height | 分辨率 | 1280x720
--fps 帧率 60
--auto-vocals 用 Demucs 分离人声 关
--cache-dir 人声缓存目录 .vocals_cache
--model Whisper 模型（AI 歌词用） base
--lang 语言 zh
--lrc 手动指定 LRC 文件 —
--font 中文字体路径 自动查找
参数 说明 默认
--width / --height 分辨率 1280x720
--fps 帧率 60
--auto-vocals 用 Demucs 分离人声 关
--cache-dir 人声缓存目录 .vocals_cache
--model Whisper 模型（AI 歌词用） base
--lang 语言 zh
--lrc 手动指定 LRC 文件 —
--font 中文字体路径 自动查找
更多参数用 --help 查看。

---

3D 频谱参数速查

```bash
Spectrum3DAudacious.py 歌.mp3 -o out.mp4 \
  --yaw-offset 180 \
  --osc-angle 40 \
  --hue-offset 0.375 \
  --flip-freq \
  --pitch 35 \
  --width 720 --height 1280 --fps 30
```

---

依赖说明

依赖 用途
numpy FFT、RMS、YIN
Pillow 逐帧绘制
soundfile 读音频
ffmpeg 解码音频 + 编码视频
faster-whisper + ctranslate2 AI 歌词逐字对齐
demucs AI 人声分离

---

已知坑

· 内存：AI 歌词对齐 + 4K 渲染时会吃 2GB 左右内存，手机容易 OOM
· 硬盘：批量跑的时候注意空间，视频和缓存会迅速占满
· ctranslate2：Termux 仓库里的版本阉割了 Whisper 模型，需要自己从源码编译
· 字体：不指定 --font 时脚本会自动找系统中文字体，找不到会显示方块

---

许可证

MIT License

---

说明

所有可视化都是离线逐帧渲染，不是录屏。
同一个音频，可以输出电平表、频谱、瀑布、音符计、歌词等等十几种形态。

这不是播放器，是一个把音频变成视频的工厂。

电话：13956275846/19251335846
V：xxjsynjd (信息技术疑难解答)
Q：3550526842
***音：信息技术疑难解答
***手：信息技术疑难解答
***站：信息技术疑难解答
小***书：信息技术疑难解答
***博：信息技术疑难解答
***度：信息技术疑难解答
今***条：信息技术疑难解答
西***视：信息技术疑难解答
抖***火山版：信息技术疑难解答
虎牙直播(绝版)：信息技术疑难解答
斗鱼：信息技术疑难解答
微信视频号/公众号：信息技术疑难解答
QQ频道号：疑难解答访问台、Termux交流社区

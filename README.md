# Python音频可视化工具集

在Android平板上用Termux+Python手搓的一套音频可视化工具。
不依赖GPU，不依赖X11，不用电脑，手机就能跑。

输入一首歌，输出一整套可视化的MP4。

---

## 包含的工具

### 核心：音频转视频

| 脚本 | 功能 | 风格 |
|---|---|---|
| `Spectrum3DAudacious.py` | 3D频谱墙 | Audacious gl-spectrum风格 |
| `Spectrum_2d.py` | 2D频谱竖条 | 经典FFT频谱 |
| `WMPVisualizer.py` | 瀑布图 | Windows Media Player XP风格 |
| `VU_Meter.py` | 电平表 | 带回落+峰值保持 |
| `LoudMeter.py` | 响度计 | BS.1770 LUFS |
| `ThrustMeter.py` | 推力计 | 低频冲击感 |
| `VibrationMeter.py` | 震力计 | 低频震动波纹 |
| `NoiseGateMeter.py` | 噪声门 | 开关指示灯 |
| `SpeakerMeter.py` | 低音+高音喇叭 | 分频驱动双喇叭 |
| `ProgressBar.py` | 播放进度条 | 纯横条 |
| `TimelineRoll.py` | 滚动时间轴 | 波形+刻度线 |
| `PitchMeter1.py` | 音符计 | YIN音高检测+滑音连音+BPM小节 |
| `WMPVisualizer.py` | 随机场景可视化 | 瀑布/漩涡/粒子/圆环/竖条 |

### AI 歌词逐字对齐

| 脚本 | 功能 |
|---|---|
| `LyricsDisplayAI.py` | 读取LRC，用Whisper逐字对齐，唱到哪弹哪 |
| `GenerateAll.py` | 一首歌，一键跑全套 |

---

## 安装

### Termux(Android)

```bash
pkg install python ffmpeg
pip install numpy pillow soundfile
dpkg -i *.deb # 或者：dpkg -i [要安装的包的文件名]
```

```bash
cp -v Python/*.py $PREFIX/bin
```
附加：AI歌词对齐

```bash
pip install faster-whisper tokenizers huggingface-hub
```

注意：Termux上ctranslate2需要自己编译(源码里带pthread_setaffinity_np的坑，需要手动patch)。
不想折腾的话，LyricsDisplayAI.py会自动降级为整句均分模式。

附加：AI人声分离(Demucs)

```bash
pip install demucs
pip install --force-reinstall xet-core/hf_xet/target/wheels/hf_xet-1.7.0-cp314-cp314-android_24_arm64_v8a.whl
```

第一次跑会自动下载模型(约80MB)，缓存到.vocals_cache/，之后秒读。

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

输出到"歌_videos/"目录，17个仪表全在里面。

只跑其中几个

```bash
GenerateAll.py 歌.mp3 --modes spectrum3d,pitch,lyrics
```

---

常用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| --width / --height | 分辨率 | 1280x720 |
| --fps | 帧率 | 60 |
| --auto-vocals | 用Demucs分离人声 | 关 |
| --cache-dir | 人声缓存目录 | .vocals_cache |
| --model Whisper | 模型(AI歌词用) | base |
| --lang | 语言 | zh |
| --lrc | 手动指定LRC文件 | — |
| --font |中文字体路径 |自动查找 |
更多参数用--help查看。

---

3D频谱参数速查

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

| 依赖 | 用途 |
|---|---|
| numpy | FFT、RMS、YIN |
| Pillow | 逐帧绘制 |
| soundfile | 读音频 |
| ffmpeg | 解码音频+编码视频 |
| faster-whisper+ctranslate2 | AI歌词逐字对齐 |
| demucs | AI人声分离 |

---

已知坑

· 内存：AI歌词对齐+4K渲染时会吃2GB左右内存，手机容易OOM(Out of Memory Killer，通常在低内存情况下出手，如果你的Termux正好在后台，那么你下一秒进Termux看到的一幕很有可能就是Termux启动画面，这个过程也叫冷启动，通常是进程被OOM Killer强制终止过后的点进去的一个画面，通常如果更恶劣的情况，会直接导致你的手机冷重启，直接导致你的手机冷重启的这玩意儿也叫内核恐慌。)

· 硬盘：批量跑的时候注意空间，视频和缓存会迅速占满

· ctranslate2：Termux仓库里的版本阉割了Whisper模型，需要自己从源码编译

· 字体：不指定--font时脚本会自动找系统中文字体，找不到会显示方块

那么如何判断Termux是否在你切换到后台的时候被杀掉了呢？其实有一点非常好判断：Termux在运行的过程当中会有一条常驻通知，长这样：

```text
>_ Termux
1 session

 [ Exit ] [ Acquire wakelock ]
```

如果你看到这条通知没了，那么就是被杀掉了，Android在退出应用的时候会顺带清理该应用的全部通知。

##快速开始(懒人版)

1.安装Termux:

从F-Droid下载(推荐):
https://f-droid.org/packages/com.termux/

或从GitHubReleases:
https://github.com/termux/termux-app/releases

2.打开Termux,安装基础环境:

```bash
   pkg update && pkg upgrade
   pkg install python ffmpeg mpv
   pip install numpy pillow soundfile
```

3.克隆本项目:

```bash
   git clone https://github.com/xxjsynjd/Python
   cd Python
```

4.安装hf_xet修复版(Termux+Python3.14必装):

```bash
   pip install hf_xet-1.7.0-cp314-cp314-android_24_arm64_v8a.whl
```

5.试跑:

```bash
   ./LoudMeter.py 你的歌.flac -o 输出.mp4
```

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

(下面均为粉丝最多的)

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

作者：沈诗淳

学校：江苏商贸职业学院-人工智能学院系部-计算机网络[2503]

下面这些东西我告诉你们：不好用、病毒、盗版这些的，我直接死妈得了！(除非有的确实是没上传或者爆内存，诈骗电话勿扰。)

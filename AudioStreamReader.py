# 预留模块：当前版本暂不需要流式读取器，音频数据由 MultiTrackRender 统一处理。
class AudioStreamReader:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("当前版本未启用 AudioStreamReader，请使用 MultiTrackRender 内置的音频处理方式。")

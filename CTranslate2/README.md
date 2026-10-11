[![CI](https://github.com/OpenNMT/CTranslate2/workflows/CI/badge.svg)](https://github.com/OpenNMT/CTranslate2/actions?query=workflow%3ACI) [![PyPI version](https://badge.fury.io/py/ctranslate2.svg)](https://badge.fury.io/py/ctranslate2) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/CTranslate2/) [![Gitter](https://badges.gitter.im/OpenNMT/CTranslate2.svg)](https://gitter.im/OpenNMT/CTranslate2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Forum](https://img.shields.io/discourse/status?server=https%3A%2F%2Fforum.opennmt.net%2F)](https://forum.opennmt.net/)

# CTranslate2

CTranslate2 是一个用于 Transformer 模型高效推理的 C++ 和 Python 库。

该项目实现了一个自定义运行时，应用了许多性能优化技术，如权重量化、层融合、批处理重排序等，从而在 CPU 和 GPU 上[加速并减少 Transformer 模型的内存占用](#基准测试)。

目前支持以下模型类型：

* 编码器-解码器模型：Transformer base/big, M2M-100, NLLB, BART, mBART, Pegasus, T5, Whisper, T5Gemma, T5Gemma2, MADLAD-400
* 仅解码器模型：GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, Llama, Mistral, Gemma, CodeGen, GPTBigCode, Falcon, Qwen2
* 仅编码器模型：BERT, DistilBERT, XLM-RoBERTa

兼容的模型需要先转换为优化的模型格式。该库包含针对多种框架的转换器：

* [OpenNMT-py](https://opennmt.net/CTranslate2/guides/opennmt_py.html)
* [OpenNMT-tf](https://opennmt.net/CTranslate2/guides/opennmt_tf.html)
* [Fairseq](https://opennmt.net/CTranslate2/guides/fairseq.html)
* [Marian](https://opennmt.net/CTranslate2/guides/marian.html)
* [OPUS-MT](https://opennmt.net/CTranslate2/guides/opus_mt.html)
* [Transformers](https://opennmt.net/CTranslate2/guides/transformers.html)

该项目面向生产环境，并提供[向后兼容性保证](https://opennmt.net/CTranslate2/versioning.html)，但也包含与模型压缩和推理加速相关的实验性功能。

## 主要特性

* **在 CPU 和 GPU 上快速高效地执行**<br/>得益于许多高级优化（层融合、填充移除、批处理重排序、原地操作、缓存机制等），在执行支持的模型和任务时，其[执行速度显著更快，所需资源也更少](#基准测试)，优于通用深度学习框架。
* **量化和降低精度**<br/>模型序列化和计算支持[降低精度](https://opennmt.net/CTranslate2/quantization.html)的权重：16 位浮点数 (FP16)、16 位脑浮点数 (BF16)、16 位整数 (INT16)、8 位整数 (INT8) 和 AWQ 量化 (INT4)。
* **支持多种 CPU 架构**<br/>该项目支持 x86-64 和 AArch64/ARM64 处理器，并集成了针对这些平台优化的多个后端：[Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)、[oneDNN](https://github.com/oneapi-src/oneDNN)、[OpenBLAS](https://www.openblas.net/)、[Ruy](https://github.com/google/ruy) 和 [Apple Accelerate](https://developer.apple.com/documentation/accelerate)。
* **自动 CPU 检测和代码分发**<br/>一个二进制文件可以包含多个后端（例如 Intel MKL 和 oneDNN）和指令集架构（例如 AVX、AVX2），并在运行时根据 CPU 信息自动选择。
* **并行和异步执行**<br/>可以使用多个 GPU 或 CPU 核心并行和异步地处理多个批次。
* **动态内存使用**<br/>得益于 CPU 和 GPU 上的缓存分配器，内存使用量会根据请求大小动态变化，同时仍能满足性能要求。
* **磁盘占用小**<br/>量化可以使模型在磁盘上缩小 4 倍，而精度损失极小。
* **易于集成**<br/>该项目依赖项很少，并公开了简单的 [Python](https://opennmt.net/CTranslate2/python/overview.html) 和 C++ API，以满足大多数集成需求。
* **可配置和交互式解码**<br/>[高级解码功能](https://opennmt.net/CTranslate2/decoding.html)允许自动补全部分序列，并在序列的特定位置返回备选方案。
* **支持张量并行以进行分布式推理**<br/>非常大的模型可以拆分到多个 GPU 上。按照此[文档](docs/parallel.md#model-and-tensor-parallelism)设置所需的环境。

其中一些特性很难用标准深度学习框架实现，这也是该项目的动机所在。

## 安装和使用

CTranslate2 可以通过 pip 安装：

```bash
pip install ctranslate2
```

Python 模块用于转换模型，只需几行代码即可翻译或生成文本：

```python
translator = ctranslate2.Translator(translation_model_path)
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
```

有关更多信息和示例，请参阅文档。

如果您有 AMD ROCm GPU，我们在发布页面上提供了特定的 Python 轮子文件。

Web 服务器

ctranslate2-web-server 是一个构建在 CTranslate2 之上的 Web 服务器，它暴露了一个兼容 OpenAI 的 REST API，使得将 CTranslate2 模型集成到已支持 OpenAI API 的应用程序中变得容易。

基准测试

我们使用多个模型翻译 En->De 测试集 newstest2014：

· OpenNMT-tf WMT14：在 WMT14 数据集（450 万行）上使用 OpenNMT-tf 训练的基础 Transformer
· OpenNMT-py WMT14：在 WMT14 数据集（450 万行）上使用 OpenNMT-py 训练的基础 Transformer
· OPUS-MT：在 2020-02-26 可用的所有 OPUS 数据（8190 万行）上使用 Marian 训练的基础 Transformer

基准测试报告了每秒生成的目标 token 数量（越高越好）。结果是在多次运行中汇总得出的。有关更多详细信息并重现这些数字，请参阅基准测试脚本。

请注意，下面呈现的结果仅对本次基准测试中使用的配置有效：绝对性能和相对性能可能会因不同的设置而变化。

CPU

 Tokens per second Max. memory BLEU
OpenNMT-tf WMT14 model   
OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) 209.2 2653MB 26.93
OpenNMT-py WMT14 model   
OpenNMT-py 3.0.4 (with PyTorch 1.13.1) 275.8 2012MB 26.77
- int8 323.3 1359MB 26.72
CTranslate2 3.6.0 658.8 849MB 26.77
- int16 733.0 672MB 26.82
- int8 860.2 529MB 26.78
- int8 + vmap 1126.2 598MB 26.64
OPUS-MT model   
Transformers 4.26.1 (with PyTorch 1.13.1) 147.3 2332MB 27.90
Marian 1.11.0 344.5 7605MB 27.93
- int16 330.2 5901MB 27.65
- int8 355.8 4763MB 27.27
CTranslate2 3.6.0 525.0 721MB 27.92
- int16 596.1 660MB 27.53
- int8 696.1 516MB 27.65

在配备 Intel(R) Xeon(R) Platinum 8275CL CPU 的 c5.2xlarge Amazon EC2 实例上使用 4 个线程执行。

GPU

 Tokens per second Max. GPU memory Max. CPU memory BLEU
OpenNMT-tf WMT14 model    
OpenNMT-tf 2.31.0 (with TensorFlow 2.11.0) 1483.5 3031MB 3122MB 26.94
OpenNMT-py WMT14 model    
OpenNMT-py 3.0.4 (with PyTorch 1.13.1) 1795.2 2973MB 3099MB 26.77
FasterTransformer 5.3 6979.0 2402MB 1131MB 26.77
- float16 8592.5 1360MB 1135MB 26.80
CTranslate2 3.6.0 6634.7 1261MB 953MB 26.77
- int8 8567.2 1005MB 807MB 26.85
- float16 10990.7 941MB 807MB 26.77
- int8 + float16 8725.4 813MB 800MB 26.83
OPUS-MT model    
Transformers 4.26.1 (with PyTorch 1.13.1) 1022.9 4097MB 2109MB 27.90
Marian 1.11.0 3241.0 3381MB 2156MB 27.92
- float16 3962.4 3239MB 1976MB 27.94
CTranslate2 3.6.0 5876.4 1197MB 754MB 27.92
- int8 7521.9 1005MB 792MB 27.79
- float16 9296.7 909MB 814MB 27.90
- int8 + float16 8362.7 813MB 766MB 27.90

在配备 NVIDIA A10G GPU（驱动版本：510.47.03）的 g5.xlarge Amazon EC2 实例上使用 CUDA 11 执行。

贡献

CTranslate2 是一个社区驱动的项目。我们欢迎各种形式的贡献：

· 新模型支持： 帮助我们实现更多的 Transformer 架构。
· 性能： 为 CPU 或 GPU 内核提出优化建议。
· 错误报告： 如果发现任何不符合预期的问题，请开启一个 issue。
· 文档： 改进我们的指南或添加新示例。

查看我们的贡献指南以了解如何设置您的开发环境。

额外资源

· 文档
· 论坛
· Gitter

```

**一点小建议：**
你可以把这段翻译后的内容存为 `README_zh.md`，放在你 GitHub 仓库的 `CTranslate2` 文件夹里，然后可以在主目录的 `README.md` 里加一句：“*关于 CTranslate2 的详细说明，请查看 [CTranslate2/README_zh.md]*”，这样不仅显得你极其专业，还能让中文用户快速了解你的底层技术依赖。

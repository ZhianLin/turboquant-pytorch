# TurboQuant

基于 PyTorch 从零实现的 [TurboQuant](https://link.gitcode.com/?target=https%3A%2F%2Farxiv.org%2Fabs%2F2504.19874&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)（ICLR 2026），这是谷歌提出的用于压缩 LLM 键值缓存的向量量化算法。已在 Windows 系统搭配 NVIDIA 显卡环境下测试通过。

我们完整复现了论文中的算法，却发现其核心创新点（QJL）在实际应用中反而产生负面影响。基于 8 个以上独立社区实现的经验总结，我们构建了一个改进版本（V3）。

> **更正（2026-03-30）：** 本 README 早期版本曾声称“在 5 倍压缩下实现 18/18 完美生成”。该结论源于一项[存在缺陷的测试](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Ftonbistudio%2Fturboquant-pytorch%2Fissues%2F14&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)，其中 `residual_window=0` 的设置导致实际未进行任何压缩。以下为更正后的结果。感谢 [@barbel-bb](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fbarbel-bb&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white) 发现此问题。

## 结果

### V3：生成测试（真实考验——模型能否生成正确文本？）

我们在一份长文档中隐藏了一个事实（“The secret project code name is AURORA-7749”），并要求模型找出该事实。结果已**验证实际压缩效果**（已记录压缩后的 token 数量）：

| 配置             | 2K 上下文               | 4K 上下文 | 压缩率（2K） |
| ---------------- | ----------------------- | --------- | ------------ |
| FP16（基准）     | EXACT                   | EXACT     | 1.0x         |
| **K6/V4 rw=128** | **EXACT**               | **EXACT** | **~2x**      |
| **K8/V4 rw=128** | **EXACT**               | **EXACT** | **~1.6x**    |
| K4/V4 rw=128     | PARTIAL（"AURORA7749"） | MISS      | ~3x          |
| K4/V4 rw=0       | MISS                    | MISS      | ~3.4x        |
| K4/V2 rw=0       | MISS                    | MISS      | ~5x          |

“EXACT”表示输出包含“AURORA-7749”。“PARTIAL”表示同时包含“AURORA”和“7749”，但并非精确字符串。“rw”即 residual window（保留为 fp16 格式的近期 token）。

**有效方案**：K6/V4 搭配 128-token 的 fp16 窗口，在两种上下文长度下均能实现 ~2 倍的实际压缩率和完美输出。使用 4 位键（K4）时，模型在短上下文下能找到关键信息，但会轻微篡改（丢失连字符）。使用 3 位键（K3）时，生成功能完全失效。

**无效方案**：无残差窗口的 3-4 位压缩会产生无意义输出，与 V2 版本表现一致。高注意力分数相似度（99.5%+）并不能保证生成功能正常。

### V3：注意力分数准确性（8K上下文）

以下结果真实有效——它们直接在捕获的KV张量上测试压缩效果，而非通过V3Cache：

| 配置               | 压缩率 | 余弦相似度 | Top-1匹配率 | Top-5匹配率 |
| ------------------ | ------ | ---------- | ----------- | ----------- |
| V3 K4/V2           | 5.1x   | **0.9996** | **94%**     | **97%**     |
| V3 K4/V2 + 保护层  | 3.6x   | **0.9997** | **99%**     | **100%**    |
| V2 3-bit (MSE+QJL) | 5.0x   | 0.9945     | 86%         | 94%         |
| V2 4-bit (MSE+QJL) | 3.8x   | 0.9983     | 86%         | 96%         |

V3通过去除QJL获得了比V2更高的注意力分数准确性。然而，仅凭高注意力分数并不能保证文本生成正常工作（见上文）。

## 什么是K4/V2？

KV缓存存储两种类型的向量：**键（Keys，K）** 和**值（Values，V）**。

- **键（Keys）** 决定模型关注哪些词——这需要精度
- **值（Values）** 是被平均的内容——误差会自然抵消

**K4/V2** 表示键使用4位，值使用2位。平均下来是3位——与统一3位相同——但分配在更重要的地方。这在相同的位预算下比统一分配产生了显著更好的结果。

## 工作原理

### 核心：随机旋转 + 劳埃德-马克斯量化（Lloyd-Max Quantization）

每个向量乘以一个随机正交矩阵，使每个坐标遵循可预测的钟形曲线分布。然后，我们对每个坐标独立应用**最优标量量化器**（Lloyd-Max），四舍五入到最近的预计算质心。

量化步骤：归一化、旋转、对每个坐标取整、存储索引和范数。 反量化步骤：查找质心、反转旋转、恢复范数。

### 关于QJL？（论文的第二阶段）

该论文增加了第二阶段：QJL残差校正，它存储1位符号信息以使内积估计在数学上无偏。我们将此实现为V2。

**它不适用于KV缓存。** 六个独立团队已证实这一点：

- QJL对于原始内积是无偏的，但注意力会通过**softmax**处理分数
- Softmax会指数级放大方差——QJL的随机噪声被放大
- 仅MSE（均方误差）内积有偏但方差更低——方差更低在经过softmax后更有利
- scos-lab测量显示，在GPT-2上使用QJL时误差增加+300%，不使用时增加+7.6%
- 我们的带QJL的V2：27次生成测试全部失败。不带QJL的V3：18次生成测试全部通过。

QJL确实适用于**向量搜索**（无softmax），这是该论文的另一个用例。它也可能适用于非softmax注意力（sigmoid、线性、门控）。

### V3 版本改进（基于社区反馈）

1. **仅保留 MSE** — 移除 QJL，所有比特用于提升重建质量（`compressors_v3.py → MSECompressor`）
2. **非对称 K/V** — 键（Keys）分配的比特数多于值（Values）（`TurboQuantV3(key_bits=4, value_bits=2)`）
3. **比特打包存储** — 实现真实压缩比，而非理论值。V2 版本存储的张量比未压缩时大 38%。（`MSECompressor.compress()` 使用比特移位）
4. **层自适应** — 为敏感的第一层/最后几层分配更多比特以提供保护（`TurboQuantV3(protected_layers=4)`）

## 快速开始

### 环境要求

- Python 3.10 及以上版本
- 支持 CUDA 的 NVIDIA GPU（已在 RTX 3060、12GB 显存上测试通过）
- Windows 11（Linux 系统同样适用）

```bash
pip install -r requirements.txt
```

针对 CUDA PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 运行生成测试（V3 — 推荐）

测试模型在使用压缩KV缓存时是否能实际生成正确文本：

```bash
python -m turboquant.generation_test
```

首次运行会下载 Qwen2.5-3B-Instruct（约 2GB）。测试不同上下文长度下的多种配置。

### 运行注意力验证（V3 与 V2）

并排比较 V2 和 V3 注意力分数的准确性：

```bash
python -m turboquant.validate_v3
```

### 运行合成测试（无需模型）

根据论文中的理论边界验证核心算法：

```bash
python -m turboquant.test_turboquant
```

### 运行原始 V2 验证

原始注意力分数对比（不包含生成过程）：

```bash
python -m turboquant.validate
```

## 项目结构

```
turboquant/
  __init__.py           # Package exports
  lloyd_max.py          # Lloyd-Max optimal scalar quantizer solver
  turboquant.py         # Core TurboQuant: TurboQuantMSE, TurboQuantProd (V2 with QJL)
  compressors.py        # V2 compressors (MSE+QJL for keys, MSE-only for values)
  compressors_v3.py     # V3 compressors (MSE-only, asymmetric K/V, bit-packed, layer-adaptive)
  test_turboquant.py    # Synthetic algorithm tests
  validate.py           # V2 real model attention comparison
  validate_v3.py        # V3 vs V2 comparison
  generation_test.py    # V3 actual text generation test
  requirements.txt
```

### 核心类

**`MSECompressor`**（`compressors_v3.py`）—— 具有位打包存储的单阶段压缩器。同时用于键（keys）和值（values）。是V3版本的核心构建模块。

**`TurboQuantV3`**（`compressors_v3.py`）—— 协调器，可创建具有不同位宽的独立键/值压缩器，并处理层自适应精度。

**`TurboQuantMSE`**（`turboquant.py`）—— 原始的第一阶段量化器。仍用于综合测试。

**`TurboQuantProd`**（`turboquant.py`）—— 原始的带QJL的两阶段量化器。保留用于参考以及QJL能正确工作的综合测试。

## 综合测试结果

核心旋转 + Lloyd-Max算法已根据论文的理论边界进行验证：

**MSE失真**（d=128，1000个随机单位向量）：

| 位数 | 实测MSE | 论文上界 | 比率   |
| ---- | ------- | -------- | ------ |
| 1位  | 0.362   | 0.680    | 0.53倍 |
| 2位  | 0.116   | 0.170    | 0.68倍 |
| 3位  | 0.034   | 0.043    | 0.81倍 |
| 4位  | 0.009   | 0.011    | 0.87倍 |

**大海捞针**（综合向量）：在所有位宽和序列长度上均实现9/9精确检索。

## 参考文献

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://link.gitcode.com/?target=https%3A%2F%2Farxiv.org%2Fabs%2F2504.19874&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)（ICLR 2026）
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://link.gitcode.com/?target=https%3A%2F%2Farxiv.org%2Fabs%2F2406.03482&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://link.gitcode.com/?target=https%3A%2F%2Farxiv.org%2Fabs%2F2502.02617&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)
- [QJL参考实现](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Famirzandieh%2FQJL&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)
- [PolarQuant参考实现](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fericshwu%2FPolarQuant&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)

## 社区贡献

多位社区成员对本实现进行了扩展，并取得了有价值的发现：

- **[scos-lab/turboquant](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fscos-lab%2Fturboquant&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— 8模型基准测试表明，K/V范数比可预测压缩质量。发现仅使用MSE在注意力方面的表现优于MSE+QJL（softmax会放大QJL方差）。异常值感知混合精度在Qwen2.5-1.5B上实现3.6位平均精度，同时PPL提升+2.1%。
- **[0xSero/turboquant](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2F0xSero%2Fturboquant&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— Triton内核 + vLLM集成。支持非对称K/V位宽的生产部署。已在RTX 5090和8x RTX 3090上测试。
- **[back2matching/turboquant](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Fback2matching%2Fturboquant&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— 可通过pip安装，可直接替换HuggingFace生成模块。支持残差窗口（最近的令牌使用fp16）。
- **[TheTom/turboquant_plus](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2FTheTom%2Fturboquant_plus&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— 层自适应压缩，注意力门控V解码（解码速度提升+22.8%）。针对Apple Silicon优化。
- **[RecursiveIntell/turbo-quant](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2FRecursiveIntell%2Fturbo-quant&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— TurboQuant + PolarQuant + QJL的Rust实现。支持零拷贝和流式处理。
- **[SCJedi/entropy-adaptive-kv-cache](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2FSCJedi%2Fentropy-adaptive-kv-cache&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white)**—— 将TurboQuant与熵自适应令牌驱逐相结合，在Qwen3.5-4B上实现12倍压缩且无质量损失。

### 社区主要发现

- **仅使用 MSE 在注意力任务上优于 MSE+QJL** ([#10](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Ftonbistudio%2Fturboquant-pytorch%2Fissues%2F10&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white), [#8](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Ftonbistudio%2Fturboquant-pytorch%2Fissues%2F8&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white))——经 6 个以上独立团队通过 Python、C 和 Rust 实现验证。
- **在相似压缩比下 Q4_0 优于 TurboQuant** ([#6](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Ftonbistudio%2Fturboquant-pytorch%2Fissues%2F6&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white))——TurboQuant 的优势体现在更高压缩率（3 位/5 倍及以下）场景，而块方法无法达到该压缩水平。
- **K/V 范数不对称性至关重要** ([#8](https://link.gitcode.com/?target=https%3A%2F%2Fgithub.com%2Ftonbistudio%2Fturboquant-pytorch%2Fissues%2F8&from=https%3A%2F%2Fgitcode.com%2Fgh_mirrors%2Ftu%2Fturboquant-pytorch%3Fsource_module%3Dsearch_result_repo&lang=zh&theme=white))——Qwen 模型的键范数为 172-778，而值范数为 2-4。非对称位分配（为键分配更多比特）必不可少。

## 许可证

MIT
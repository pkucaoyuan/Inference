# LUMINA模型CFG优化配置说明

## 概述

LUMINA-Image-2.0模型在推理过程中启用了两种关键的CFG（Classifier-Free Guidance）优化技术，以提高推理效率并保持生成质量：

1. **CFG-Renormalization (CFG-Renorm)**
2. **CFG-Truncation (CFG-Trunc)**

## 代码实现

### 1. LUMINA模型加载配置

```python
# 文件: inference_benchmark.py (第775-782行)
from diffusers import Lumina2Pipeline

print("正在加载Lumina模型...")
pipe = Lumina2Pipeline.from_pretrained(
    "./Lumina-Image-2.0",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
```

### 2. CFG优化参数配置

#### 实际测量模式中的CFG配置

```python
# 文件: inference_benchmark.py (第360-367行)
kwargs = {
    'prompt': prompt,
    'height': size[0],
    'width': size[1],
    'num_inference_steps': steps,
    'guidance_scale': 4.0,           # CFG引导强度
    'cfg_trunc_ratio': 1.0,          # CFG截断比例 (已启用)
    'cfg_normalization': True,       # CFG重新归一化 (已启用)
    'max_sequence_length': 256       # 最大序列长度
}
```

#### 基准测试模式中的CFG配置

```python
# 文件: inference_benchmark.py (第608-614行)
kwargs = {
    'prompt': prompt,
    'height': size[0],
    'width': size[1],
    'num_inference_steps': steps,
    'guidance_scale': 4.0,           # CFG引导强度
    'cfg_trunc_ratio': 1.0,          # CFG截断比例 (已启用)
    'cfg_normalization': True,       # CFG重新归一化 (已启用)
    'max_sequence_length': 256       # 最大序列长度
}
```

## CFG优化技术详解

### 1. CFG-Renormalization (CFG-Renorm)

**原理**：
- 在CFG引导过程中，大尺度的CFG可能导致某些维度的激活值异常高
- CFG-Renorm通过重新归一化修正后的速度来稳定生成过程
- 使用条件速度的幅值来调整修正后的速度

**数学公式**：
```
vt = vtu + w(vtc - vtu)
其中：
- vt: 修正后的速度
- vtu: 无条件速度
- vtc: 条件速度
- w: CFG尺度
```

**代码实现**：
```python
'cfg_normalization': True  # 启用CFG重新归一化
```

**优势**：
- ✅ 提高CFG引导生成的稳定性
- ✅ 减少视觉伪影
- ✅ 不增加额外计算成本

### 2. CFG-Truncation (CFG-Trunc)

**原理**：
- 研究表明文本信息主要在早期生成阶段被捕获
- 在早期时间步后继续评估条件速度可能是冗余的
- CFG-Trunc在指定时间步后停止条件速度计算

**数学公式**：
```
vt = {
    vtu + w(vtc - vtu)  if t ≥ α
    vtu                 if t < α
}
其中：
- α: 预定义的阈值
- t: 当前时间步
```

**代码实现**：
```python
'cfg_trunc_ratio': 1.0  # CFG截断比例 (1.0表示完全启用)
```

**优势**：
- ✅ 提高推理效率
- ✅ 减少计算开销
- ✅ 保持生成质量

## 参数配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `guidance_scale` | 4.0 | CFG引导强度，平衡生成质量和多样性 |
| `cfg_trunc_ratio` | 1.0 | CFG截断比例，1.0表示完全启用截断 |
| `cfg_normalization` | True | 启用CFG重新归一化 |
| `max_sequence_length` | 256 | 最大文本序列长度 |
| `num_inference_steps` | 30 | 推理步数（LUMINA推荐） |

## 性能影响

### 推理效率提升
- **CFG-Truncation**: 减少后期时间步的条件速度计算
- **CFG-Renormalization**: 提高数值稳定性，减少重试次数

### 质量保证
- **CFG-Renormalization**: 减少视觉伪影，提高生成质量
- **CFG-Truncation**: 在保持质量的前提下提高效率

## 使用建议

### 1. 参数调优
```python
# 高质量模式（较慢）
'guidance_scale': 5.0,
'cfg_trunc_ratio': 0.8,
'num_inference_steps': 50

# 快速模式（较快）
'guidance_scale': 3.5,
'cfg_trunc_ratio': 1.0,
'num_inference_steps': 20

# 平衡模式（当前配置）
'guidance_scale': 4.0,
'cfg_trunc_ratio': 1.0,
'num_inference_steps': 30
```

### 2. 模型兼容性
- 这些CFG优化专门为LUMINA-Image-2.0设计
- 其他模型可能需要不同的参数配置
- 建议根据具体模型调整参数

## 基准测试结果

在当前的基准测试配置下，LUMINA模型能够：
- 在30步内生成高质量图像
- 保持稳定的推理性能
- 有效利用CFG优化技术

## 参考文献

1. LUMINA-Image-2.0论文中的CFG优化技术
2. Classifier-Free Guidance相关研究
3. CFG-Renormalization和CFG-Truncation技术文档

---

**注意**: 当前配置已经过优化，建议在修改参数前先进行小规模测试以验证效果。

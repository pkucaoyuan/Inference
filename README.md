# 图像生成模型推理基准测试项目

## 项目概述
本项目用于比较FLUX、LUMINA和Neta LUMINA三个图像生成模型的推理性能和参数量分析。

## 文件结构

### 核心推理脚本
- **`inference_benchmark.py`** - 主要的推理基准测试脚本，支持FLUX和LUMINA模型
- **`simple_comfyui_test.py`** - Neta LUMINA的ComfyUI测试脚本
- **`run_image_inference.py`** - 运行所有模型推理的主脚本

### 参数量分析工具
- **`model_parameter_analysis.py`** - 完整版参数量分析工具
- **`quick_parameter_analysis.py`** - 快速参数量分析工具
- **`test_parameter_analysis.py`** - 本地模型测试工具
- **`attention_analysis.py`** - 专门分析Attention层参数量的脚本
- **`corrected_parameter_analysis.py`** - 修正版参数量分析（符合官方统计标准）
- **`run_parameter_analysis.py`** - 参数量分析启动脚本

### 模型文件
- **`FLUX.1-dev/`** - FLUX模型文件
- **`Lumina-Image-2.0/`** - LUMINA模型文件
- **`Neta-Lumina/`** - Neta LUMINA模型文件和ComfyUI工作流

### 文档
- **`PARAMETER_ANALYSIS_README.md`** - 参数量分析技术文档
- **`README_PARAMETER_ANALYSIS.md`** - 参数量分析工具包总览
- **`USAGE_GUIDE.md`** - 详细使用说明

## 快速开始

### 1. 运行推理基准测试
```bash
python run_image_inference.py
```

### 2. 运行参数量分析
```bash
# 快速分析
python quick_parameter_analysis.py

# 修正版分析（符合官方标准）
python corrected_parameter_analysis.py

# 专门分析Attention层
python attention_analysis.py
```

### 3. 使用启动脚本
```bash
# 参数量分析启动脚本
python run_parameter_analysis.py
```

## 功能特点

### 推理性能测试
- 实际测量GPU推理时间
- 各层时间分解（Text Encoding、UNet/Transformer、VAE）
- Attention层时间统计
- 内存使用监控
- 图像输出和结果保存

### 参数量分析
- 13种层类型自动分类
- 精确的参数量统计
- 内存占用分析
- FLUX vs LUMINA对比
- Attention层专门分析

### 模型支持
- **FLUX.1-dev**: 12B参数的核心生成器
- **LUMINA-Image-2.0**: 2.6B参数
- **Neta LUMINA**: 通过ComfyUI运行

## 系统要求

### 硬件要求
- GPU: 建议16GB+显存
- 内存: 建议32GB+系统内存
- 存储: 至少50GB可用空间

### 软件要求
- Python: 3.8+
- PyTorch: 2.0+
- CUDA: 11.8+ (如果使用GPU)

### 依赖包
```bash
pip install torch torchvision diffusers transformers accelerate
```

## 输出结果

### 推理性能报告
- 各模型平均推理时间
- 各层时间分解
- GPU内存使用统计
- 生成的图像文件

### 参数量分析报告
- 各层类型参数量统计
- 模型对比分析
- Attention层专门分析
- JSON格式详细数据

## 注意事项

1. **首次运行**: 需要下载模型文件，可能需要较长时间
2. **内存要求**: 建议至少16GB GPU内存
3. **网络要求**: 需要稳定的网络连接下载模型
4. **存储空间**: 模型文件较大，确保有足够存储空间

## 更新日志

- **v1.0**: 初始版本，支持基本推理测试
- **v1.1**: 添加参数量分析功能
- **v1.2**: 改进Attention层识别
- **v1.3**: 添加修正版分析脚本
- **v1.4**: 整理文件结构，删除冗余文件

## 许可证

MIT License

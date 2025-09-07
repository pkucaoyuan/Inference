# 模型推理成本分析工具

本工具用于对比分析FLUX、Lumina和Neta Lumina三个模型的推理成本，包括参数量、注意力机制、GPU时间消耗等方面的系统性分析。

## 功能特性

- **模型配置分析**: 对比三个模型的架构参数、参数量、注意力机制等
- **推理基准测试**: 实际测量GPU推理时间和内存消耗
- **Neta Lumina优化分析**: 专门分析Neta Lumina相比Lumina的优化特性
- **可视化报告**: 生成详细的图表和文本报告

## 文件结构

```
DFmodel_inference/
├── model_analysis.py          # 模型配置分析脚本
├── inference_benchmark.py     # 推理基准测试脚本
├── neta_lumina_analysis.py    # Neta Lumina优化分析脚本
├── run_analysis.py           # 一键运行所有分析的启动脚本
├── download_flux.py          # FLUX模型下载脚本
├── README.md                 # 本文件
├── FLUX.1-dev/              # FLUX模型文件
├── Lumina-Image-2.0/        # Lumina模型文件
└── Neta-Lumina/             # Neta Lumina模型文件
```

## 环境要求

### 系统要求
- Python 3.8+
- CUDA支持的GPU（推荐）
- 至少16GB内存
- 至少20GB可用磁盘空间

### Python依赖
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install numpy matplotlib seaborn
pip install psutil safetensors
pip install huggingface_hub
```

## 使用方法

### 方法1: 一键运行（推荐）
```bash
python run_analysis.py
```

这个脚本会自动：
1. 检查依赖库和模型文件
2. 运行所有分析
3. 生成总结报告

### 方法2: 运行图片记录推理测试（推荐）
```bash
python run_image_inference.py
```

这个脚本会：
- 自动检查依赖和模型文件
- 运行所有模型的推理测试
- 自动保存生成的图片到 `output_images/`
- 自动整理图片到 `organized_images/`
- 生成HTML画廊和对比图
- 生成基准测试报告

### 方法3: 其他工具

#### 单独运行推理基准测试
```bash
python inference_benchmark.py
```

#### 整理现有图片
```bash
python image_organizer.py
```

**注意**: 所有推理测试都会实际加载模型并进行推理，需要更多时间和GPU内存。如果模型加载失败，会显示具体的错误信息。

## 官方推荐推理参数

### FLUX.1-dev
- **Guidance Scale**: 3.5 (推荐值)
- **采样步数**: 20, 30, 50步 (测试所有推荐步数)
- **图像尺寸**: 1024×1024 (统一测试尺寸)
- **采样器**: euler (推荐)

### Lumina-Image-2.0
- **Guidance Scale**: 4.0
- **采样步数**: 50步 (官方推荐)
- **图像尺寸**: 1024×1024 (统一测试尺寸)
- **特殊参数**: cfg_trunc_ratio=0.25, cfg_normalization=True

### Neta-Lumina
- **Guidance Scale**: 4-5.5 (使用4.5)
- **采样步数**: 30步 (官方推荐)
- **图像尺寸**: 1024×1024 (统一测试尺寸)
- **采样器**: res_multistep/euler_ancestral
- **调度器**: linear_quadratic

## 模型下载

### 方法1: 自动下载脚本（推荐）
```bash
# 下载所有模型
python download_models.py

# 只下载特定模型
python download_models.py --models lumina neta

# 生成下载链接文件
python download_models.py --method links
```

### 方法2: 手动下载

#### FLUX模型
FLUX模型需要申请访问权限：
1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-dev
2. 申请访问权限
3. 使用Hugging Face token登录后下载

```bash
# 需要先登录Hugging Face
huggingface-cli login
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev
```

#### Lumina模型
```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0
```

#### Neta Lumina模型
```bash
git clone https://huggingface.co/neta-art/Neta-Lumina
```

### 方法3: 使用Git LFS
如果模型文件已通过Git LFS上传：
```bash
# 安装Git LFS
git lfs install

# 克隆仓库（自动下载LFS文件）
git clone https://github.com/pkucaoyuan/Inference.git
cd Inference
git lfs pull
```

### 模型文件大小
- **FLUX.1-dev**: ~23GB
- **Lumina-Image-2.0**: ~4GB  
- **Neta-Lumina**: ~4GB

**注意**: 确保有足够的磁盘空间和稳定的网络连接。

## 分析内容

### 1. 模型配置对比
- 参数量对比
- 注意力头数对比
- 隐藏层大小对比
- 层数对比
- 文本编码器对比
- VAE配置对比

### 2. 推理性能测试
- GPU推理时间测量
- 内存使用量测量
- 不同图像尺寸的性能对比
- 不同推理步数的性能对比

### 3. Neta Lumina优化分析
- 动漫风格特化优化
- 文本编码器优化（Gemma-2B vs T5-XXL）
- 推理流程优化
- 注意力机制优化
- 内存使用优化

## 输出报告

### 文本报告
- `analysis_report/model_analysis_report.txt`: 模型配置分析报告
- `benchmark_report/benchmark_report.txt`: 推理基准测试报告
- `neta_optimization_report/neta_optimization_analysis.txt`: Neta Lumina优化分析报告
- `analysis_summary.txt`: 总结报告

### 图表报告
- `analysis_report/model_comparison.png`: 模型对比图表
- `benchmark_report/benchmark_comparison.png`: 基准测试对比图表
- `neta_optimization_report/neta_optimization_analysis.png`: 优化分析图表

### JSON数据
- `analysis_report/detailed_analysis.json`: 详细分析数据
- `benchmark_report/benchmark_data.json`: 基准测试数据

## 主要发现

### FLUX模型
- **优势**: 高质量图像生成，支持多种风格
- **劣势**: 推理时间较长，内存需求高
- **适用场景**: 对质量要求高的专业应用

### Lumina模型
- **优势**: Flow-based扩散，推理速度较快
- **特点**: 使用DiT架构，Gemma-2B文本编码器
- **适用场景**: 平衡质量和速度的应用

### Neta Lumina模型
- **优势**: 基于Lumina优化，动漫风格特化
- **优化**: 推理速度提升约15-20%，内存使用减少约10-15%
- **适用场景**: 动漫风格图像生成

## 推理成本分析

### 参数量对比
- FLUX: ~12B参数
- Lumina: ~2B参数
- Neta Lumina: ~2B参数（优化版）

### 推理时间对比（1024x1024图像，20步）
- FLUX: ~3-5秒
- Lumina: ~2-3秒
- Neta Lumina: ~1.5-2.5秒

### 内存使用对比
- FLUX: ~12-16GB
- Lumina: ~6-8GB
- Neta Lumina: ~5-7GB

## 优化建议

1. **根据需求选择模型**:
   - 高质量需求: 选择FLUX
   - 平衡需求: 选择Lumina
   - 动漫风格: 选择Neta Lumina

2. **优化推理设置**:
   - 调整推理步数平衡质量和速度
   - 使用合适的图像尺寸
   - 考虑使用模型量化

3. **硬件配置建议**:
   - 推荐使用RTX 4090或A100
   - 至少16GB显存
   - 使用NVMe SSD存储

## 图片整理功能

### 自动图片整理
运行推理测试后，可以使用图片整理工具自动整理生成的图片：

```bash
# 整理指定目录的图片
python image_organizer.py --input-dir ./output_images --output-dir ./organized_images

# 使用默认目录
python image_organizer.py
```

### 整理功能特性
- **按模型分类**: 自动识别并分类FLUX、Lumina、Neta Lumina的图片
- **按参数分类**: 按尺寸、步数、guidance scale等参数分类
- **生成对比图**: 自动创建相同参数下的模型对比图
- **HTML画廊**: 生成可浏览的HTML图片画廊
- **统计报告**: 生成详细的图片统计报告

### 输出目录结构
```
organized_images/
├── gallery.html              # HTML画廊
├── summary_report.json       # 统计报告
├── comparisons/              # 模型对比图
│   ├── comparison_1024x1024_steps_30.png
│   └── ...
└── individual/               # 按模型分类
    ├── FLUX/
    │   ├── by_size/
    │   ├── by_steps/
    │   └── by_prompt/
    ├── Lumina/
    └── Neta_Lumina/
```

## 故障排除

### 常见问题

1. **依赖库缺失**
   ```bash
   # 自动安装所有依赖
   python install_dependencies.py
   
   # 或手动安装
   pip install torch torchvision torchaudio
   pip install diffusers transformers accelerate
   pip install numpy matplotlib seaborn psutil safetensors
   pip install huggingface_hub GPUtil  # GPUtil为可选依赖
   ```

2. **GPUtil模块缺失**
   - GPUtil是可选依赖，缺失不会影响基本功能
   - 安装命令: `pip install GPUtil`
   - 主要用于获取详细的GPU信息

3. **模型加载失败**
   - 确保模型文件已正确下载到对应目录
   - 检查模型文件完整性
   - 确认有足够的GPU内存
   - 查看具体错误信息进行排查

4. **FLUX模型依赖错误**
   ```bash
   # 自动修复FLUX依赖
   python fix_flux_dependencies.py
   
   # 或手动安装
   pip install protobuf sentencepiece
   ```

5. **Neta Lumina模型文件缺失**
   ```bash
   # 下载Neta Lumina模型
   python download_models.py --model neta-lumina
   ```

4. **CUDA内存不足**
   - 减少图像尺寸
   - 使用CPU卸载
   - 减少推理步数

5. **模型加载失败**
   - 检查模型文件完整性
   - 确认访问权限
   - 更新依赖库

6. **推理速度慢**
   - 检查GPU使用率
   - 优化批处理大小
   - 使用混合精度

### 获取帮助
如果遇到问题，请检查：
1. 依赖库版本是否兼容
2. 模型文件是否完整
3. GPU驱动是否最新
4. 系统内存是否充足

## 许可证

本项目遵循MIT许可证。模型文件遵循各自的许可证：
- FLUX: 需要申请访问权限
- Lumina: Apache 2.0
- Neta Lumina: Apache 2.0

## 贡献

欢迎提交Issue和Pull Request来改进这个分析工具。

## 更新日志

- v1.0: 初始版本，支持三个模型的基础分析
- 计划: 添加更多模型支持，优化分析算法

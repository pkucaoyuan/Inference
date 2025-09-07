# 快速开始指南

## 在其他机器上运行项目

### 1. 克隆项目
```bash
git clone https://github.com/pkucaoyuan/Inference.git
cd Inference
```

### 2. 安装依赖
```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install numpy matplotlib seaborn
pip install psutil safetensors huggingface_hub
```

### 3. 下载模型文件

#### 方法A: 自动下载（推荐）
```bash
python download_models.py
```

#### 方法B: 手动下载
```bash
# Lumina模型（公开）
git clone https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0

# Neta Lumina模型（公开）
git clone https://huggingface.co/neta-art/Neta-Lumina

# FLUX模型（需要权限）
# 1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-dev
# 2. 申请访问权限
# 3. 登录后下载
huggingface-cli login
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev
```

### 4. 运行分析
```bash
# 一键运行所有分析
python run_analysis.py

# 或者分别运行
python model_analysis.py
python inference_benchmark.py
python neta_lumina_analysis.py
```

## 模型文件说明

### 为什么模型文件不能直接上传到GitHub？

1. **文件大小限制**: GitHub单个文件限制100MB，仓库限制1GB
2. **模型文件很大**: 
   - FLUX.1-dev: ~23GB
   - Lumina-Image-2.0: ~4GB
   - Neta-Lumina: ~4GB
3. **带宽成本**: 上传/下载大文件消耗大量带宽

### 解决方案

#### 方案1: Git LFS（推荐）
```bash
# 安装Git LFS
git lfs install

# 配置LFS跟踪大文件
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "*.pt"

# 提交配置
git add .gitattributes
git commit -m "Configure Git LFS"
git push
```

#### 方案2: 自动下载脚本
我们提供了`download_models.py`脚本，可以自动下载所有模型文件。

#### 方案3: 云存储链接
可以将模型文件上传到云存储（如Google Drive、百度网盘），然后在README中提供下载链接。

## 故障排除

### 模型下载失败
1. **网络问题**: 检查网络连接，使用VPN或代理
2. **权限问题**: 确保有Hugging Face访问权限
3. **磁盘空间**: 确保有足够空间（至少30GB）

### 依赖安装失败
1. **Python版本**: 确保使用Python 3.8+
2. **CUDA版本**: 确保CUDA版本与PyTorch兼容
3. **系统依赖**: 安装必要的系统库

### 运行错误
1. **内存不足**: 减少批处理大小或使用CPU
2. **GPU问题**: 检查CUDA安装和GPU驱动
3. **模型路径**: 确保模型文件在正确位置

## 获取帮助

如果遇到问题：
1. 查看README.md中的详细说明
2. 检查错误日志
3. 在GitHub Issues中提问
4. 联系项目维护者

## 贡献

欢迎贡献代码：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

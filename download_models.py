#!/usr/bin/env python3
"""
模型下载脚本
自动下载FLUX、Lumina和Neta Lumina模型文件
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import argparse

def check_git_lfs():
    """检查Git LFS是否安装"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Git LFS已安装")
            return True
        else:
            print("✗ Git LFS未安装")
            return False
    except FileNotFoundError:
        print("✗ Git LFS未安装")
        return False

def install_git_lfs():
    """安装Git LFS"""
    print("正在安装Git LFS...")
    try:
        # Windows
        if os.name == 'nt':
            subprocess.run(['winget', 'install', 'Git.Git-LFS'], check=True)
        # macOS
        elif sys.platform == 'darwin':
            subprocess.run(['brew', 'install', 'git-lfs'], check=True)
        # Linux
        else:
            subprocess.run(['sudo', 'apt-get', 'install', 'git-lfs'], check=True)
        
        print("✓ Git LFS安装成功")
        return True
    except Exception as e:
        print(f"✗ Git LFS安装失败: {e}")
        return False

def download_flux_model():
    """下载FLUX模型"""
    print("\n开始下载FLUX模型...")
    
    try:
        # 尝试下载FLUX.1-dev
        print("尝试下载FLUX.1-dev...")
        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir="./FLUX.1-dev",
            local_dir_use_symlinks=False
        )
        print(f"✓ FLUX.1-dev下载成功: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ FLUX.1-dev下载失败: {e}")
        
        try:
            # 尝试下载FLUX.1-schnell
            print("尝试下载FLUX.1-schnell...")
            model_path = snapshot_download(
                repo_id="black-forest-labs/FLUX.1-schnell",
                local_dir="./FLUX.1-schnell",
                local_dir_use_symlinks=False
            )
            print(f"✓ FLUX.1-schnell下载成功: {model_path}")
            return True
            
        except Exception as e2:
            print(f"✗ FLUX.1-schnell下载失败: {e2}")
            print("\nFLUX模型下载失败的可能原因:")
            print("1. 需要申请访问权限: https://huggingface.co/black-forest-labs/FLUX.1-dev")
            print("2. 需要Hugging Face token: https://huggingface.co/settings/tokens")
            print("3. 网络连接问题")
            return False

def download_lumina_model():
    """下载Lumina模型"""
    print("\n开始下载Lumina模型...")
    
    try:
        model_path = snapshot_download(
            repo_id="Alpha-VLLM/Lumina-Image-2.0",
            local_dir="./Lumina-Image-2.0",
            local_dir_use_symlinks=False
        )
        print(f"✓ Lumina模型下载成功: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Lumina模型下载失败: {e}")
        return False

def download_neta_lumina_model():
    """下载Neta Lumina模型"""
    print("\n开始下载Neta Lumina模型...")
    
    try:
        model_path = snapshot_download(
            repo_id="neta-art/Neta-Lumina",
            local_dir="./Neta-Lumina",
            local_dir_use_symlinks=False
        )
        print(f"✓ Neta Lumina模型下载成功: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Neta Lumina模型下载失败: {e}")
        return False

def download_models_with_git():
    """使用Git下载模型（如果配置了LFS）"""
    print("\n使用Git下载模型...")
    
    models = [
        ("FLUX.1-dev", "https://huggingface.co/black-forest-labs/FLUX.1-dev"),
        ("Lumina-Image-2.0", "https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0"),
        ("Neta-Lumina", "https://huggingface.co/neta-art/Neta-Lumina")
    ]
    
    for name, url in models:
        if os.path.exists(name):
            print(f"✓ {name}已存在，跳过")
            continue
            
        try:
            print(f"下载 {name}...")
            subprocess.run(['git', 'clone', url, name], check=True)
            print(f"✓ {name}下载成功")
        except Exception as e:
            print(f"✗ {name}下载失败: {e}")

def create_model_links():
    """创建模型链接文件"""
    print("\n创建模型链接文件...")
    
    links_content = """# 模型下载链接

## FLUX模型
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **FLUX.1-schnell**: https://huggingface.co/black-forest-labs/FLUX.1-schnell

**注意**: FLUX模型需要申请访问权限

## Lumina模型
- **Lumina-Image-2.0**: https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0

## Neta Lumina模型
- **Neta-Lumina**: https://huggingface.co/neta-art/Neta-Lumina

## 下载方法

### 方法1: 使用Hugging Face Hub
```bash
pip install huggingface_hub
python download_models.py
```

### 方法2: 使用Git
```bash
git clone https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0
git clone https://huggingface.co/neta-art/Neta-Lumina
```

### 方法3: 手动下载
访问上述链接，手动下载模型文件到对应目录。

## 目录结构
```
DFmodel_inference/
├── FLUX.1-dev/          # FLUX模型文件
├── Lumina-Image-2.0/    # Lumina模型文件
└── Neta-Lumina/         # Neta Lumina模型文件
```
"""
    
    with open("MODEL_DOWNLOAD_LINKS.md", "w", encoding="utf-8") as f:
        f.write(links_content)
    
    print("✓ 模型链接文件已创建: MODEL_DOWNLOAD_LINKS.md")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型下载脚本")
    parser.add_argument("--method", choices=["huggingface", "git", "links"], 
                       default="huggingface", help="下载方法")
    parser.add_argument("--models", nargs="+", 
                       choices=["flux", "lumina", "neta"], 
                       default=["flux", "lumina", "neta"],
                       help="要下载的模型")
    
    args = parser.parse_args()
    
    print("模型下载工具")
    print("=" * 50)
    
    if args.method == "links":
        create_model_links()
        return
    
    # 检查Git LFS
    if args.method == "git" and not check_git_lfs():
        if not install_git_lfs():
            print("请手动安装Git LFS后重试")
            return
    
    # 下载模型
    success_count = 0
    
    if "flux" in args.models:
        if download_flux_model():
            success_count += 1
    
    if "lumina" in args.models:
        if download_lumina_model():
            success_count += 1
    
    if "neta" in args.models:
        if download_neta_lumina_model():
            success_count += 1
    
    print(f"\n下载完成: {success_count}/{len(args.models)} 个模型成功")
    
    if success_count < len(args.models):
        print("\n部分模型下载失败，请检查:")
        print("1. 网络连接")
        print("2. Hugging Face访问权限")
        print("3. 磁盘空间")
        print("\n也可以使用以下命令查看下载链接:")
        print("python download_models.py --method links")

if __name__ == "__main__":
    main()

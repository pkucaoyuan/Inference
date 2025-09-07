#!/usr/bin/env python3
"""
依赖安装脚本
自动安装所有必需的依赖库
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package} 安装失败: {e}")
        return False

def main():
    """主函数"""
    print("依赖安装脚本")
    print("=" * 50)
    
    # 必需依赖
    required_packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy",
        "matplotlib",
        "seaborn",
        "psutil",
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "huggingface_hub"
    ]
    
    # 可选依赖
    optional_packages = [
        "GPUtil",  # GPU监控
        "protobuf",  # FLUX模型需要
    ]
    
    print("安装必需依赖...")
    failed_required = []
    
    for package in required_packages:
        if not install_package(package):
            failed_required.append(package)
    
    print("\n安装可选依赖...")
    failed_optional = []
    
    for package in optional_packages:
        if not install_package(package):
            failed_optional.append(package)
    
    # 总结
    print("\n" + "=" * 50)
    print("安装总结:")
    
    if not failed_required:
        print("✓ 所有必需依赖安装成功")
    else:
        print(f"✗ 以下必需依赖安装失败: {', '.join(failed_required)}")
        print("请手动安装这些依赖:")
        for package in failed_required:
            print(f"  pip install {package}")
    
    if not failed_optional:
        print("✓ 所有可选依赖安装成功")
    else:
        print(f"⚠ 以下可选依赖安装失败: {', '.join(failed_optional)}")
        print("可选依赖缺失不会影响基本功能")
    
    if not failed_required:
        print("\n🎉 所有必需依赖安装完成！现在可以运行分析了。")
        print("运行命令: python run_analysis.py")
    else:
        print("\n❌ 部分依赖安装失败，请检查错误信息并手动安装。")

if __name__ == "__main__":
    main()

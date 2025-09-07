#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试依赖库是否正确安装
"""

def test_dependencies():
    """测试所有依赖库"""
    packages = {
        "torch": "torch",
        "numpy": "numpy", 
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "psutil": "psutil",
        "transformers": "transformers",
        "diffusers": "diffusers",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface_hub",
        "sentencepiece": "sentencepiece",
        "protobuf": "google.protobuf"
    }
    
    print("测试依赖库安装状态...")
    print("=" * 40)
    
    all_ok = True
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            all_ok = False
    
    print("=" * 40)
    if all_ok:
        print("✓ 所有依赖库都已正确安装！")
    else:
        print("✗ 部分依赖库缺失，请安装缺失的库")
    
    return all_ok

if __name__ == "__main__":
    test_dependencies()

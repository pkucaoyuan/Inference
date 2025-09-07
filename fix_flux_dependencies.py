#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复FLUX模型依赖问题
"""

import subprocess
import sys

def install_package(package_name):
    """安装Python包"""
    try:
        print(f"正在安装 {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {package_name} 安装成功")
            return True
        else:
            print(f"✗ {package_name} 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {package_name} 安装出错: {e}")
        return False

def main():
    """主函数"""
    print("FLUX模型依赖修复工具")
    print("=" * 40)
    
    # 需要安装的包
    packages = [
        "protobuf",
        "sentencepiece"
    ]
    
    print("检查并安装FLUX模型所需的依赖...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n安装完成: {success_count}/{len(packages)} 个包")
    
    if success_count == len(packages):
        print("\n✓ 所有依赖安装成功！")
        print("现在可以重新运行FLUX模型测试了。")
    else:
        print("\n✗ 部分依赖安装失败，请手动安装：")
        for package in packages:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()

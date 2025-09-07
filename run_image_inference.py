#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行能记录图片的推理测试
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

from inference_benchmark import InferenceBenchmark
from image_organizer import ImageOrganizer

def check_dependencies():
    """检查依赖库"""
    required_packages = [
        "torch", "numpy", "matplotlib", "seaborn", "psutil", 
        "transformers", "diffusers", "safetensors", "huggingface_hub",
        "sentencepiece", "protobuf"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖库:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print("python install_dependencies.py")
        return False
    
    return True

def check_models():
    """检查模型文件"""
    models = {
        "FLUX": Path("./FLUX.1-dev"),
        "Lumina": Path("./Lumina-Image-2.0"),
        "Neta Lumina": Path("./Neta-Lumina")
    }
    
    missing_models = []
    for name, path in models.items():
        if not path.exists():
            missing_models.append(name)
        elif name == "Neta Lumina" and not (path / "model_index.json").exists():
            missing_models.append(f"{name} (文件不完整)")
    
    if missing_models:
        print("缺少以下模型:")
        for model in missing_models:
            print(f"  - {model}")
        print("\n请运行以下命令下载:")
        print("python download_models.py")
        return False
    
    return True

def main():
    """主函数"""
    print("图片记录推理测试工具")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查模型
    if not check_models():
        return
    
    # 设置输出目录
    output_dir = "./output_images"
    
    print(f"\n开始运行推理测试...")
    print(f"图片将保存到: {output_dir}")
    
    # 运行推理测试
    benchmark_runner = InferenceBenchmark(output_dir)
    
    # 测试所有模型
    print("\n测试FLUX模型...")
    flux_results = benchmark_runner.benchmark_flux()
    
    print("\n测试Lumina模型...")
    lumina_results = benchmark_runner.benchmark_lumina()
    
    print("\n测试Neta Lumina模型...")
    neta_results = benchmark_runner.benchmark_neta_lumina()
    
    # 收集结果
    results = []
    if flux_results:
        results.append(flux_results)
    if lumina_results:
        results.append(lumina_results)
    if neta_results:
        results.append(neta_results)
    
    if not results:
        print("错误: 所有模型测试都失败了")
        return
    
    benchmark_runner.results = results
    
    # 生成基准测试报告
    print("\n生成基准测试报告...")
    benchmark_runner.generate_benchmark_report()
    
    # 整理图片
    print("\n整理生成的图片...")
    organizer = ImageOrganizer(output_dir)
    organizer.organize_all()
    
    print("\n" + "=" * 50)
    print("推理测试完成！")
    print("=" * 50)
    print(f"原始图片目录: {output_dir}")
    print(f"整理后目录: {organizer.organized_dir}")
    print(f"HTML画廊: {organizer.organized_dir / 'gallery.html'}")
    print(f"基准测试报告: benchmark_report/")
    print("\n您可以:")
    print("1. 打开HTML画廊查看所有生成的图片")
    print("2. 查看基准测试报告了解性能对比")
    print("3. 在organized目录中按模型、尺寸、步数等分类查看图片")

if __name__ == "__main__":
    main()

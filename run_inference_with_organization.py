#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行推理测试并自动整理图片结果
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

from inference_benchmark import InferenceBenchmark
from image_organizer import ImageOrganizer
from run_analysis import check_dependencies, check_models

def main():
    """主函数"""
    print("模型推理测试与图片整理工具")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，请先安装所有必需依赖。")
        return
    
    # 检查模型文件
    if not check_models():
        print("模型文件检查失败，请确保所有模型已下载。")
        return
    
    # 设置输出目录
    output_dir = "./output_images"
    
    print(f"\n开始运行推理测试...")
    print(f"图片将保存到: {output_dir}")
    
    # 运行推理测试
    benchmark_runner = InferenceBenchmark(output_dir)
    
    # 测试所有模型
    flux_results = benchmark_runner.benchmark_flux()
    lumina_results = benchmark_runner.benchmark_lumina()
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
    print("\n开始整理图片...")
    organizer = ImageOrganizer(output_dir)
    organizer.organize_all()
    
    print("\n" + "=" * 50)
    print("推理测试与图片整理完成！")
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

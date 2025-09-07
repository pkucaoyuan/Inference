#!/usr/bin/env python3
"""
模型分析启动脚本
一键运行所有分析：模型配置分析、推理基准测试、Neta Lumina优化分析
"""

import os
import sys
import time
from pathlib import Path

def check_dependencies():
    """检查依赖库"""
    print("检查依赖库...")
    
    required_packages = [
        'torch',
        'numpy',
        'matplotlib',
        'seaborn',
        'psutil',
        'transformers',
        'diffusers',
        'safetensors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (缺失)")
    
    if missing_packages:
        print(f"\n缺少以下依赖库: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("所有依赖库检查完成！")
    return True

def check_models():
    """检查模型文件"""
    print("\n检查模型文件...")
    
    models = {
        'FLUX': 'FLUX.1-dev',
        'Lumina': 'Lumina-Image-2.0',
        'Neta Lumina': 'Neta-Lumina'
    }
    
    missing_models = []
    
    for name, path in models.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            missing_models.append(name)
            print(f"✗ {name}: {path} (缺失)")
    
    if missing_models:
        print(f"\n缺少以下模型: {', '.join(missing_models)}")
        print("请确保模型已正确下载到对应目录")
        return False
    
    print("所有模型检查完成！")
    return True

def run_analysis():
    """运行分析"""
    print("\n开始运行分析...")
    
    # 1. 运行模型配置分析
    print("\n1. 运行模型配置分析...")
    try:
        from model_analysis import main as model_analysis_main
        model_analysis_main()
        print("✓ 模型配置分析完成")
    except Exception as e:
        print(f"✗ 模型配置分析失败: {e}")
    
    # 2. 运行推理基准测试
    print("\n2. 运行推理基准测试...")
    try:
        from inference_benchmark import main as benchmark_main
        benchmark_main()
        print("✓ 推理基准测试完成")
    except Exception as e:
        print(f"✗ 推理基准测试失败: {e}")
    
    # 3. 运行Neta Lumina优化分析
    print("\n3. 运行Neta Lumina优化分析...")
    try:
        from neta_lumina_analysis import main as neta_analysis_main
        neta_analysis_main()
        print("✓ Neta Lumina优化分析完成")
    except Exception as e:
        print(f"✗ Neta Lumina优化分析失败: {e}")

def generate_summary_report():
    """生成总结报告"""
    print("\n生成总结报告...")
    
    summary_path = Path("analysis_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("模型推理成本分析总结报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("分析内容:\n")
        f.write("1. 模型配置对比分析\n")
        f.write("2. 推理性能基准测试\n")
        f.write("3. Neta Lumina优化特性分析\n\n")
        
        f.write("生成的报告目录:\n")
        f.write("- analysis_report/: 模型配置分析报告\n")
        f.write("- benchmark_report/: 推理基准测试报告\n")
        f.write("- neta_optimization_report/: Neta Lumina优化分析报告\n\n")
        
        f.write("主要发现:\n")
        f.write("- FLUX: 高质量图像生成，推理时间较长\n")
        f.write("- Lumina: Flow-based扩散，推理速度较快\n")
        f.write("- Neta Lumina: 基于Lumina优化，针对动漫风格特化\n\n")
        
        f.write("优化建议:\n")
        f.write("- 根据需求选择合适的模型\n")
        f.write("- 考虑推理时间和质量的平衡\n")
        f.write("- 利用Neta Lumina的动漫特化优势\n")
    
    print(f"总结报告已生成: {summary_path}")

def main():
    """主函数"""
    print("模型推理成本分析启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺失的依赖库，然后重新运行。")
        return
    
    # 检查模型
    if not check_models():
        print("\n请先下载缺失的模型，然后重新运行。")
        return
    
    # 运行分析
    run_analysis()
    
    # 生成总结报告
    generate_summary_report()
    
    print("\n" + "=" * 50)
    print("所有分析完成！")
    print("请查看以下目录中的详细报告:")
    print("- analysis_report/")
    print("- benchmark_report/")
    print("- neta_optimization_report/")
    print("- analysis_summary.txt")

if __name__ == "__main__":
    main()

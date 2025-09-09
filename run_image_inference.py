#!/usr/bin/env python3
"""
运行图像推理测试 - 只测试FLUX和LUMINA
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

from inference_benchmark import InferenceBenchmark

def check_dependencies():
    """检查依赖库"""
    # 定义包名和导入名的映射
    packages = {
        'torch': 'torch',
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'safetensors': 'safetensors',
        'pillow': 'PIL',
        'numpy': 'numpy',
        'psutil': 'psutil',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"缺少以下依赖库: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print("python install_dependencies.py")
        return False
    
    return True

def check_models():
    """检查模型文件"""
    models = {
        "FLUX": Path("./FLUX.1-dev"),
        "Lumina": Path("./Lumina-Image-2.0")
    }
    
    print("检查模型文件完整性...")
    print("=" * 40)
    
    all_ok = True
    for model_name, model_path in models.items():
        if model_path.exists():
            # 检查关键文件
            key_files = ["model_index.json"]
            missing_files = []
            
            for file_name in key_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"❌ {model_name}: 缺少文件 {missing_files}")
                all_ok = False
            else:
                print(f"✅ {model_name}: 文件完整")
        else:
            print(f"❌ {model_name}: 目录不存在")
            all_ok = False
    
    if not all_ok:
        print("\n请运行以下命令下载模型:")
        print("python download_models.py")
        return False
    
    return True

def main():
    """主函数"""
    print("FLUX和LUMINA图像推理测试")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查模型
    if not check_models():
        return
    
    print("\n开始推理测试...")
    
    # 创建推理基准测试器
    benchmark = InferenceBenchmark()
    
    # 运行测试
    results = benchmark.run_all_benchmarks()
    
    if results:
        print(f"\n✅ 测试完成，共生成 {len(results)} 个结果")
        
        print("\n🎉 所有测试完成！")
        print("📁 查看生成的图片和报告:")
        print("   - 图片目录: unified_output_*/")
        print("   - 报告文件: benchmark_report_*/")
    else:
        print("❌ 测试失败")

if __name__ == "__main__":
    main()

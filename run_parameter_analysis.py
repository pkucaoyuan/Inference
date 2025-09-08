#!/usr/bin/env python3
"""
参数量分析工具启动脚本
提供简单的菜单选择界面
"""

import os
import sys
import subprocess

def print_menu():
    """打印菜单"""
    print("\n" + "="*50)
    print("模型参数量分析工具")
    print("="*50)
    print("1. 快速分析 (推荐)")
    print("2. 完整分析")
    print("3. 本地模型测试")
    print("4. 仅分析FLUX")
    print("5. 仅分析LUMINA")
    print("6. 查看使用说明")
    print("0. 退出")
    print("="*50)

def run_quick_analysis():
    """运行快速分析"""
    print("\n🚀 启动快速分析...")
    try:
        subprocess.run([sys.executable, "quick_parameter_analysis.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 快速分析失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到 quick_parameter_analysis.py 文件")

def run_full_analysis():
    """运行完整分析"""
    print("\n🔍 启动完整分析...")
    try:
        subprocess.run([sys.executable, "model_parameter_analysis.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 完整分析失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到 model_parameter_analysis.py 文件")

def run_local_test():
    """运行本地模型测试"""
    print("\n🧪 启动本地模型测试...")
    try:
        subprocess.run([sys.executable, "test_parameter_analysis.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 本地测试失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到 test_parameter_analysis.py 文件")

def run_flux_only():
    """仅分析FLUX"""
    print("\n🎨 启动FLUX分析...")
    try:
        subprocess.run([sys.executable, "model_parameter_analysis.py", "--flux-only"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FLUX分析失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到 model_parameter_analysis.py 文件")

def run_lumina_only():
    """仅分析LUMINA"""
    print("\n🌟 启动LUMINA分析...")
    try:
        subprocess.run([sys.executable, "model_parameter_analysis.py", "--lumina-only"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ LUMINA分析失败: {e}")
    except FileNotFoundError:
        print("❌ 找不到 model_parameter_analysis.py 文件")

def show_usage_guide():
    """显示使用说明"""
    print("\n📖 使用说明:")
    print("-" * 30)
    print("1. 快速分析: 自动分析FLUX和LUMINA，生成对比报告")
    print("2. 完整分析: 详细分析所有模型，支持命令行参数")
    print("3. 本地测试: 使用本地已下载的模型进行分析")
    print("4. 仅分析FLUX: 只分析FLUX模型")
    print("5. 仅分析LUMINA: 只分析LUMINA模型")
    print("\n💡 提示:")
    print("- 首次运行需要下载模型，可能需要较长时间")
    print("- 建议使用快速分析开始")
    print("- 如果遇到内存问题，可以尝试本地测试")
    print("- 详细说明请查看 USAGE_GUIDE.md")

def check_dependencies():
    """检查依赖"""
    required_packages = ['torch', 'diffusers', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install torch diffusers transformers accelerate")
        return False
    
    return True

def main():
    """主函数"""
    print("🔧 检查依赖...")
    if not check_dependencies():
        return
    
    print("✅ 依赖检查通过")
    
    while True:
        print_menu()
        
        try:
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 再见！")
                break
            elif choice == "1":
                run_quick_analysis()
            elif choice == "2":
                run_full_analysis()
            elif choice == "3":
                run_local_test()
            elif choice == "4":
                run_flux_only()
            elif choice == "5":
                run_lumina_only()
            elif choice == "6":
                show_usage_guide()
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()

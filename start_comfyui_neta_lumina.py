#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动ComfyUI进行Neta Lumina测试
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def find_comfyui():
    """查找ComfyUI安装路径"""
    possible_paths = [
        Path("./ComfyUI"),
        Path("../ComfyUI"),
        Path("~/ComfyUI").expanduser(),
        Path("~/comfyui").expanduser(),
        Path("/opt/ComfyUI"),
        Path("/usr/local/ComfyUI")
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "main.py").exists():
            return path
    
    return None

def check_neta_lumina_models(comfyui_path):
    """检查Neta Lumina模型文件"""
    models_dir = comfyui_path / "models"
    
    required_files = {
        "UNet": models_dir / "unet" / "neta-lumina-v1.0.safetensors",
        "Text Encoder": models_dir / "text_encoders" / "gemma_2_2b_fp16.safetensors",
        "VAE": models_dir / "vae" / "ae.safetensors"
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    return missing_files

def main():
    """主函数"""
    print("ComfyUI Neta Lumina快速启动器")
    print("=" * 40)
    
    # 查找ComfyUI
    comfyui_path = find_comfyui()
    if not comfyui_path:
        print("❌ 未找到ComfyUI安装路径")
        print("请确保ComfyUI已安装，或手动指定路径")
        return
    
    print(f"✅ 找到ComfyUI: {comfyui_path}")
    
    # 检查模型文件
    missing_files = check_neta_lumina_models(comfyui_path)
    if missing_files:
        print("❌ 缺少Neta Lumina模型文件:")
        for file in missing_files:
            print(f"   {file}")
        print("\n请下载模型文件到ComfyUI对应目录")
        return
    
    print("✅ Neta Lumina模型文件完整")
    
    # 检查工作流文件
    workflow_file = Path("./Neta-Lumina/lumina_workflow.json")
    if not workflow_file.exists():
        print("❌ 工作流文件不存在: lumina_workflow.json")
        print("请确保已下载Neta-Lumina模型")
        return
    
    print("✅ 工作流文件存在")
    
    # 启动ComfyUI
    print("\n启动ComfyUI...")
    print("请在浏览器中访问: http://localhost:8188")
    print("然后加载工作流文件: ./Neta-Lumina/lumina_workflow.json")
    print("按Ctrl+C停止服务")
    
    try:
        # 切换到ComfyUI目录
        os.chdir(comfyui_path)
        
        # 启动ComfyUI
        subprocess.run([
            sys.executable, "main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188"
        ])
    except KeyboardInterrupt:
        print("\nComfyUI已停止")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()

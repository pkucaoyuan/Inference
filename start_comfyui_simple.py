#!/usr/bin/env python3
"""
简单的ComfyUI启动脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """启动ComfyUI"""
    # ComfyUI路径
    comfyui_path = Path("../ComfyUI")
    
    if not comfyui_path.exists():
        print(f"❌ ComfyUI路径不存在: {comfyui_path}")
        print("请确保ComfyUI已正确安装")
        return
    
    main_py = comfyui_path / "main.py"
    if not main_py.exists():
        print(f"❌ ComfyUI主文件不存在: {main_py}")
        return
    
    print(f"✅ 找到ComfyUI: {comfyui_path}")
    print("启动ComfyUI...")
    
    try:
        # 启动ComfyUI
        process = subprocess.Popen([
            sys.executable, "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188"
        ], cwd=comfyui_path)
        
        print(f"✅ ComfyUI已启动 (PID: {process.pid})")
        print("访问地址: http://localhost:8188")
        print("按 Ctrl+C 停止ComfyUI")
        
        # 保持运行
        process.wait()
        
    except KeyboardInterrupt:
        print("\n停止ComfyUI...")
        process.terminate()
        process.wait()
        print("✅ ComfyUI已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()

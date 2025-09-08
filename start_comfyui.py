#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动ComfyUI的脚本
用于运行Neta Lumina模型测试
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

class ComfyUIStarter:
    """ComfyUI启动器"""
    
    def __init__(self, comfyui_path="../ComfyUI", port=8188, gpu_id=0):
        self.comfyui_path = Path(comfyui_path)
        self.port = port
        self.gpu_id = gpu_id
        self.process = None
        
    def check_comfyui_path(self):
        """检查ComfyUI路径是否存在"""
        if not self.comfyui_path.exists():
            print(f"❌ ComfyUI路径不存在: {self.comfyui_path}")
            print("请确保ComfyUI已正确安装")
            return False
            
        main_py = self.comfyui_path / "main.py"
        if not main_py.exists():
            print(f"❌ ComfyUI主文件不存在: {main_py}")
            return False
            
        print(f"✅ 找到ComfyUI: {self.comfyui_path}")
        return True
    
    def check_models(self):
        """检查Neta Lumina模型文件"""
        print("检查Neta Lumina模型文件...")
        
        # 检查UNet模型
        unet_path = self.comfyui_path / "models" / "unet" / "neta-lumina-v1.0.safetensors"
        if unet_path.exists():
            size_gb = unet_path.stat().st_size / (1024**3)
            print(f"✅ UNet: {unet_path} ({size_gb:.2f} GB)")
        else:
            print(f"❌ UNet: {unet_path} 不存在")
            return False
            
        # 检查Text Encoder模型
        text_encoder_path = self.comfyui_path / "models" / "text_encoders" / "gemma_2_2b_fp16.safetensors"
        if text_encoder_path.exists():
            size_gb = text_encoder_path.stat().st_size / (1024**3)
            print(f"✅ Text Encoder: {text_encoder_path} ({size_gb:.2f} GB)")
        else:
            print(f"❌ Text Encoder: {text_encoder_path} 不存在")
            return False
            
        # 检查VAE模型
        vae_path = self.comfyui_path / "models" / "vae" / "ae.safetensors"
        if vae_path.exists():
            size_gb = vae_path.stat().st_size / (1024**3)
            print(f"✅ VAE: {vae_path} ({size_gb:.2f} GB)")
        else:
            print(f"❌ VAE: {vae_path} 不存在")
            return False
            
        return True
    
    def start_comfyui(self):
        """启动ComfyUI"""
        if not self.check_comfyui_path():
            return False
            
        if not self.check_models():
            print("请先下载Neta Lumina模型文件")
            return False
        
        print(f"启动ComfyUI (端口: {self.port}, GPU: {self.gpu_id})...")
        
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # 启动ComfyUI
        try:
            self.process = subprocess.Popen([
                sys.executable, "main.py",
                "--listen", "127.0.0.1",
                "--port", str(self.port),
                "--gpu-memory-utilization", "0.8"
            ], cwd=self.comfyui_path, env=env)
            
            print(f"✅ ComfyUI已启动 (PID: {self.process.pid})")
            print(f"访问地址: http://localhost:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ 启动ComfyUI失败: {e}")
            return False
    
    def wait_for_startup(self, timeout=60):
        """等待ComfyUI启动完成"""
        print("等待ComfyUI启动...")
        
        import requests
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/", timeout=5)
                if response.status_code == 200:
                    print("✅ ComfyUI启动完成")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("❌ ComfyUI启动超时")
        return False
    
    def stop_comfyui(self):
        """停止ComfyUI"""
        if self.process:
            print("停止ComfyUI...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("✅ ComfyUI已停止")
            except subprocess.TimeoutExpired:
                print("强制停止ComfyUI...")
                self.process.kill()
                self.process.wait()
                print("✅ ComfyUI已强制停止")
    
    def run(self):
        """运行ComfyUI"""
        try:
            if not self.start_comfyui():
                return False
                
            if not self.wait_for_startup():
                return False
                
            print("\n" + "="*50)
            print("ComfyUI已启动，可以开始测试")
            print("="*50)
            print("按 Ctrl+C 停止ComfyUI")
            
            # 保持运行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n收到停止信号...")
            self.stop_comfyui()
        except Exception as e:
            print(f"❌ 运行错误: {e}")
            self.stop_comfyui()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="启动ComfyUI")
    parser.add_argument("--comfyui-path", default="../ComfyUI", help="ComfyUI路径")
    parser.add_argument("--port", type=int, default=8188, help="端口号")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    starter = ComfyUIStarter(
        comfyui_path=args.comfyui_path,
        port=args.port,
        gpu_id=args.gpu_id
    )
    
    starter.run()

if __name__ == "__main__":
    main()

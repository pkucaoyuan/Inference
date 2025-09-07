#!/usr/bin/env python3
"""
FLUX模型下载脚本
支持多种下载方式
"""

import os
import requests
from huggingface_hub import snapshot_download, hf_hub_download

def download_flux_models():
    """下载FLUX模型的不同组件"""
    
    print("开始下载FLUX模型...")
    
    try:
        # 方法1：使用snapshot_download下载整个仓库
        print("尝试使用snapshot_download下载...")
        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir="./FLUX.1-dev",
            local_dir_use_symlinks=False
        )
        print(f"模型下载成功到: {model_path}")
        return True
        
    except Exception as e:
        print(f"snapshot_download失败: {e}")
        
        try:
            # 方法2：尝试下载FLUX.1-schnell版本
            print("尝试下载FLUX.1-schnell版本...")
            model_path = snapshot_download(
                repo_id="black-forest-labs/FLUX.1-schnell",
                local_dir="./FLUX.1-schnell",
                local_dir_use_symlinks=False
            )
            print(f"FLUX.1-schnell下载成功到: {model_path}")
            return True
            
        except Exception as e2:
            print(f"FLUX.1-schnell下载失败: {e2}")
            
            try:
                # 方法3：手动下载关键文件
                print("尝试手动下载关键文件...")
                files_to_download = [
                    "flux1-dev.safetensors",
                    "ae.safetensors", 
                    "clip_l.safetensors",
                    "t5xxl_fp16.safetensors"
                ]
                
                os.makedirs("./FLUX.1-dev", exist_ok=True)
                
                for filename in files_to_download:
                    try:
                        print(f"下载 {filename}...")
                        hf_hub_download(
                            repo_id="black-forest-labs/FLUX.1-dev",
                            filename=filename,
                            local_dir="./FLUX.1-dev"
                        )
                        print(f"{filename} 下载成功")
                    except Exception as file_e:
                        print(f"{filename} 下载失败: {file_e}")
                
                return True
                
            except Exception as e3:
                print(f"手动下载也失败: {e3}")
                return False

def download_flux_schnell():
    """下载FLUX.1-schnell版本（可能不需要权限）"""
    try:
        print("尝试下载FLUX.1-schnell...")
        model_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            local_dir="./FLUX.1-schnell",
            local_dir_use_symlinks=False
        )
        print(f"FLUX.1-schnell下载成功到: {model_path}")
        return True
    except Exception as e:
        print(f"FLUX.1-schnell下载失败: {e}")
        return False

if __name__ == "__main__":
    print("FLUX模型下载工具")
    print("=" * 50)
    
    # 首先尝试下载FLUX.1-dev
    if download_flux_models():
        print("FLUX.1-dev下载成功！")
    else:
        print("FLUX.1-dev下载失败，尝试FLUX.1-schnell...")
        if download_flux_schnell():
            print("FLUX.1-schnell下载成功！")
        else:
            print("所有下载方法都失败了。")
            print("\n请尝试以下解决方案：")
            print("1. 访问 https://huggingface.co/black-forest-labs/FLUX.1-dev 申请访问权限")
            print("2. 获取Hugging Face token并设置环境变量：")
            print("   export HUGGINGFACE_HUB_TOKEN=your_token_here")
            print("3. 手动从网页下载模型文件")

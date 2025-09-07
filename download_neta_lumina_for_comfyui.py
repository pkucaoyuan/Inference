#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为ComfyUI下载Neta Lumina模型文件
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_neta_lumina_models(comfyui_path: str = None):
    """下载Neta Lumina模型文件到ComfyUI目录"""
    
    if comfyui_path:
        comfyui_path = Path(comfyui_path)
    else:
        # 自动查找ComfyUI路径
        possible_paths = [
            Path("./ComfyUI"),
            Path("../ComfyUI"),
            Path("~/ComfyUI").expanduser(),
            Path("~/comfyui").expanduser(),
            Path("/opt/ComfyUI"),
            Path("/usr/local/ComfyUI")
        ]
        
        comfyui_path = None
        for path in possible_paths:
            if path.exists() and (path / "main.py").exists():
                comfyui_path = path
                break
    
    if not comfyui_path:
        print("❌ 未找到ComfyUI安装路径")
        print("请手动指定ComfyUI路径:")
        print("python download_neta_lumina_for_comfyui.py --comfyui-path /path/to/ComfyUI")
        return False
    
    print(f"✅ ComfyUI路径: {comfyui_path}")
    
    # 创建模型目录
    models_dir = comfyui_path / "models"
    unet_dir = models_dir / "unet"
    text_encoder_dir = models_dir / "text_encoders"
    vae_dir = models_dir / "vae"
    
    for dir_path in [unet_dir, text_encoder_dir, vae_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 下载模型文件
    models_to_download = [
        {
            "name": "UNet",
            "repo_id": "neta-art/Neta-Lumina",
            "filename": "Unet/neta-lumina-v1.0.safetensors",
            "local_path": unet_dir / "neta-lumina-v1.0.safetensors"
        },
        {
            "name": "Text Encoder",
            "repo_id": "neta-art/Neta-Lumina", 
            "filename": "Text Encoder/gemma_2_2b_fp16.safetensors",
            "local_path": text_encoder_dir / "gemma_2_2b_fp16.safetensors"
        },
        {
            "name": "VAE",
            "repo_id": "neta-art/Neta-Lumina",
            "filename": "VAE/ae.safetensors", 
            "local_path": vae_dir / "ae.safetensors"
        }
    ]
    
    print("开始下载Neta Lumina模型文件...")
    
    for model in models_to_download:
        print(f"\n下载 {model['name']}...")
        try:
            # 检查文件是否已存在
            if model['local_path'].exists():
                size_gb = model['local_path'].stat().st_size / (1024**3)
                print(f"✅ {model['name']} 已存在 ({size_gb:.2f} GB)")
                continue
            
            # 下载文件
            downloaded_path = hf_hub_download(
                repo_id=model['repo_id'],
                filename=model['filename'],
                local_dir=model['local_path'].parent,
                local_dir_use_symlinks=False
            )
            
            # 重命名文件
            if downloaded_path != str(model['local_path']):
                Path(downloaded_path).rename(model['local_path'])
            
            size_gb = model['local_path'].stat().st_size / (1024**3)
            print(f"✅ {model['name']} 下载完成 ({size_gb:.2f} GB)")
            
        except Exception as e:
            print(f"❌ {model['name']} 下载失败: {e}")
            return False
    
    print("\n✅ 所有模型文件下载完成！")
    print(f"模型文件位置:")
    print(f"  UNet: {unet_dir / 'neta-lumina-v1.0.safetensors'}")
    print(f"  Text Encoder: {text_encoder_dir / 'gemma_2_2b_fp16.safetensors'}")
    print(f"  VAE: {vae_dir / 'ae.safetensors'}")
    
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="为ComfyUI下载Neta Lumina模型文件")
    parser.add_argument("--comfyui-path", help="ComfyUI安装路径")
    
    args = parser.parse_args()
    
    success = download_neta_lumina_models(args.comfyui_path)
    
    if success:
        print("\n🎉 下载完成！现在可以启动ComfyUI进行测试:")
        print("python start_comfyui_neta_lumina.py")
    else:
        print("\n❌ 下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main()

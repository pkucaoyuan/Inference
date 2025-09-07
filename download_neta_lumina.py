#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neta Lumina模型下载脚本
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url: str, filepath: Path, description: str = "Downloading"):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ 下载完成: {filepath}")
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

def main():
    """主函数"""
    print("Neta Lumina模型下载工具")
    print("=" * 40)
    
    # 创建目录
    neta_dir = Path("./Neta-Lumina")
    neta_dir.mkdir(exist_ok=True)
    
    # 下载文件列表
    files_to_download = [
        {
            "url": "https://huggingface.co/neta-art/Neta-Lumina/resolve/main/neta-lumina-v1.0-all-in-one.safetensors",
            "path": neta_dir / "neta-lumina-v1.0-all-in-one.safetensors",
            "description": "Neta Lumina All-in-One模型"
        }
    ]
    
    print("开始下载Neta Lumina模型文件...")
    print("注意: 这是一个大文件（约5GB），请确保网络连接稳定")
    
    success_count = 0
    for file_info in files_to_download:
        if file_info["path"].exists():
            print(f"文件已存在，跳过: {file_info['path']}")
            success_count += 1
            continue
        
        print(f"\n下载: {file_info['description']}")
        if download_file(file_info["url"], file_info["path"], file_info["description"]):
            success_count += 1
    
    print(f"\n下载完成: {success_count}/{len(files_to_download)} 个文件")
    
    if success_count == len(files_to_download):
        print("\n✓ 所有文件下载完成！")
        print("现在可以运行推理测试了。")
    else:
        print("\n✗ 部分文件下载失败，请检查网络连接后重试。")

if __name__ == "__main__":
    main()

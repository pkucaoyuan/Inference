#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型文件完整性
"""

from pathlib import Path

def check_models():
    """检查模型文件"""
    models = {
        "FLUX": Path("./FLUX.1-dev"),
        "Lumina": Path("./Lumina-Image-2.0"),
        "Neta Lumina": Path("./Neta-Lumina")
    }
    
    print("检查模型文件完整性...")
    print("=" * 40)
    
    all_ok = True
    for name, path in models.items():
        print(f"\n检查 {name}:")
        if not path.exists():
            print(f"  ✗ 目录不存在: {path}")
            all_ok = False
            continue
        
        print(f"  ✓ 目录存在: {path}")
        
        if name == "Neta Lumina":
            # Neta Lumina使用ComfyUI格式
            required_files = ["lumina_workflow.json", "README.md"]
            for file in required_files:
                file_path = path / file
                if file_path.exists():
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ 缺少: {file}")
                    all_ok = False
        else:
            # FLUX和Lumina使用标准diffusers格式
            required_files = ["model_index.json"]
            for file in required_files:
                file_path = path / file
                if file_path.exists():
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ 缺少: {file}")
                    all_ok = False
        
        # 列出目录中的所有文件
        print(f"  目录内容:")
        try:
            files = list(path.iterdir())
            for file in files[:10]:  # 只显示前10个文件
                print(f"    - {file.name}")
            if len(files) > 10:
                print(f"    ... 还有 {len(files) - 10} 个文件")
        except Exception as e:
            print(f"    ✗ 无法读取目录: {e}")
    
    print("\n" + "=" * 40)
    if all_ok:
        print("✓ 所有模型文件检查通过！")
    else:
        print("✗ 部分模型文件缺失或不完整")
    
    return all_ok

if __name__ == "__main__":
    check_models()

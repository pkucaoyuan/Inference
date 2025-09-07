#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Neta Lumina模型（支持all-in-one格式）
"""

import os
import sys
import torch
from pathlib import Path
import time

def test_neta_lumina_all_in_one():
    """测试Neta Lumina all-in-one模型"""
    print("测试Neta Lumina all-in-one模型")
    print("=" * 40)
    
    # 检查模型文件
    neta_dir = Path("./Neta-Lumina")
    all_in_one_file = neta_dir / "neta-lumina-v1.0-all-in-one.safetensors"
    
    if not all_in_one_file.exists():
        print("未找到 neta-lumina-v1.0-all-in-one.safetensors")
        print("请下载该文件到 Neta-Lumina 目录")
        return False
    
    print(f"找到模型文件: {all_in_one_file}")
    print(f"文件大小: {all_in_one_file.stat().st_size / (1024**3):.2f} GB")
    
    # 尝试使用diffusers加载
    try:
        from diffusers import Lumina2Pipeline
        
        print("尝试使用Lumina2Pipeline加载...")
        pipe = Lumina2Pipeline.from_pretrained(
            "./Neta-Lumina",
            torch_dtype=torch.bfloat16
        )
        
        # 不使用CPU卸载，保持GPU内存统计准确
        # pipe.enable_model_cpu_offload()
        
        print("模型加载成功！")
        
        # 测试推理
        prompt = "A beautiful anime character in a magical garden, detailed, high quality"
        
        print(f"开始推理测试...")
        print(f"提示词: {prompt}")
        
        start_time = time.time()
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=30,
            guidance_scale=4.0,
            cfg_trunc_ratio=1.0,
            cfg_normalization=True,
            max_sequence_length=256
        ).images[0]
        end_time = time.time()
        
        # 保存图片
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"neta_lumina_test_{timestamp}.png"
        image.save(output_path)
        
        print(f"推理完成！")
        print(f"推理时间: {end_time - start_time:.2f}秒")
        print(f"图片已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"加载失败: {e}")
        print("可能的原因:")
        print("1. 模型格式不兼容")
        print("2. 缺少必要的依赖")
        print("3. 需要使用ComfyUI环境")
        return False

def test_neta_lumina_components():
    """测试Neta Lumina分离组件"""
    print("\n测试Neta Lumina分离组件")
    print("=" * 40)
    
    neta_dir = Path("./Neta-Lumina")
    
    # 检查组件文件
    components = {
        "UNet": neta_dir / "Unet" / "neta-lumina-v1.0.safetensors",
        "Text Encoder": neta_dir / "Text Encoder" / "gemma_2_2b_fp16.safetensors", 
        "VAE": neta_dir / "VAE" / "ae.safetensors"
    }
    
    missing_components = []
    for name, path in components.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: 缺失")
            missing_components.append(name)
    
    if missing_components:
        print(f"\n缺少组件: {', '.join(missing_components)}")
        print("请下载完整的组件文件")
        return False
    
    print("\n所有组件文件完整")
    return True

def main():
    """主函数"""
    print("Neta Lumina模型测试工具")
    print("=" * 50)
    
    # 测试all-in-one格式
    success1 = test_neta_lumina_all_in_one()
    
    # 测试分离组件
    success2 = test_neta_lumina_components()
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"All-in-one格式: {'成功' if success1 else '失败'}")
    print(f"分离组件格式: {'完整' if success2 else '不完整'}")
    
    if not success1 and not success2:
        print("\n建议:")
        print("1. 使用ComfyUI环境运行")
        print("2. 检查模型文件完整性")
        print("3. 更新diffusers库到最新版本")

if __name__ == "__main__":
    main()

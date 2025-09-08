#!/usr/bin/env python3
"""
测试参数量分析功能
使用本地已下载的模型进行测试
"""

import torch
import torch.nn as nn
from collections import defaultdict
import json
from datetime import datetime
import os

def classify_layer_type(module, full_name):
    """根据模块类型和名称分类层"""
    module_type = type(module).__name__
    
    # Attention相关
    if any(keyword in module_type.lower() for keyword in ['attention', 'attn', 'self_attn', 'cross_attn']):
        return "Attention层"
    
    # 线性层
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "线性/卷积层"
    
    # 归一化层
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        return "归一化层"
    
    # 激活函数
    if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
        return "激活函数"
    
    # 检查Swish激活函数（可能不存在于某些PyTorch版本）
    try:
        if isinstance(module, nn.Swish):
            return "激活函数"
    except AttributeError:
        pass
    
    # 嵌入层
    if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
        return "嵌入层"
    
    # 位置编码
    if 'position' in full_name.lower() or 'pos_embed' in full_name.lower():
        return "位置编码"
    
    # 时间步编码
    if 'time' in full_name.lower() or 'timestep' in full_name.lower():
        return "时间步编码"
    
    # 前馈网络
    if any(keyword in full_name.lower() for keyword in ['feed_forward', 'mlp', 'ffn']):
        return "前馈网络"
    
    # Transformer块
    if any(keyword in module_type.lower() for keyword in ['transformer', 'block']):
        return "Transformer块"
    
    # 残差连接
    if 'residual' in full_name.lower() or 'resnet' in full_name.lower():
        return "残差连接"
    
    # VAE相关
    if any(keyword in full_name.lower() for keyword in ['encoder', 'decoder', 'vae']):
        return "VAE层"
    
    # 文本编码器
    if any(keyword in full_name.lower() for keyword in ['text_encoder', 'text_model']):
        return "文本编码器"
    
    # 其他
    return "其他层"

def analyze_model_parameters(model, model_name="Model"):
    """分析模型各层参数量"""
    layer_stats = {}
    
    total_params = 0
    total_size_mb = 0.0
    
    def analyze_module(module, prefix=""):
        nonlocal total_params, total_size_mb
        
        # 使用list()创建子模块的副本，避免在迭代时修改字典
        children = list(module.named_children())
        
        for name, child in children:
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 计算当前模块参数
            module_params = sum(p.numel() for p in child.parameters())
            module_size_mb = sum(p.numel() * p.element_size() for p in child.parameters()) / (1024 * 1024)
            
            if module_params > 0:
                # 根据模块类型分类
                layer_type = classify_layer_type(child, full_name)
                
                # 确保layer_type键存在
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {
                        'parameters': 0,
                        'layers': 0,
                        'size_mb': 0.0
                    }
                
                layer_stats[layer_type]['parameters'] += module_params
                layer_stats[layer_type]['layers'] += 1
                layer_stats[layer_type]['size_mb'] += module_size_mb
                
                total_params += module_params
                total_size_mb += module_size_mb
                
                print(f"  {full_name}: {module_params:,} 参数 ({module_size_mb:.2f}MB) - {layer_type}")
            
            # 递归分析子模块
            analyze_module(child, full_name)
    
    print(f"\n=== {model_name} 参数量分析 ===")
    analyze_module(model)
    
    # 添加总计
    layer_stats['总计'] = {
        'parameters': total_params,
        'layers': sum(stats['layers'] for stats in layer_stats.values() if stats != layer_stats['总计']),
        'size_mb': total_size_mb
    }
    
    return dict(layer_stats), total_params, total_size_mb

def test_with_local_models():
    """使用本地模型进行测试"""
    print("开始测试参数量分析功能...")
    
    # 检查本地模型目录
    flux_path = "FLUX.1-dev"
    lumina_path = "Lumina-Image-2.0"
    
    results = {}
    
    # 测试FLUX模型
    if os.path.exists(flux_path):
        print(f"\n找到FLUX模型目录: {flux_path}")
        try:
            from diffusers import FluxPipeline
            pipe = FluxPipeline.from_pretrained(
                flux_path,
                torch_dtype=torch.float16,
                device_map="balanced"
            )
            
            flux_results = {}
            components = {
                'Text Encoder': pipe.text_encoder,
                'Text Encoder 2': pipe.text_encoder_2,
                'Transformer': pipe.transformer,
                'VAE': pipe.vae
            }
            
            for comp_name, comp_model in components.items():
                if comp_model is not None:
                    print(f"\n--- 分析 {comp_name} ---")
                    layer_stats, total_params, total_size = analyze_model_parameters(comp_model, comp_name)
                    flux_results[comp_name] = {
                        'layer_stats': layer_stats,
                        'total_parameters': total_params,
                        'total_size_mb': total_size,
                        'total_size_gb': total_size / 1024
                    }
            
            results['FLUX'] = flux_results
            print(f"\n✅ FLUX模型分析完成")
            
        except Exception as e:
            print(f"❌ FLUX模型分析失败: {e}")
            results['FLUX'] = {'error': str(e)}
    else:
        print(f"❌ 未找到FLUX模型目录: {flux_path}")
        results['FLUX'] = {'error': 'Model not found'}
    
    # 测试LUMINA模型
    if os.path.exists(lumina_path):
        print(f"\n找到LUMINA模型目录: {lumina_path}")
        try:
            from diffusers import Lumina2Pipeline
            pipe = Lumina2Pipeline.from_pretrained(
                lumina_path,
                torch_dtype=torch.float16,
                device_map="balanced"
            )
            
            lumina_results = {}
            components = {
                'Text Encoder': pipe.text_encoder,
                'Transformer': pipe.transformer,
                'VAE': pipe.vae
            }
            
            for comp_name, comp_model in components.items():
                if comp_model is not None:
                    print(f"\n--- 分析 {comp_name} ---")
                    layer_stats, total_params, total_size = analyze_model_parameters(comp_model, comp_name)
                    lumina_results[comp_name] = {
                        'layer_stats': layer_stats,
                        'total_parameters': total_params,
                        'total_size_mb': total_size,
                        'total_size_gb': total_size / 1024
                    }
            
            results['LUMINA'] = lumina_results
            print(f"\n✅ LUMINA模型分析完成")
            
        except Exception as e:
            print(f"❌ LUMINA模型分析失败: {e}")
            results['LUMINA'] = {'error': str(e)}
    else:
        print(f"❌ 未找到LUMINA模型目录: {lumina_path}")
        results['LUMINA'] = {'error': 'Model not found'}
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_parameter_analysis_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 测试完成！结果已保存到: {report_file}")
    
    # 打印简单统计
    print(f"\n=== 简单统计 ===")
    for model_name, model_data in results.items():
        if isinstance(model_data, dict) and 'error' not in model_data:
            total_params = sum(comp_data.get('total_parameters', 0) for comp_data in model_data.values() if isinstance(comp_data, dict))
            print(f"{model_name}: {total_params:,} 参数")

def main():
    """主函数"""
    test_with_local_models()

if __name__ == "__main__":
    main()

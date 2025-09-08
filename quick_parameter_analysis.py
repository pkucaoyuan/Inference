#!/usr/bin/env python3
"""
快速模型参数量分析脚本
快速计算FLUX和LUMINA各类型层的参数量
"""

import torch
import torch.nn as nn
from diffusers import FluxPipeline, Lumina2Pipeline
from collections import defaultdict
import json
from datetime import datetime

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
            
            # 递归分析子模块
            analyze_module(child, full_name)
    
    print(f"\n=== {model_name} 参数量分析 ===")
    analyze_module(model)
    
    # 添加总计
    total_layers = sum(stats['layers'] for stats in layer_stats.values())
    layer_stats['总计'] = {
        'parameters': total_params,
        'layers': total_layers,
        'size_mb': total_size_mb
    }
    
    return dict(layer_stats), total_params, total_size_mb

def quick_analyze_flux():
    """快速分析FLUX模型"""
    print("正在加载FLUX模型...")
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 分析各个组件
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
                results[comp_name] = {
                    'layer_stats': layer_stats,
                    'total_parameters': total_params,
                    'total_size_mb': total_size,
                    'total_size_gb': total_size / 1024
                }
        
        return results
        
    except Exception as e:
        print(f"❌ FLUX模型分析失败: {e}")
        return {'error': str(e)}

def quick_analyze_lumina():
    """快速分析LUMINA模型"""
    print("正在加载LUMINA模型...")
    try:
        pipe = Lumina2Pipeline.from_pretrained(
            "Alpha-VLLM/Lumina-Image-2.0",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 分析各个组件
        components = {
            'Text Encoder': pipe.text_encoder,
            'Transformer': pipe.transformer,
            'VAE': pipe.vae
        }
        
        for comp_name, comp_model in components.items():
            if comp_model is not None:
                print(f"\n--- 分析 {comp_name} ---")
                layer_stats, total_params, total_size = analyze_model_parameters(comp_model, comp_name)
                results[comp_name] = {
                    'layer_stats': layer_stats,
                    'total_parameters': total_params,
                    'total_size_mb': total_size,
                    'total_size_gb': total_size / 1024
                }
        
        return results
        
    except Exception as e:
        print(f"❌ LUMINA模型分析失败: {e}")
        return {'error': str(e)}

def print_comparison_table(flux_data, lumina_data):
    """打印对比表格"""
    print(f"\n=== 模型参数量对比 ===")
    print(f"{'组件':<20} {'FLUX参数':<15} {'LUMINA参数':<15} {'差异':<15}")
    print("-" * 70)
    
    # 获取所有组件
    all_components = set()
    if isinstance(flux_data, dict) and 'error' not in flux_data:
        all_components.update(flux_data.keys())
    if isinstance(lumina_data, dict) and 'error' not in lumina_data:
        all_components.update(lumina_data.keys())
    
    for comp in sorted(all_components):
        flux_params = 0
        lumina_params = 0
        
        if comp in flux_data and 'total_parameters' in flux_data[comp]:
            flux_params = flux_data[comp]['total_parameters']
        
        if comp in lumina_data and 'total_parameters' in lumina_data[comp]:
            lumina_params = lumina_data[comp]['total_parameters']
        
        diff = flux_params - lumina_params
        diff_str = f"{diff:+,}" if diff != 0 else "0"
        
        print(f"{comp:<20} {flux_params:>12,} {lumina_params:>12,} {diff_str:>12}")

def print_layer_type_comparison(flux_data, lumina_data):
    """打印各层类型对比"""
    print(f"\n=== 各层类型参数量对比 ===")
    
    for model_name, model_data in [("FLUX", flux_data), ("LUMINA", lumina_data)]:
        if isinstance(model_data, dict) and 'error' not in model_data:
            print(f"\n--- {model_name} 各层类型统计 ---")
            
            # 合并所有组件的层统计
            combined_stats = defaultdict(lambda: {'parameters': 0, 'layers': 0, 'size_mb': 0})
            
            for comp_name, comp_data in model_data.items():
                if 'layer_stats' in comp_data:
                    for layer_type, stats in comp_data['layer_stats'].items():
                        if layer_type != '总计':
                            combined_stats[layer_type]['parameters'] += stats['parameters']
                            combined_stats[layer_type]['layers'] += stats['layers']
                            combined_stats[layer_type]['size_mb'] += stats['size_mb']
            
            # 按参数量排序
            sorted_layers = sorted(combined_stats.items(), 
                                 key=lambda x: x[1]['parameters'], reverse=True)
            
            print(f"{'层类型':<20} {'参数数量':<15} {'层数':<10} {'大小(MB)':<12}")
            print("-" * 60)
            
            for layer_type, stats in sorted_layers:
                print(f"{layer_type:<20} {stats['parameters']:>12,} {stats['layers']:>8} {stats['size_mb']:>10.2f}")

def main():
    """主函数"""
    print("开始快速模型参数量分析...")
    
    # 分析FLUX
    flux_data = quick_analyze_flux()
    
    # 分析LUMINA
    lumina_data = quick_analyze_lumina()
    
    # 打印对比表格
    print_comparison_table(flux_data, lumina_data)
    
    # 打印各层类型对比
    print_layer_type_comparison(flux_data, lumina_data)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'FLUX': flux_data,
        'LUMINA': lumina_data,
        'timestamp': timestamp
    }
    
    report_file = f"quick_parameter_analysis_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析完成！结果已保存到: {report_file}")

if __name__ == "__main__":
    main()

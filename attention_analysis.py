#!/usr/bin/env python3
"""
专门分析Attention层参数量的脚本
统计FLUX和LUMINA模型中所有Attention层的大小
"""

import torch
import torch.nn as nn
from diffusers import FluxPipeline, Lumina2Pipeline
from collections import defaultdict
import json
from datetime import datetime

def classify_attention_layers(module, full_name=""):
    """专门识别Attention层"""
    module_type = type(module).__name__
    
    # Attention相关关键词
    attention_keywords = [
        'attention', 'attn', 'self_attn', 'cross_attn', 
        'multihead', 'mha', 'qkv', 'query', 'key', 'value',
        'selfattention', 'crossattention', 'attentionblock'
    ]
    
    # 检查模块类型
    if any(keyword in module_type.lower() for keyword in attention_keywords):
        return True
    
    # 检查模块名称
    if any(keyword in full_name.lower() for keyword in attention_keywords):
        return True
    
    return False

def analyze_attention_parameters(model, model_name="Model"):
    """分析模型中的Attention层参数量"""
    attention_stats = {
        'total_parameters': 0,
        'total_size_mb': 0.0,
        'layers': 0,
        'layer_details': []
    }
    
    def analyze_module(module, prefix=""):
        children = list(module.named_children())
        
        for name, child in children:
            full_name = f"{prefix}.{name}" if prefix else name
            
            # 检查是否为叶子节点
            has_children = len(list(child.named_children())) > 0
            
            if not has_children:
                # 检查是否为Attention层
                if classify_attention_layers(child, full_name):
                    module_params = sum(p.numel() for p in child.parameters())
                    module_size_mb = sum(p.numel() * p.element_size() for p in child.parameters()) / (1024 * 1024)
                    
                    if module_params > 0:
                        attention_stats['total_parameters'] += module_params
                        attention_stats['total_size_mb'] += module_size_mb
                        attention_stats['layers'] += 1
                        
                        layer_detail = {
                            'name': full_name,
                            'parameters': module_params,
                            'size_mb': module_size_mb,
                            'module_type': type(child).__name__
                        }
                        attention_stats['layer_details'].append(layer_detail)
                        
                        print(f"  {full_name}: {module_params:,} 参数 ({module_size_mb:.2f}MB) - {type(child).__name__}")
            
            # 递归分析子模块
            analyze_module(child, full_name)
    
    print(f"\n=== {model_name} Attention层分析 ===")
    analyze_module(model)
    
    return attention_stats

def analyze_flux_attention():
    """分析FLUX模型的Attention层"""
    print("正在加载FLUX模型...")
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 分析各个组件的Attention层
        components = {
            'Text Encoder': pipe.text_encoder,
            'Text Encoder 2': pipe.text_encoder_2,
            'Transformer': pipe.transformer,
            'VAE': pipe.vae
        }
        
        for comp_name, comp_model in components.items():
            if comp_model is not None:
                print(f"\n--- 分析 {comp_name} Attention层 ---")
                attention_stats = analyze_attention_parameters(comp_model, comp_name)
                results[comp_name] = attention_stats
        
        return results
        
    except Exception as e:
        print(f"❌ FLUX模型分析失败: {e}")
        return {'error': str(e)}

def analyze_lumina_attention():
    """分析LUMINA模型的Attention层"""
    print("正在加载LUMINA模型...")
    try:
        pipe = Lumina2Pipeline.from_pretrained(
            "Alpha-VLLM/Lumina-Image-2.0",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 分析各个组件的Attention层
        components = {
            'Text Encoder': pipe.text_encoder,
            'Transformer': pipe.transformer,
            'VAE': pipe.vae
        }
        
        for comp_name, comp_model in components.items():
            if comp_model is not None:
                print(f"\n--- 分析 {comp_name} Attention层 ---")
                attention_stats = analyze_attention_parameters(comp_model, comp_name)
                results[comp_name] = attention_stats
        
        return results
        
    except Exception as e:
        print(f"❌ LUMINA模型分析失败: {e}")
        return {'error': str(e)}

def print_attention_summary(flux_data, lumina_data):
    """打印Attention层统计摘要"""
    print(f"\n=== Attention层参数量统计摘要 ===")
    print(f"{'模型':<15} {'组件':<20} {'Attention层数':<15} {'参数量':<20} {'大小(MB)':<15}")
    print("-" * 85)
    
    # FLUX统计
    if isinstance(flux_data, dict) and 'error' not in flux_data:
        flux_total_params = 0
        flux_total_layers = 0
        flux_total_size = 0.0
        
        for comp_name, comp_data in flux_data.items():
            if 'total_parameters' in comp_data:
                flux_total_params += comp_data['total_parameters']
                flux_total_layers += comp_data['layers']
                flux_total_size += comp_data['total_size_mb']
                
                print(f"{'FLUX':<15} {comp_name:<20} {comp_data['layers']:<15} {comp_data['total_parameters']:>15,} {comp_data['total_size_mb']:>12.2f}")
        
        print(f"{'FLUX':<15} {'总计':<20} {flux_total_layers:<15} {flux_total_params:>15,} {flux_total_size:>12.2f}")
    
    print()
    
    # LUMINA统计
    if isinstance(lumina_data, dict) and 'error' not in lumina_data:
        lumina_total_params = 0
        lumina_total_layers = 0
        lumina_total_size = 0.0
        
        for comp_name, comp_data in lumina_data.items():
            if 'total_parameters' in comp_data:
                lumina_total_params += comp_data['total_parameters']
                lumina_total_layers += comp_data['layers']
                lumina_total_size += comp_data['total_size_mb']
                
                print(f"{'LUMINA':<15} {comp_name:<20} {comp_data['layers']:<15} {comp_data['total_parameters']:>15,} {comp_data['total_size_mb']:>12.2f}")
        
        print(f"{'LUMINA':<15} {'总计':<20} {lumina_total_layers:<15} {lumina_total_params:>15,} {lumina_total_size:>12.2f}")
    
    # 对比
    print(f"\n=== 对比分析 ===")
    if isinstance(flux_data, dict) and 'error' not in flux_data and isinstance(lumina_data, dict) and 'error' not in lumina_data:
        flux_total = sum(comp_data.get('total_parameters', 0) for comp_data in flux_data.values() if isinstance(comp_data, dict))
        lumina_total = sum(comp_data.get('total_parameters', 0) for comp_data in lumina_data.values() if isinstance(comp_data, dict))
        
        print(f"FLUX Attention层总参数量: {flux_total:,}")
        print(f"LUMINA Attention层总参数量: {lumina_total:,}")
        print(f"差异: {flux_total - lumina_total:+,}")
        print(f"比例: {flux_total/lumina_total:.2f}x" if lumina_total > 0 else "比例: N/A")

def main():
    """主函数"""
    print("开始Attention层参数量分析...")
    
    # 分析FLUX
    flux_data = analyze_flux_attention()
    
    # 分析LUMINA
    lumina_data = analyze_lumina_attention()
    
    # 打印统计摘要
    print_attention_summary(flux_data, lumina_data)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'FLUX': flux_data,
        'LUMINA': lumina_data,
        'timestamp': timestamp
    }
    
    report_file = f"attention_analysis_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析完成！结果已保存到: {report_file}")

if __name__ == "__main__":
    main()

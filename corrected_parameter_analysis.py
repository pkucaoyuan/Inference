#!/usr/bin/env python3
"""
修正版模型参数量分析脚本
根据官方信息修正统计逻辑，只统计核心生成组件
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
    
    # Attention相关 - 更全面的识别
    attention_keywords = ['attention', 'attn', 'self_attn', 'cross_attn', 'multihead', 'mha', 'qkv', 'query', 'key', 'value']
    if any(keyword in module_type.lower() for keyword in attention_keywords):
        return "Attention层"
    
    # 检查模块名称中是否包含attention相关关键词
    if any(keyword in full_name.lower() for keyword in attention_keywords):
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

def analyze_model_parameters(model, model_name="Model", include_text_encoder=False):
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
            
            # 检查是否为叶子节点（没有子模块）
            has_children = len(list(child.named_children())) > 0
            
            if not has_children:
                # 只计算叶子节点的参数
                module_params = sum(p.numel() for p in child.parameters())
                module_size_mb = sum(p.numel() * p.element_size() for p in child.parameters()) / (1024 * 1024)
                
                if module_params > 0:
                    # 根据模块类型分类
                    layer_type = classify_layer_type(child, full_name)
                    
                    # 如果不包含文本编码器，跳过文本编码器相关层
                    if not include_text_encoder and layer_type == "文本编码器":
                        continue
                    
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
    total_layers = sum(stats['layers'] for stats in layer_stats.values())
    layer_stats['总计'] = {
        'parameters': total_params,
        'layers': total_layers,
        'size_mb': total_size_mb
    }
    
    return dict(layer_stats), total_params, total_size_mb

def corrected_analyze_flux():
    """修正版FLUX模型分析 - 只统计核心生成组件"""
    print("正在加载FLUX模型...")
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 只分析核心生成组件，不包含文本编码器
        components = {
            'Transformer (核心)': pipe.transformer,
            'VAE': pipe.vae
        }
        
        for comp_name, comp_model in components.items():
            if comp_model is not None:
                print(f"\n--- 分析 {comp_name} ---")
                layer_stats, total_params, total_size = analyze_model_parameters(
                    comp_model, comp_name, include_text_encoder=False
                )
                results[comp_name] = {
                    'layer_stats': layer_stats,
                    'total_parameters': total_params,
                    'total_size_mb': total_size,
                    'total_size_gb': total_size / 1024
                }
        
        # 单独分析文本编码器（用于对比）
        if pipe.text_encoder is not None:
            print(f"\n--- 分析 Text Encoder (对比用) ---")
            layer_stats, total_params, total_size = analyze_model_parameters(
                pipe.text_encoder, "Text Encoder", include_text_encoder=True
            )
            results['Text Encoder (对比)'] = {
                'layer_stats': layer_stats,
                'total_parameters': total_params,
                'total_size_mb': total_size,
                'total_size_gb': total_size / 1024
            }
        
        if pipe.text_encoder_2 is not None:
            print(f"\n--- 分析 Text Encoder 2 (对比用) ---")
            layer_stats, total_params, total_size = analyze_model_parameters(
                pipe.text_encoder_2, "Text Encoder 2", include_text_encoder=True
            )
            results['Text Encoder 2 (对比)'] = {
                'layer_stats': layer_stats,
                'total_parameters': total_params,
                'total_size_mb': total_size,
                'total_size_gb': total_size / 1024
            }
        
        return results
        
    except Exception as e:
        print(f"❌ FLUX模型分析失败: {e}")
        return {'error': str(e)}

def corrected_analyze_lumina():
    """修正版LUMINA模型分析 - 只统计核心生成组件"""
    print("正在加载LUMINA模型...")
    try:
        pipe = Lumina2Pipeline.from_pretrained(
            "Alpha-VLLM/Lumina-Image-2.0",
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
        results = {}
        
        # 只分析核心生成组件，不包含文本编码器
        components = {
            'Transformer (核心)': pipe.transformer,
            'VAE': pipe.vae
        }
        
        for comp_name, comp_model in components.items():
            if comp_model is not None:
                print(f"\n--- 分析 {comp_name} ---")
                layer_stats, total_params, total_size = analyze_model_parameters(
                    comp_model, comp_name, include_text_encoder=False
                )
                results[comp_name] = {
                    'layer_stats': layer_stats,
                    'total_parameters': total_params,
                    'total_size_mb': total_size,
                    'total_size_gb': total_size / 1024
                }
        
        # 单独分析文本编码器（用于对比）
        if pipe.text_encoder is not None:
            print(f"\n--- 分析 Text Encoder (对比用) ---")
            layer_stats, total_params, total_size = analyze_model_parameters(
                pipe.text_encoder, "Text Encoder", include_text_encoder=True
            )
            results['Text Encoder (对比)'] = {
                'layer_stats': layer_stats,
                'total_parameters': total_params,
                'total_size_mb': total_size,
                'total_size_gb': total_size / 1024
            }
        
        return results
        
    except Exception as e:
        print(f"❌ LUMINA模型分析失败: {e}")
        return {'error': str(e)}

def print_corrected_comparison(flux_data, lumina_data):
    """打印修正后的对比表格"""
    print(f"\n=== 修正版模型参数量对比 ===")
    print(f"{'组件':<25} {'FLUX参数':<20} {'LUMINA参数':<20} {'差异':<20}")
    print("-" * 90)
    
    # 核心组件对比
    flux_core = flux_data.get('Transformer (核心)', {})
    lumina_core = lumina_data.get('Transformer (核心)', {})
    
    flux_core_params = flux_core.get('total_parameters', 0)
    lumina_core_params = lumina_core.get('total_parameters', 0)
    
    print(f"{'核心Transformer':<25} {flux_core_params:>15,} {lumina_core_params:>15,} {flux_core_params - lumina_core_params:>+15,}")
    
    # VAE对比
    flux_vae = flux_data.get('VAE', {})
    lumina_vae = lumina_data.get('VAE', {})
    
    flux_vae_params = flux_vae.get('total_parameters', 0)
    lumina_vae_params = lumina_vae.get('total_parameters', 0)
    
    print(f"{'VAE':<25} {flux_vae_params:>15,} {lumina_vae_params:>15,} {flux_vae_params - lumina_vae_params:>+15,}")
    
    # 总计（核心组件）
    flux_total = flux_core_params + flux_vae_params
    lumina_total = lumina_core_params + lumina_vae_params
    
    print(f"{'总计(核心组件)':<25} {flux_total:>15,} {lumina_total:>15,} {flux_total - lumina_total:>+15,}")
    
    print(f"\n=== 官方信息对比 ===")
    print(f"FLUX官方: ≈12B参数 (核心生成器)")
    print(f"LUMINA官方: ≈2.6B参数")
    print(f"FLUX实际: {flux_total/1e9:.2f}B参数")
    print(f"LUMINA实际: {lumina_total/1e9:.2f}B参数")
    
    # 文本编码器信息（对比用）
    print(f"\n=== 文本编码器信息（对比用）===")
    flux_text = flux_data.get('Text Encoder (对比)', {})
    flux_text2 = flux_data.get('Text Encoder 2 (对比)', {})
    lumina_text = lumina_data.get('Text Encoder (对比)', {})
    
    if flux_text:
        print(f"FLUX Text Encoder: {flux_text.get('total_parameters', 0):,} 参数")
    if flux_text2:
        print(f"FLUX Text Encoder 2: {flux_text2.get('total_parameters', 0):,} 参数")
    if lumina_text:
        print(f"LUMINA Text Encoder: {lumina_text.get('total_parameters', 0):,} 参数")

def main():
    """主函数"""
    print("开始修正版模型参数量分析...")
    print("注意：只统计核心生成组件，不包含文本编码器")
    
    # 分析FLUX
    flux_data = corrected_analyze_flux()
    
    # 分析LUMINA
    lumina_data = corrected_analyze_lumina()
    
    # 打印修正后的对比表格
    print_corrected_comparison(flux_data, lumina_data)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'FLUX': flux_data,
        'LUMINA': lumina_data,
        'timestamp': timestamp,
        'note': '只统计核心生成组件，不包含文本编码器'
    }
    
    report_file = f"corrected_parameter_analysis_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 修正版分析完成！结果已保存到: {report_file}")

if __name__ == "__main__":
    main()

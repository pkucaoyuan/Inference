#!/usr/bin/env python3
"""
模型参数量分析脚本
计算FLUX和LUMINA各类型层的参数量统计
"""

import torch
import torch.nn as nn
from diffusers import FluxPipeline, Lumina2Pipeline
from collections import defaultdict
import json
import os
from datetime import datetime

class ModelParameterAnalyzer:
    def __init__(self, device="cuda"):
        self.device = device
        self.results = {}
        
    def count_parameters(self, model, model_name="Model"):
        """计算模型各层参数量"""
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
                        layer_type = self._classify_layer_type(child, full_name)
                        
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
    
    def _classify_layer_type(self, module, full_name):
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
    
    def analyze_flux_model(self):
        """分析FLUX模型"""
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
                    layer_stats, total_params, total_size = self.count_parameters(comp_model, comp_name)
                    results[comp_name] = {
                        'layer_stats': layer_stats,
                        'total_parameters': total_params,
                        'total_size_mb': total_size,
                        'total_size_gb': total_size / 1024
                    }
            
            self.results['FLUX'] = results
            print(f"\n✅ FLUX模型分析完成")
            
        except Exception as e:
            print(f"❌ FLUX模型分析失败: {e}")
            self.results['FLUX'] = {'error': str(e)}
    
    def analyze_lumina_model(self):
        """分析LUMINA模型"""
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
                    layer_stats, total_params, total_size = self.count_parameters(comp_model, comp_name)
                    results[comp_name] = {
                        'layer_stats': layer_stats,
                        'total_parameters': total_params,
                        'total_size_mb': total_size,
                        'total_size_gb': total_size / 1024
                    }
            
            self.results['LUMINA'] = results
            print(f"\n✅ LUMINA模型分析完成")
            
        except Exception as e:
            print(f"❌ LUMINA模型分析失败: {e}")
            self.results['LUMINA'] = {'error': str(e)}
    
    def generate_report(self):
        """生成分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_parameter_analysis_{timestamp}.json"
        
        # 保存详细结果
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 模型参数量分析报告 ===")
        print(f"报告已保存到: {report_file}")
        
        # 生成对比表格
        self._print_comparison_table()
    
    def _print_comparison_table(self):
        """打印对比表格"""
        print(f"\n=== 模型参数量对比 ===")
        print(f"{'组件':<20} {'FLUX参数':<15} {'LUMINA参数':<15} {'差异':<15}")
        print("-" * 70)
        
        flux_data = self.results.get('FLUX', {})
        lumina_data = self.results.get('LUMINA', {})
        
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
        
        # 打印各层类型统计
        self._print_layer_type_comparison()
    
    def _print_layer_type_comparison(self):
        """打印各层类型对比"""
        print(f"\n=== 各层类型参数量对比 ===")
        
        for model_name, model_data in self.results.items():
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
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始模型参数量分析...")
        print(f"使用设备: {self.device}")
        
        # 分析FLUX
        self.analyze_flux_model()
        
        # 分析LUMINA
        self.analyze_lumina_model()
        
        # 生成报告
        self.generate_report()
        
        print(f"\n✅ 分析完成！")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型参数量分析工具")
    parser.add_argument("--device", default="cuda", help="使用的设备 (cuda/cpu)")
    parser.add_argument("--flux-only", action="store_true", help="仅分析FLUX模型")
    parser.add_argument("--lumina-only", action="store_true", help="仅分析LUMINA模型")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    analyzer = ModelParameterAnalyzer(device=args.device)
    
    if args.flux_only:
        analyzer.analyze_flux_model()
    elif args.lumina_only:
        analyzer.analyze_lumina_model()
    else:
        analyzer.run_analysis()

if __name__ == "__main__":
    main()

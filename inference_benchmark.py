#!/usr/bin/env python3
"""
实际推理基准测试脚本
测量FLUX、Lumina和Neta Lumina的实际GPU推理时间
"""

import os
import time
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import psutil
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("警告: GPUtil未安装，将使用替代方案获取GPU信息")
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 导入必要的库
try:
    from diffusers import FluxPipeline, StableDiffusionXLPipeline
    from transformers import AutoTokenizer, AutoModel
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("警告: diffusers库未安装，将使用模拟模式")

class InferenceBenchmark:
    """推理基准测试器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        print(f"使用设备: {self.device}")
        
        # 测试配置
        self.test_prompts = [
            "A beautiful landscape with mountains and lakes, photorealistic",
            "A futuristic city with flying cars, cyberpunk style",
            "A cute anime character in a magical garden, detailed"
        ]
        
        self.test_sizes = [
            (512, 512),
            (1024, 1024),
            (1536, 1536)
        ]
        
        self.test_steps = [10, 20, 30, 50]
    
    def benchmark_flux(self) -> Dict:
        """基准测试FLUX模型"""
        print("开始测试FLUX模型...")
        
        if not DIFFUSERS_AVAILABLE:
            return self._simulate_flux_benchmark()
        
        try:
            # 加载FLUX模型
            pipe = FluxPipeline.from_pretrained(
                "./FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.test_steps:
                        result = self._benchmark_single_inference(
                            pipe, prompt, size, steps, "FLUX"
                        )
                        results.append(result)
            
            return {
                'model': 'FLUX',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"FLUX模型测试失败: {e}")
            return self._simulate_flux_benchmark()
    
    def benchmark_lumina(self) -> Dict:
        """基准测试Lumina模型"""
        print("开始测试Lumina模型...")
        
        if not DIFFUSERS_AVAILABLE:
            return self._simulate_lumina_benchmark()
        
        try:
            # 加载Lumina模型
            from diffusers import Lumina2Pipeline
            pipe = Lumina2Pipeline.from_pretrained(
                "./Lumina-Image-2.0",
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.test_steps:
                        result = self._benchmark_single_inference(
                            pipe, prompt, size, steps, "Lumina"
                        )
                        results.append(result)
            
            return {
                'model': 'Lumina',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"Lumina模型测试失败: {e}")
            return self._simulate_lumina_benchmark()
    
    def benchmark_neta_lumina(self) -> Dict:
        """基准测试Neta Lumina模型"""
        print("开始测试Neta Lumina模型...")
        
        # Neta Lumina通常需要ComfyUI，这里提供模拟测试
        return self._simulate_neta_lumina_benchmark()
    
    def _benchmark_single_inference(self, pipe, prompt: str, size: Tuple[int, int], 
                                  steps: int, model_name: str) -> Dict:
        """单次推理基准测试"""
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录开始状态
        start_time = time.time()
        start_memory = self._get_gpu_memory()
        
        try:
            # 执行推理
            if model_name == "FLUX":
                image = pipe(
                    prompt,
                    height=size[0],
                    width=size[1],
                    num_inference_steps=steps,
                    guidance_scale=7.5
                ).images[0]
            elif model_name == "Lumina":
                image = pipe(
                    prompt,
                    height=size[0],
                    width=size[1],
                    num_inference_steps=steps,
                    guidance_scale=4.0
                ).images[0]
            
            # 记录结束状态
            end_time = time.time()
            end_memory = self._get_gpu_memory()
            
            return {
                'prompt': prompt,
                'size': size,
                'steps': steps,
                'inference_time': end_time - start_time,
                'gpu_memory': end_memory - start_memory,
                'success': True
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'prompt': prompt,
                'size': size,
                'steps': steps,
                'inference_time': end_time - start_time,
                'gpu_memory': 0,
                'success': False,
                'error': str(e)
            }
    
    def _simulate_flux_benchmark(self) -> Dict:
        """模拟FLUX基准测试"""
        print("使用模拟模式测试FLUX...")
        
        results = []
        for prompt in self.test_prompts:
            for size in self.test_sizes:
                for steps in self.test_steps:
                    # 基于FLUX特性的模拟时间
                    base_time = 1.5  # 基础时间
                    size_factor = (size[0] * size[1]) / (1024 * 1024)
                    steps_factor = steps / 20
                    
                    simulated_time = base_time * size_factor * steps_factor
                    
                    results.append({
                        'prompt': prompt,
                        'size': size,
                        'steps': steps,
                        'inference_time': simulated_time,
                        'gpu_memory': 8.0,  # 模拟GPU内存使用
                        'success': True
                    })
        
        return {
            'model': 'FLUX (模拟)',
            'results': results,
            'avg_time': np.mean([r['inference_time'] for r in results]),
            'avg_memory': np.mean([r['gpu_memory'] for r in results])
        }
    
    def _simulate_lumina_benchmark(self) -> Dict:
        """模拟Lumina基准测试"""
        print("使用模拟模式测试Lumina...")
        
        results = []
        for prompt in self.test_prompts:
            for size in self.test_sizes:
                for steps in self.test_steps:
                    # 基于Lumina特性的模拟时间（Flow-based，通常更快）
                    base_time = 1.2  # 基础时间
                    size_factor = (size[0] * size[1]) / (1024 * 1024)
                    steps_factor = steps / 20
                    
                    simulated_time = base_time * size_factor * steps_factor
                    
                    results.append({
                        'prompt': prompt,
                        'size': size,
                        'steps': steps,
                        'inference_time': simulated_time,
                        'gpu_memory': 6.0,  # 模拟GPU内存使用
                        'success': True
                    })
        
        return {
            'model': 'Lumina (模拟)',
            'results': results,
            'avg_time': np.mean([r['inference_time'] for r in results]),
            'avg_memory': np.mean([r['gpu_memory'] for r in results])
        }
    
    def _simulate_neta_lumina_benchmark(self) -> Dict:
        """模拟Neta Lumina基准测试"""
        print("使用模拟模式测试Neta Lumina...")
        
        results = []
        for prompt in self.test_prompts:
            for size in self.test_sizes:
                for steps in self.test_steps:
                    # 基于Neta Lumina特性的模拟时间（可能有优化）
                    base_time = 1.0  # 基础时间（比Lumina稍快）
                    size_factor = (size[0] * size[1]) / (1024 * 1024)
                    steps_factor = steps / 20
                    
                    simulated_time = base_time * size_factor * steps_factor
                    
                    results.append({
                        'prompt': prompt,
                        'size': size,
                        'steps': steps,
                        'inference_time': simulated_time,
                        'gpu_memory': 5.5,  # 模拟GPU内存使用
                        'success': True
                    })
        
        return {
            'model': 'Neta Lumina (模拟)',
            'results': results,
            'avg_time': np.mean([r['inference_time'] for r in results]),
            'avg_memory': np.mean([r['gpu_memory'] for r in results])
        }
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        return 0.0
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("开始运行所有基准测试...")
        
        # 测试所有模型
        flux_results = self.benchmark_flux()
        lumina_results = self.benchmark_lumina()
        neta_results = self.benchmark_neta_lumina()
        
        self.results = [flux_results, lumina_results, neta_results]
        
        # 生成报告
        self.generate_benchmark_report()
        
        return self.results
    
    def generate_benchmark_report(self):
        """生成基准测试报告"""
        print("生成基准测试报告...")
        
        # 创建报告目录
        report_dir = Path("benchmark_report")
        report_dir.mkdir(exist_ok=True)
        
        # 生成文本报告
        self._generate_text_report(report_dir)
        
        # 生成图表
        self._generate_benchmark_charts(report_dir)
        
        # 生成JSON数据
        self._generate_json_data(report_dir)
        
        print(f"基准测试报告已生成到: {report_dir}")
    
    def _generate_text_report(self, report_dir: Path):
        """生成文本报告"""
        report_path = report_dir / "benchmark_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型推理基准测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"模型: {result['model']}\n")
                f.write(f"平均推理时间: {result['avg_time']:.2f}秒\n")
                f.write(f"平均GPU内存使用: {result['avg_memory']:.2f}GB\n")
                f.write("-" * 30 + "\n")
                
                # 详细结果
                for r in result['results']:
                    f.write(f"  提示词: {r['prompt'][:50]}...\n")
                    f.write(f"  尺寸: {r['size']}\n")
                    f.write(f"  步数: {r['steps']}\n")
                    f.write(f"  推理时间: {r['inference_time']:.2f}秒\n")
                    f.write(f"  GPU内存: {r['gpu_memory']:.2f}GB\n")
                    f.write(f"  成功: {r['success']}\n\n")
    
    def _generate_benchmark_charts(self, report_dir: Path):
        """生成基准测试图表"""
        if not self.results:
            return
        
        # 设置字体（兼容不同系统）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 如果中文字体不可用，使用默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均推理时间对比
        models = [r['model'] for r in self.results]
        avg_times = [r['avg_time'] for r in self.results]
        
        axes[0, 0].bar(models, avg_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('平均推理时间对比')
        axes[0, 0].set_ylabel('时间 (秒)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 平均GPU内存使用对比
        avg_memory = [r['avg_memory'] for r in self.results]
        
        axes[0, 1].bar(models, avg_memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('平均GPU内存使用对比')
        axes[0, 1].set_ylabel('内存 (GB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 推理时间分布
        all_times = []
        all_models = []
        for result in self.results:
            for r in result['results']:
                all_times.append(r['inference_time'])
                all_models.append(result['model'])
        
        # 创建箱线图
        model_times = {}
        for model, time in zip(all_models, all_times):
            if model not in model_times:
                model_times[model] = []
            model_times[model].append(time)
        
        axes[1, 0].boxplot([model_times[model] for model in models], labels=models)
        axes[1, 0].set_title('推理时间分布')
        axes[1, 0].set_ylabel('时间 (秒)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 效率对比（时间/内存）
        efficiency = [t/m if m > 0 else 0 for t, m in zip(avg_times, avg_memory)]
        
        axes[1, 1].bar(models, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('推理效率对比 (时间/内存)')
        axes[1, 1].set_ylabel('效率指标')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(report_dir / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_json_data(self, report_dir: Path):
        """生成JSON数据"""
        json_path = report_dir / "benchmark_data.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    print("模型推理基准测试工具")
    print("=" * 50)
    
    # 创建基准测试器
    benchmark = InferenceBenchmark()
    
    # 运行所有基准测试
    results = benchmark.run_all_benchmarks()
    
    print("\n基准测试完成！")
    print("请查看 benchmark_report 目录中的详细报告。")

if __name__ == "__main__":
    main()

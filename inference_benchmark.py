#!/usr/bin/env python3
"""
实际推理基准测试脚本
测量FLUX和Lumina的实际GPU推理时间
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
    
    def __init__(self, output_dir: str = "./output_images"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        
        # 使用时间戳创建唯一的输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{output_dir}_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        print(f"使用设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        
        # 测试配置
        self.test_prompts = [
            "A beautiful landscape with mountains and lakes, photorealistic",
            "A futuristic city with flying cars, cyberpunk style",
            "A cute anime character in a magical garden, detailed"
        ]
        
        self.test_sizes = [
            (1024, 1024)  # 只测试1024尺寸
        ]
        
        # 根据官方推荐设置测试步数
        self.model_recommended_steps = {
            "FLUX": [50],               # FLUX官方示例使用50步
            "Lumina": [30]              # Lumina默认30步
        }
    
    def benchmark_flux(self) -> Dict:
        """基准测试FLUX模型"""
        print("开始测试FLUX模型...")
        
        if not DIFFUSERS_AVAILABLE:
            print("错误: diffusers库不可用，无法进行真实推理测试")
            return None
        
        # 直接调用真实测试函数
        return self._real_flux_benchmark()
    
    def benchmark_lumina(self) -> Dict:
        """基准测试Lumina模型"""
        print("开始测试Lumina模型...")
        
        if not DIFFUSERS_AVAILABLE:
            print("错误: diffusers库不可用，无法进行真实推理测试")
            return None
        
        # 直接调用真实测试函数
        return self._real_lumina_benchmark()
    
    
    def _benchmark_single_inference(self, pipe, prompt: str, size: Tuple[int, int], 
                                  steps: int, model_name: str) -> Dict:
        """单次推理基准测试"""
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 记录开始状态
        start_time = time.time()
        start_memory = self._get_gpu_memory_nvidia_smi()
        if start_memory == 0.0:
            # 如果nvidia-smi失败，使用PyTorch的CUDA内存监控
            start_memory = self._get_gpu_memory()
            print(f"🔍 使用PyTorch CUDA内存监控: {start_memory:.2f}GB")
        
        try:
            # 记录推理开始时间
            inference_start_time = time.time()
            
            # 执行推理 - 使用官方推荐参数
            if model_name == "FLUX":
                # FLUX官方示例参数
                image = pipe(
                    prompt,
                    height=size[0],
                    width=size[1],
                    guidance_scale=3.5,
                    num_inference_steps=steps,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif model_name == "Lumina":
                # Lumina官方默认参数
                image = pipe(
                    prompt,
                    height=size[0],
                    width=size[1],
                    num_inference_steps=steps,
                    guidance_scale=4.0,
                    cfg_trunc_ratio=1.0,  # 官方默认值
                    cfg_normalization=True,
                    max_sequence_length=256
                ).images[0]
            
            # 记录推理结束时间
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            
            # 保存生成的图片
            save_start_time = time.time()
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{model_name.lower().replace(' ', '_')}_{size[0]}x{size[1]}_steps_{steps}_cfg_{3.5 if model_name == 'FLUX' else 4.0 if model_name == 'Lumina' else 4.5}_{safe_prompt}.png"
            image_path = self.output_dir / filename
            image.save(image_path)
            save_time = time.time() - save_start_time
            print(f"保存图片: {image_path} (耗时: {save_time:.2f}秒)")
            
            # 记录结束状态
            end_time = time.time()
            end_memory = self._get_gpu_memory_nvidia_smi()
            if end_memory == 0.0:
                # 如果nvidia-smi失败，使用PyTorch的CUDA内存监控
                end_memory = self._get_gpu_memory()
                print(f"🔍 使用PyTorch CUDA内存监控: {end_memory:.2f}GB")
            
            return {
                'prompt': prompt,
                'size': size,
                'steps': steps,
                'inference_time': inference_time,  # 使用纯推理时间
                'total_time': end_time - start_time,  # 总时间（包括保存）
                'save_time': save_time,  # 保存时间
                'gpu_memory': end_memory,  # 使用实际使用的内存，而不是变化量
                'success': True
            }
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_gpu_memory_nvidia_smi()
            if end_memory == 0.0:
                end_memory = self._get_gpu_memory()
            return {
                'prompt': prompt,
                'size': size,
                'steps': steps,
                'inference_time': end_time - start_time,
                'total_time': end_time - start_time,
                'save_time': 0.0,
                'gpu_memory': end_memory,  # 记录实际使用的内存
                'success': False,
                'error': str(e)
            }
    
    def _real_flux_benchmark(self) -> Dict:
        """真实FLUX基准测试"""
        print("开始真实FLUX模型测试...")
        
        try:
            # 尝试加载FLUX模型
            from diffusers import FluxPipeline
            
            print("正在加载FLUX模型...")
            pipe = FluxPipeline.from_pretrained(
                "./FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map="cuda"  # 使用cuda而不是auto
            )
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.model_recommended_steps["FLUX"]:
                        result = self._benchmark_single_inference(
                            pipe, prompt, size, steps, "FLUX"
                        )
                        results.append(result)
            
            return {
                'model': 'FLUX (Real Test)',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"FLUX真实测试失败: {e}")
            return None
    
    def _real_lumina_benchmark(self) -> Dict:
        """真实Lumina基准测试"""
        print("开始真实Lumina模型测试...")
        
        try:
            # 尝试加载Lumina模型
            from diffusers import Lumina2Pipeline
            
            print("正在加载Lumina模型...")
            pipe = Lumina2Pipeline.from_pretrained(
                "./Lumina-Image-2.0",
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.model_recommended_steps["Lumina"]:
                        result = self._benchmark_single_inference(
                            pipe, prompt, size, steps, "Lumina"
                        )
                        results.append(result)
            
            return {
                'model': 'Lumina (Real Test)',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"Lumina真实测试失败: {e}")
            return None
    
    
    
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存使用量"""
        if torch.cuda.is_available():
            # 获取当前分配的内存
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            # 获取缓存的内存
            cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            return allocated + cached
        return 0.0
    
    def _get_gpu_memory_nvidia_smi(self) -> float:
        """使用nvidia-smi获取GPU内存使用量"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # 处理多行输出，取第一行
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    memory_mb = float(lines[0].strip())
                    memory_gb = memory_mb / 1024.0  # 转换为GB
                    print(f"🔍 GPU内存监控: {memory_mb:.0f}MB ({memory_gb:.2f}GB)")
                    return memory_gb
                else:
                    print("⚠️ nvidia-smi输出为空")
            else:
                print(f"⚠️ nvidia-smi命令失败: {result.stderr}")
        except Exception as e:
            print(f"⚠️ GPU内存监控异常: {e}")
        return 0.0
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("开始运行所有基准测试...")
        
        # 测试所有模型
        flux_results = self.benchmark_flux()
        lumina_results = self.benchmark_lumina()
        
        # 只收集成功的结果
        self.results = []
        if flux_results:
            self.results.append(flux_results)
        if lumina_results:
            self.results.append(lumina_results)
        
        if not self.results:
            print("错误: 所有模型测试都失败了，无法生成报告")
            return []
        
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
        
        # 1. Average Inference Time Comparison
        models = [r['model'] for r in self.results]
        avg_times = [r['avg_time'] for r in self.results]
        
        axes[0, 0].bar(models, avg_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Average Inference Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average GPU Memory Usage Comparison
        avg_memory = [r['avg_memory'] for r in self.results]
        
        axes[0, 1].bar(models, avg_memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Average GPU Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (GB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Inference Time Distribution
        all_times = []
        all_models = []
        for result in self.results:
            for r in result['results']:
                all_times.append(r['inference_time'])
                all_models.append(result['model'])
        
        # Create box plot
        model_times = {}
        for model, time in zip(all_models, all_times):
            if model not in model_times:
                model_times[model] = []
            model_times[model].append(time)
        
        axes[1, 0].boxplot([model_times[model] for model in models], labels=models)
        axes[1, 0].set_title('Inference Time Distribution')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Efficiency Comparison (Time/Memory)
        efficiency = [t/m if m > 0 else 0 for t, m in zip(avg_times, avg_memory)]
        
        axes[1, 1].bar(models, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Inference Efficiency Comparison (Time/Memory)')
        axes[1, 1].set_ylabel('Efficiency Metric')
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

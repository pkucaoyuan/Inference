#!/usr/bin/env python3
"""
真实推理测试脚本
使用真实的模型进行推理测试，测量实际的GPU时间和内存消耗
"""

import os
import time
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import psutil
from pathlib import Path
import matplotlib.pyplot as plt

class RealInferenceTester:
    """真实推理测试器"""
    
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
            (1024, 1024)
        ]
        
        self.test_steps = [10, 20]
    
    def test_flux_model(self) -> Dict:
        """测试FLUX模型"""
        print("开始真实FLUX模型测试...")
        
        try:
            from diffusers import FluxPipeline
            
            print("正在加载FLUX模型...")
            pipe = FluxPipeline.from_pretrained(
                "./FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.test_steps:
                        result = self._single_inference_test(
                            pipe, prompt, size, steps, "FLUX"
                        )
                        results.append(result)
                        print(f"  {prompt[:30]}... {size} {steps}步: {result['inference_time']:.2f}s")
            
            return {
                'model': 'FLUX',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"FLUX模型测试失败: {e}")
            return None
    
    def test_lumina_model(self) -> Dict:
        """测试Lumina模型"""
        print("开始真实Lumina模型测试...")
        
        try:
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
                    for steps in self.test_steps:
                        result = self._single_inference_test(
                            pipe, prompt, size, steps, "Lumina"
                        )
                        results.append(result)
                        print(f"  {prompt[:30]}... {size} {steps}步: {result['inference_time']:.2f}s")
            
            return {
                'model': 'Lumina',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"Lumina模型测试失败: {e}")
            return None
    
    def test_neta_lumina_model(self) -> Dict:
        """测试Neta Lumina模型"""
        print("开始真实Neta Lumina模型测试...")
        
        try:
            from diffusers import Lumina2Pipeline
            
            print("正在加载Neta Lumina模型...")
            pipe = Lumina2Pipeline.from_pretrained(
                "./Neta-Lumina",
                torch_dtype=torch.bfloat16
            )
            pipe.enable_model_cpu_offload()
            
            results = []
            for prompt in self.test_prompts:
                for size in self.test_sizes:
                    for steps in self.test_steps:
                        result = self._single_inference_test(
                            pipe, prompt, size, steps, "Neta Lumina"
                        )
                        results.append(result)
                        print(f"  {prompt[:30]}... {size} {steps}步: {result['inference_time']:.2f}s")
            
            return {
                'model': 'Neta Lumina',
                'results': results,
                'avg_time': np.mean([r['inference_time'] for r in results]),
                'avg_memory': np.mean([r['gpu_memory'] for r in results])
            }
            
        except Exception as e:
            print(f"Neta Lumina模型测试失败: {e}")
            return None
    
    def _single_inference_test(self, pipe, prompt: str, size: Tuple[int, int], 
                             steps: int, model_name: str) -> Dict:
        """单次推理测试"""
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
            elif model_name in ["Lumina", "Neta Lumina"]:
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
    
    def _get_gpu_memory(self) -> float:
        """获取GPU内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        return 0.0
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始真实推理测试...")
        
        # 测试所有模型
        flux_results = self.test_flux_model()
        lumina_results = self.test_lumina_model()
        neta_results = self.test_neta_lumina_model()
        
        # 收集成功的结果
        self.results = []
        if flux_results:
            self.results.append(flux_results)
        if lumina_results:
            self.results.append(lumina_results)
        if neta_results:
            self.results.append(neta_results)
        
        # 生成报告
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """生成测试报告"""
        print("生成真实推理测试报告...")
        
        # 创建报告目录
        report_dir = Path("real_inference_report")
        report_dir.mkdir(exist_ok=True)
        
        # 生成文本报告
        self._generate_text_report(report_dir)
        
        # 生成图表
        self._generate_charts(report_dir)
        
        # 生成JSON数据
        self._generate_json_data(report_dir)
        
        print(f"真实推理测试报告已生成到: {report_dir}")
    
    def _generate_text_report(self, report_dir: Path):
        """生成文本报告"""
        report_path = report_dir / "real_inference_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("真实推理测试报告\n")
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
    
    def _generate_charts(self, report_dir: Path):
        """生成图表"""
        if not self.results:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均推理时间对比
        models = [r['model'] for r in self.results]
        avg_times = [r['avg_time'] for r in self.results]
        
        axes[0, 0].bar(models, avg_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Average Inference Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 平均GPU内存使用对比
        avg_memory = [r['avg_memory'] for r in self.results]
        
        axes[0, 1].bar(models, avg_memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Average GPU Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory (GB)')
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
        axes[1, 0].set_title('Inference Time Distribution')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 效率对比（时间/内存）
        efficiency = [t/m if m > 0 else 0 for t, m in zip(avg_times, avg_memory)]
        
        axes[1, 1].bar(models, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('Inference Efficiency Comparison (Time/Memory)')
        axes[1, 1].set_ylabel('Efficiency Metric')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(report_dir / "real_inference_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_json_data(self, report_dir: Path):
        """生成JSON数据"""
        json_path = report_dir / "real_inference_data.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    print("真实推理测试工具")
    print("=" * 50)
    
    # 创建测试器
    tester = RealInferenceTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    print("\n真实推理测试完成！")
    print("请查看 real_inference_report 目录中的详细报告。")

if __name__ == "__main__":
    main()

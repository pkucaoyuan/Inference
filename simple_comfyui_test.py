#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的ComfyUI测试工具
直接使用ComfyUI的WebSocket API或简单的HTTP请求
"""

import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import psutil
import requests

class SimpleComfyUITester:
    """简化的ComfyUI测试器"""
    
    def __init__(self, comfyui_port=8188):
        self.comfyui_port = comfyui_port
        self.comfyui_url = f"http://localhost:{comfyui_port}"
        self.results = []
        
    def get_gpu_memory(self):
        """获取GPU内存使用量"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    memory_mb = float(lines[0])
                    return memory_mb / 1024.0  # 转换为GB
        except Exception as e:
            print(f"获取GPU内存失败: {e}")
        
        return 0.0
    
    def get_system_memory(self):
        """获取系统内存使用量"""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024**3)  # 转换为GB
        except Exception as e:
            print(f"获取系统内存失败: {e}")
            return 0.0
    
    def check_comfyui_status(self):
        """检查ComfyUI状态"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def create_simple_workflow(self, prompt, negative_prompt="", steps=30, cfg=4.0):
        """创建简单的工作流"""
        # 创建一个简化的Neta Lumina工作流
        workflow = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {},
                "widgets_values": ["neta-lumina-v1.0.safetensors", "default"]
            },
            "2": {
                "class_type": "ModelSamplingAuraFlow", 
                "inputs": {"model": ["1", 0]},
                "widgets_values": [6]
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["9", 0]
                },
                "widgets_values": [int(time.time()) % 1000000, "randomize", steps, cfg, "res_multistep", "linear_quadratic", 1]
            },
            "4": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["5", 0]
                },
                "widgets_values": []
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {},
                "widgets_values": ["ae.safetensors"]
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["8", 0]},
                "widgets_values": [prompt]
            },
            "7": {
                "class_type": "CLIPTextEncode", 
                "inputs": {"clip": ["8", 0]},
                "widgets_values": [negative_prompt]
            },
            "8": {
                "class_type": "CLIPLoader",
                "inputs": {},
                "widgets_values": ["gemma_2_2b_fp16.safetensors", "lumina2", "default"]
            },
            "9": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {},
                "widgets_values": [1024, 1024, 1]
            },
            "11": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["4", 0]},
                "widgets_values": []
            }
        }
        
        return workflow
    
    def send_inference_request(self, prompt, negative_prompt="", steps=30, cfg=4.0):
        """发送推理请求"""
        # 创建简单工作流
        workflow = self.create_simple_workflow(prompt, negative_prompt, steps, cfg)
        
        # 构建请求数据
        request_data = {
            "prompt": workflow,
            "client_id": "neta_lumina_test"
        }
        
        try:
            response = requests.post(f"{self.comfyui_url}/prompt", json=request_data)
            if response.status_code == 200:
                print("✅ 推理请求已发送")
                return True
            else:
                print(f"❌ 推理请求失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 发送推理请求失败: {e}")
            return False
    
    def wait_for_completion(self, timeout=300):
        """等待推理完成"""
        print("等待推理完成...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.comfyui_url}/queue")
                if response.status_code == 200:
                    queue_data = response.json()
                    queue_pending = queue_data.get('queue_pending', [])
                    queue_running = queue_data.get('queue_running', [])
                    
                    if not queue_pending and not queue_running:
                        print("✅ 推理完成！")
                        return True
                    
                    print(f"队列状态: 等待中 {len(queue_pending)}, 运行中 {len(queue_running)}")
                
                time.sleep(2)
            except Exception as e:
                print(f"检查队列状态失败: {e}")
                time.sleep(2)
        
        print("❌ 等待超时")
        return False
    
    def run_inference_test(self, prompt, negative_prompt="", steps=30, cfg=4.0):
        """运行推理测试"""
        print(f"\n开始推理测试...")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"步数: {steps}, CFG: {cfg}")
        
        # 记录开始状态
        start_time = time.time()
        start_gpu_memory = self.get_gpu_memory()
        start_system_memory = self.get_system_memory()
        
        print(f"开始状态 - GPU内存: {start_gpu_memory:.2f}GB, 系统内存: {start_system_memory:.2f}GB")
        
        # 发送推理请求
        if not self.send_inference_request(prompt, negative_prompt, steps, cfg):
            return None
        
        # 等待完成
        if not self.wait_for_completion():
            return None
        
        # 记录结束状态
        end_time = time.time()
        end_gpu_memory = self.get_gpu_memory()
        end_system_memory = self.get_system_memory()
        
        # 计算统计信息
        inference_time = end_time - start_time
        gpu_memory_used = end_gpu_memory - start_gpu_memory
        system_memory_used = end_system_memory - start_system_memory
        
        result = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'cfg': cfg,
            'inference_time': inference_time,
            'start_gpu_memory': start_gpu_memory,
            'end_gpu_memory': end_gpu_memory,
            'gpu_memory_used': gpu_memory_used,
            'start_system_memory': start_system_memory,
            'end_system_memory': end_system_memory,
            'system_memory_used': system_memory_used,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"推理完成 - 时间: {inference_time:.2f}秒")
        print(f"GPU内存变化: {gpu_memory_used:+.2f}GB")
        print(f"系统内存变化: {system_memory_used:+.2f}GB")
        
        return result
    
    def run_batch_tests(self):
        """运行批量测试"""
        test_configs = [
            {
                "prompt": "A beautiful anime character in a magical garden, detailed, high quality",
                "negative_prompt": "",
                "steps": 30,
                "cfg": 4.0
            },
            {
                "prompt": "A futuristic city with flying cars, cyberpunk style, anime",
                "negative_prompt": "blurry, low quality",
                "steps": 30,
                "cfg": 4.5
            },
            {
                "prompt": "A cute cat in a cozy room, warm lighting, detailed",
                "negative_prompt": "",
                "steps": 20,
                "cfg": 3.5
            }
        ]
        
        print("简化ComfyUI Neta Lumina批量推理测试")
        print("=" * 50)
        
        # 检查ComfyUI状态
        if not self.check_comfyui_status():
            print("❌ ComfyUI未运行，请先启动ComfyUI")
            return []
        
        print("✅ ComfyUI连接正常")
        
        # 运行测试
        for i, config in enumerate(test_configs, 1):
            print(f"\n测试 {i}/{len(test_configs)}")
            result = self.run_inference_test(**config)
            if result:
                self.results.append(result)
            else:
                print(f"❌ 测试 {i} 失败")
        
        return self.results
    
    def save_results(self, filename=None):
        """保存测试结果"""
        if not self.results:
            print("没有测试结果可保存")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_comfyui_neta_lumina_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试结果已保存: {filename}")
        return filename
    
    def print_summary(self):
        """打印测试总结"""
        if not self.results:
            print("没有测试结果")
            return
        
        print("\n" + "=" * 50)
        print("简化ComfyUI Neta Lumina测试总结")
        print("=" * 50)
        
        total_tests = len(self.results)
        total_time = sum(r['inference_time'] for r in self.results)
        avg_time = total_time / total_tests
        
        print(f"总测试数: {total_tests}")
        print(f"总推理时间: {total_time:.2f}秒")
        print(f"平均推理时间: {avg_time:.2f}秒")
        
        print("\n详细结果:")
        for i, result in enumerate(self.results, 1):
            print(f"测试 {i}:")
            print(f"  推理时间: {result['inference_time']:.2f}秒")
            print(f"  GPU内存使用: {result['gpu_memory_used']:+.2f}GB")
            print(f"  系统内存使用: {result['system_memory_used']:+.2f}GB")
            print(f"  提示词: {result['prompt'][:50]}...")
        
        print("=" * 50)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化ComfyUI Neta Lumina测试工具")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI端口")
    parser.add_argument("--prompt", default="A beautiful anime character in a magical garden, detailed, high quality", 
                       help="推理提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG值")
    parser.add_argument("--batch", action="store_true", help="运行批量测试")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = SimpleComfyUITester(args.port)
    
    if args.batch:
        # 批量测试
        results = tester.run_batch_tests()
        tester.print_summary()
        if results:
            tester.save_results()
    else:
        # 单次测试
        result = tester.run_inference_test(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            steps=args.steps,
            cfg=args.cfg
        )
        
        if result:
            tester.results = [result]
            tester.print_summary()
            tester.save_results()
        else:
            print("❌ 测试失败")

if __name__ == "__main__":
    main()

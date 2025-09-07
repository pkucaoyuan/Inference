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
import base64
from PIL import Image
import io

class SimpleComfyUITester:
    """简化的ComfyUI测试器"""
    
    def __init__(self, comfyui_port=8188):
        self.comfyui_port = comfyui_port
        self.comfyui_url = f"http://localhost:{comfyui_port}"
        self.results = []
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"neta_lumina_output_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        
    def get_gpu_memory(self):
        """获取GPU内存使用量"""
        try:
            # 获取GPU内存使用情况
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # 处理多GPU情况，取第一个GPU的内存使用量
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    # 格式: "used,total"
                    used_mb, total_mb = lines[0].split(',')
                    used_gb = float(used_mb) / 1024.0
                    total_gb = float(total_mb) / 1024.0
                    print(f"GPU内存: {used_gb:.2f}GB / {total_gb:.2f}GB")
                    return used_gb
        except Exception as e:
            print(f"获取GPU内存失败: {e}")
        
        return 0.0
    
    def save_image(self, image_data, prompt, steps, cfg, test_index):
        """保存生成的图片"""
        try:
            # 创建安全的文件名
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"neta_lumina_test{test_index}_{steps}steps_cfg{cfg}_{safe_prompt}.png"
            filepath = self.output_dir / filename
            
            # 保存图片
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"✅ 图片已保存: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"❌ 保存图片失败: {e}")
            return None
    
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
                "inputs": {
                    "unet_name": "neta-lumina-v1.0.safetensors",
                    "weight_dtype": "default"
                }
            },
            "2": {
                "class_type": "ModelSamplingAuraFlow", 
                "inputs": {
                    "model": ["1", 0],
                    "shift": 6
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["9", 0],
                    "seed": int(time.time()) % 1000000,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "res_multistep",
                    "scheduler": "linear_quadratic",
                    "denoise": 1
                }
            },
            "4": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["5", 0]
                }
            },
            "5": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "ae.safetensors"
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["8", 0],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode", 
                "inputs": {
                    "clip": ["8", 0],
                    "text": negative_prompt
                }
            },
            "8": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "type": "lumina2",
                    "clip_name": "gemma_2_2b_fp16.safetensors"
                }
            },
            "9": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                }
            },
            "11": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["4", 0]}
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
        """等待推理完成并获取图片"""
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
                        # 尝试获取生成的图片
                        image_data = self.get_generated_image()
                        if image_data is None:
                            print("尝试从输出目录获取图片...")
                            image_data = self.get_latest_image_from_output()
                        return image_data
                    
                    print(f"队列状态: 等待中 {len(queue_pending)}, 运行中 {len(queue_running)}")
                
                time.sleep(2)
            except Exception as e:
                print(f"检查队列状态失败: {e}")
                time.sleep(2)
        
        print("❌ 等待超时")
        return None
    
    def get_generated_image(self):
        """获取生成的图片"""
        try:
            # 等待一下让ComfyUI完成图片保存
            time.sleep(2)
            
            # 获取历史记录
            response = requests.get(f"{self.comfyui_url}/history")
            if response.status_code == 200:
                history = response.json()
                print(f"历史记录数量: {len(history)}")
                
                # 找到最新的完成记录
                latest_success = None
                for prompt_id, data in history.items():
                    status = data.get('status', {})
                    if status.get('status_str') == 'success':
                        latest_success = (prompt_id, data)
                
                if latest_success:
                    prompt_id, data = latest_success
                    print(f"找到成功记录: {prompt_id}")
                    outputs = data.get('outputs', {})
                    print(f"输出节点数量: {len(outputs)}")
                    
                    # 查找所有可能的图片输出节点
                    for node_id, node_output in outputs.items():
                        print(f"检查节点 {node_id}: {node_output}")
                        if 'images' in node_output:
                            images = node_output['images']
                            if images:
                                # 获取第一张图片
                                image_info = images[0]
                                image_filename = image_info.get('filename')
                                print(f"找到图片文件: {image_filename}")
                                
                                if image_filename:
                                    # 下载图片
                                    image_response = requests.get(f"{self.comfyui_url}/view?filename={image_filename}")
                                    if image_response.status_code == 200:
                                        print(f"✅ 成功下载图片: {image_filename}")
                                        return image_response.content
                                    else:
                                        print(f"❌ 下载图片失败: {image_response.status_code}")
                else:
                    print("⚠️ 未找到成功的推理记录")
                    return None
            else:
                print(f"❌ 获取历史记录失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 获取图片失败: {e}")
            return None
    
    def get_latest_image_from_output(self):
        """从ComfyUI输出目录获取最新图片"""
        try:
            # ComfyUI默认输出目录
            output_dirs = [
                "ComfyUI/output",
                "output", 
                "ComfyUI/outputs",
                "outputs"
            ]
            
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    # 获取目录中所有图片文件
                    image_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        image_files.extend(Path(output_dir).glob(ext))
                    
                    if image_files:
                        # 按修改时间排序，获取最新的
                        latest_file = max(image_files, key=os.path.getmtime)
                        print(f"找到最新图片: {latest_file}")
                        
                        # 读取图片
                        with open(latest_file, 'rb') as f:
                            return f.read()
            
            print("⚠️ 在输出目录中未找到图片文件")
            return None
            
        except Exception as e:
            print(f"❌ 从输出目录获取图片失败: {e}")
            return None
    
    def run_inference_test(self, prompt, negative_prompt="", steps=20, cfg=4.0):
        """运行推理测试"""
        print(f"\n开始推理测试...")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"步数: {steps}, CFG: {cfg}")
        
        # 记录开始状态
        start_time = time.time()
        print("正在获取开始状态...")
        start_gpu_memory = self.get_gpu_memory()
        start_system_memory = self.get_system_memory()
        
        print(f"开始状态 - GPU内存: {start_gpu_memory:.2f}GB, 系统内存: {start_system_memory:.2f}GB")
        
        # 发送推理请求
        if not self.send_inference_request(prompt, negative_prompt, steps, cfg):
            return None
        
        # 等待完成并获取图片
        image_data = self.wait_for_completion()
        if image_data is None:
            return None
        
        # 记录结束状态
        end_time = time.time()
        print("正在获取结束状态...")
        end_gpu_memory = self.get_gpu_memory()
        end_system_memory = self.get_system_memory()
        
        # 计算统计信息
        inference_time = end_time - start_time
        gpu_memory_used = end_gpu_memory - start_gpu_memory
        system_memory_used = end_system_memory - start_system_memory
        
        print(f"结束状态 - GPU内存: {end_gpu_memory:.2f}GB, 系统内存: {end_system_memory:.2f}GB")
        
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
            'image_data': image_data,
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
                "steps": 20,
                "cfg": 4.0
            },
            {
                "prompt": "A futuristic city with flying cars, cyberpunk style, anime",
                "negative_prompt": "blurry, low quality",
                "steps": 20,
                "cfg": 5.0
            },
            {
                "prompt": "A cute cat in a cozy room, warm lighting, detailed",
                "negative_prompt": "",
                "steps": 20,
                "cfg": 5.5
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
                # 保存图片
                if 'image_data' in result and result['image_data']:
                    image_path = self.save_image(
                        result['image_data'], 
                        config['prompt'], 
                        config['steps'], 
                        config['cfg'], 
                        i
                    )
                    result['image_path'] = image_path
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
        
        # 创建不包含图片数据的副本用于JSON保存
        results_for_json = []
        for result in self.results:
            result_copy = result.copy()
            if 'image_data' in result_copy:
                del result_copy['image_data']  # 移除二进制图片数据
            results_for_json.append(result_copy)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
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
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
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

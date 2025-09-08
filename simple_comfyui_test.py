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
    
    def __init__(self, comfyui_port=8188, gpu_id=0):
        self.comfyui_port = comfyui_port
        self.comfyui_url = f"http://localhost:{comfyui_port}"
        self.gpu_id = gpu_id  # 指定使用的GPU ID
        self.results = []
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"neta_lumina_output_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {self.output_dir}")
        print(f"监控GPU ID: {self.gpu_id}")
        
    def get_gpu_memory(self):
        """获取指定GPU的内存使用量"""
        try:
            # 获取指定GPU的内存使用情况
            result = subprocess.run([
                'nvidia-smi',
                f'--id={self.gpu_id}',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    # 格式: "used,total"
                    used_mb, total_mb = lines[0].split(',')
                    used_gb = float(used_mb) / 1024.0
                    total_gb = float(total_mb) / 1024.0
                    print(f"GPU {self.gpu_id} 内存: {used_gb:.2f}GB / {total_gb:.2f}GB")
                    return used_gb
        except Exception as e:
            print(f"获取GPU {self.gpu_id} 内存失败: {e}")
        
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
    
    def get_model_parameters(self):
        """获取模型各部分参数量"""
        try:
            model_info = {}
            
            # UNet模型参数量
            unet_path = Path("../ComfyUI/models/unet/neta-lumina-v1.0.safetensors")
            if unet_path.exists():
                unet_size = unet_path.stat().st_size / (1024**3)  # GB
                model_info['unet_size_gb'] = unet_size
                # 估算参数量 (假设FP16，每个参数2字节)
                model_info['unet_parameters'] = int(unet_size * 1024**3 / 2)
            
            # Text Encoder参数量
            text_encoder_path = Path("../ComfyUI/models/text_encoders/gemma_2_2b_fp16.safetensors")
            if text_encoder_path.exists():
                te_size = text_encoder_path.stat().st_size / (1024**3)  # GB
                model_info['text_encoder_size_gb'] = te_size
                model_info['text_encoder_parameters'] = int(te_size * 1024**3 / 2)
            
            # VAE参数量
            vae_path = Path("../ComfyUI/models/vae/ae.safetensors")
            if vae_path.exists():
                vae_size = vae_path.stat().st_size / (1024**3)  # GB
                model_info['vae_size_gb'] = vae_size
                model_info['vae_parameters'] = int(vae_size * 1024**3 / 2)
            
            # 计算总参数量
            total_params = sum([
                model_info.get('unet_parameters', 0),
                model_info.get('text_encoder_parameters', 0),
                model_info.get('vae_parameters', 0)
            ])
            model_info['total_parameters'] = total_params
            
            return model_info
        except Exception as e:
            print(f"获取模型参数量失败: {e}")
            return {}
    
    def get_detailed_gpu_memory(self):
        """获取指定GPU的详细内存信息"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                f'--id={self.gpu_id}',
                '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    # 格式: "used,total,free,utilization,temperature,power"
                    parts = lines[0].split(',')
                    used_mb = float(parts[0])
                    total_mb = float(parts[1])
                    free_mb = float(parts[2])
                    utilization = float(parts[3])
                    temperature = float(parts[4]) if len(parts) > 4 else 0
                    power = float(parts[5]) if len(parts) > 5 else 0
                    
                    used_gb = used_mb / 1024.0
                    total_gb = total_mb / 1024.0
                    free_gb = free_mb / 1024.0
                    
                    return {
                        'gpu_id': self.gpu_id,
                        'used_gb': used_gb,
                        'total_gb': total_gb,
                        'free_gb': free_gb,
                        'utilization_percent': utilization,
                        'temperature_c': temperature,
                        'power_watts': power
                    }
        except Exception as e:
            print(f"获取GPU {self.gpu_id} 详细内存失败: {e}")
        
        return {}
    
    def monitor_inference_progress(self, timeout=300):
        """监控推理进度，获取各阶段时间"""
        print("开始监控推理进度...")
        
        start_time = time.time()
        progress_data = {
            'text_encoding_start': None,
            'text_encoding_end': None,
            'unet_start': None,
            'unet_end': None,
            'vae_decode_start': None,
            'vae_decode_end': None,
            'total_steps': 0,
            'current_step': 0,
            'step_times': []
        }
        
        while time.time() - start_time < timeout:
            try:
                # 检查队列状态
                response = requests.get(f"{self.comfyui_url}/queue")
                if response.status_code == 200:
                    queue_data = response.json()
                    queue_pending = queue_data.get('queue_pending', [])
                    queue_running = queue_data.get('queue_running', [])
                    
                    if not queue_pending and not queue_running:
                        print("✅ 推理完成！")
                        break
                    
                    # 检查历史记录获取进度信息
                    history_response = requests.get(f"{self.comfyui_url}/history")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        # 这里可以解析历史记录获取更详细的进度信息
                        # 由于ComfyUI API限制，我们主要依赖队列状态
                
                time.sleep(1)  # 更频繁的检查
            except Exception as e:
                print(f"监控进度失败: {e}")
                time.sleep(2)
        
        # 计算各阶段时间
        if progress_data['text_encoding_start'] and progress_data['text_encoding_end']:
            progress_data['text_encoding_time'] = progress_data['text_encoding_end'] - progress_data['text_encoding_start']
        
        if progress_data['unet_start'] and progress_data['unet_end']:
            progress_data['unet_time'] = progress_data['unet_end'] - progress_data['unet_start']
        
        if progress_data['vae_decode_start'] and progress_data['vae_decode_end']:
            progress_data['vae_decode_time'] = progress_data['vae_decode_end'] - progress_data['vae_decode_start']
        
        return progress_data
    
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
                        return True
                    
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
        
        # 获取模型参数量信息
        print("正在获取模型参数量信息...")
        model_info = self.get_model_parameters()
        
        # 记录开始状态
        start_time = time.time()
        print("正在获取开始状态...")
        start_gpu_memory = self.get_gpu_memory()
        start_detailed_gpu = self.get_detailed_gpu_memory()
        start_system_memory = self.get_system_memory()
        
        print(f"开始状态 - GPU内存: {start_gpu_memory:.2f}GB, 系统内存: {start_system_memory:.2f}GB")
        if start_detailed_gpu:
            print(f"GPU利用率: {start_detailed_gpu.get('utilization_percent', 0):.1f}%")
        
        # 发送推理请求
        request_time = time.time()
        if not self.send_inference_request(prompt, negative_prompt, steps, cfg):
            return None
        
        # 监控推理进度
        completion_time = time.time()
        progress_data = self.monitor_inference_progress()
        
        # 暂时跳过图片获取，专注于性能数据
        image_data = None
        
        # 记录结束状态
        end_time = time.time()
        print("正在获取结束状态...")
        end_gpu_memory = self.get_gpu_memory()
        end_detailed_gpu = self.get_detailed_gpu_memory()
        end_system_memory = self.get_system_memory()
        
        # 计算各部分时间
        total_inference_time = end_time - start_time
        request_time_taken = completion_time - request_time
        processing_time = end_time - completion_time
        
        # 计算统计信息
        gpu_memory_used = end_gpu_memory - start_gpu_memory
        system_memory_used = end_system_memory - start_system_memory
        
        print(f"结束状态 - GPU内存: {end_gpu_memory:.2f}GB, 系统内存: {end_system_memory:.2f}GB")
        if end_detailed_gpu:
            print(f"GPU利用率: {end_detailed_gpu.get('utilization_percent', 0):.1f}%")
        
        result = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'cfg': cfg,
            'total_inference_time': total_inference_time,
            'request_time': request_time_taken,
            'processing_time': processing_time,
            'start_gpu_memory': start_gpu_memory,
            'end_gpu_memory': end_gpu_memory,
            'gpu_memory_used': gpu_memory_used,
            'start_system_memory': start_system_memory,
            'end_system_memory': end_system_memory,
            'system_memory_used': system_memory_used,
            'model_parameters': model_info,
            'gpu_details_start': start_detailed_gpu,
            'gpu_details_end': end_detailed_gpu,
            'progress_data': progress_data,
            'gpu_id': self.gpu_id,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"推理完成 - 总时间: {total_inference_time:.2f}秒")
        print(f"  - 请求时间: {request_time_taken:.2f}秒")
        print(f"  - 处理时间: {processing_time:.2f}秒")
        print(f"GPU内存变化: {gpu_memory_used:+.2f}GB")
        print(f"系统内存变化: {system_memory_used:+.2f}GB")
        
        # 打印模型参数量信息
        if model_info:
            print(f"模型参数量:")
            if 'unet_parameters' in model_info:
                print(f"  - UNet: {model_info['unet_parameters']:,} 参数 ({model_info.get('unet_size_gb', 0):.2f}GB)")
            if 'text_encoder_parameters' in model_info:
                print(f"  - Text Encoder: {model_info['text_encoder_parameters']:,} 参数 ({model_info.get('text_encoder_size_gb', 0):.2f}GB)")
            if 'vae_parameters' in model_info:
                print(f"  - VAE: {model_info['vae_parameters']:,} 参数 ({model_info.get('vae_size_gb', 0):.2f}GB)")
            if 'total_parameters' in model_info:
                print(f"  - 总计: {model_info['total_parameters']:,} 参数")
        
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
                self.results.append(result)
                print(f"✅ 测试 {i} 完成")
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
        
        # 直接保存结果（不包含图片数据）
        results_for_json = self.results
        
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
        successful_tests = len([r for r in self.results if r])
        
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"失败测试: {total_tests - successful_tests}")
        
        if successful_tests > 0:
            # 基本性能统计
            avg_total_time = sum(r.get('total_inference_time', 0) for r in self.results if r) / successful_tests
            avg_request_time = sum(r.get('request_time', 0) for r in self.results if r) / successful_tests
            avg_processing_time = sum(r.get('processing_time', 0) for r in self.results if r) / successful_tests
            
            print(f"\n时间统计:")
            print(f"  平均总推理时间: {avg_total_time:.2f}秒")
            print(f"  平均请求时间: {avg_request_time:.2f}秒")
            print(f"  平均处理时间: {avg_processing_time:.2f}秒")
            
            # 内存统计
            avg_gpu_memory = sum(r.get('gpu_memory_used', 0) for r in self.results if r) / successful_tests
            avg_system_memory = sum(r.get('system_memory_used', 0) for r in self.results if r) / successful_tests
            
            print(f"\n内存统计:")
            print(f"  平均GPU内存使用: {avg_gpu_memory:.2f}GB")
            print(f"  平均系统内存使用: {avg_system_memory:.2f}GB")
            
            # 模型参数量信息（从第一个结果获取）
            first_result = self.results[0]
            if 'model_parameters' in first_result and first_result['model_parameters']:
                model_info = first_result['model_parameters']
                print(f"\n模型参数量:")
                if 'unet_parameters' in model_info:
                    print(f"  UNet: {model_info['unet_parameters']:,} 参数 ({model_info.get('unet_size_gb', 0):.2f}GB)")
                if 'text_encoder_parameters' in model_info:
                    print(f"  Text Encoder: {model_info['text_encoder_parameters']:,} 参数 ({model_info.get('text_encoder_size_gb', 0):.2f}GB)")
                if 'vae_parameters' in model_info:
                    print(f"  VAE: {model_info['vae_parameters']:,} 参数 ({model_info.get('vae_size_gb', 0):.2f}GB)")
                if 'total_parameters' in model_info:
                    print(f"  总计: {model_info['total_parameters']:,} 参数")
            
            # GPU利用率统计
            gpu_utilizations = []
            for r in self.results:
                if r and 'gpu_details_start' in r and r['gpu_details_start']:
                    gpu_utilizations.append(r['gpu_details_start'].get('utilization_percent', 0))
            
            if gpu_utilizations:
                avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations)
                print(f"\nGPU利用率: {avg_gpu_util:.1f}%")
        
        print("\n详细结果:")
        for i, result in enumerate(self.results, 1):
            if result:
                print(f"测试 {i}:")
                print(f"  总推理时间: {result.get('total_inference_time', 0):.2f}秒")
                print(f"  请求时间: {result.get('request_time', 0):.2f}秒")
                print(f"  处理时间: {result.get('processing_time', 0):.2f}秒")
                print(f"  GPU内存使用: {result.get('gpu_memory_used', 0):+.2f}GB")
                print(f"  系统内存使用: {result.get('system_memory_used', 0):+.2f}GB")
                print(f"  提示词: {result.get('prompt', 'N/A')[:50]}...")
            else:
                print(f"测试 {i}: 失败")
        
        print("=" * 50)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简化ComfyUI Neta Lumina测试工具")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI端口")
    parser.add_argument("--gpu-id", type=int, default=0, help="指定使用的GPU ID")
    parser.add_argument("--prompt", default="A beautiful anime character in a magical garden, detailed, high quality", 
                       help="推理提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG值")
    parser.add_argument("--batch", action="store_true", help="运行批量测试")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = SimpleComfyUITester(args.port, args.gpu_id)
    
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

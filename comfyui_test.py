#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Neta Lumina测试工具
集成模型下载、启动、性能监控和推理测试
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
from huggingface_hub import hf_hub_download

class ComfyUITester:
    """ComfyUI测试器"""
    
    def __init__(self, comfyui_port=8188):
        self.comfyui_port = comfyui_port
        self.comfyui_url = f"http://localhost:{comfyui_port}"
        self.comfyui_path = None
        self.results = []
        
    def find_comfyui(self):
        """查找ComfyUI安装路径"""
        possible_paths = [
            Path("./ComfyUI"),
            Path("../ComfyUI"),
            Path("~/ComfyUI").expanduser(),
            Path("~/comfyui").expanduser(),
            Path("/opt/ComfyUI"),
            Path("/usr/local/ComfyUI")
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "main.py").exists():
                self.comfyui_path = path
                print(f"✅ 找到ComfyUI: {path}")
                return True
        
        print("❌ 未找到ComfyUI安装路径")
        return False
    
    def download_models(self):
        """下载Neta Lumina模型文件到ComfyUI"""
        if not self.comfyui_path:
            print("❌ ComfyUI路径未设置")
            return False
        
        print("开始下载Neta Lumina模型文件...")
        
        # 创建模型目录
        models_dir = self.comfyui_path / "models"
        unet_dir = models_dir / "unet"
        text_encoder_dir = models_dir / "text_encoders"
        vae_dir = models_dir / "vae"
        
        for dir_path in [unet_dir, text_encoder_dir, vae_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 下载模型文件
        models_to_download = [
            {
                "name": "UNet",
                "repo_id": "neta-art/Neta-Lumina",
                "filename": "Unet/neta-lumina-v1.0.safetensors",
                "local_path": unet_dir / "neta-lumina-v1.0.safetensors"
            },
            {
                "name": "Text Encoder",
                "repo_id": "neta-art/Neta-Lumina", 
                "filename": "Text Encoder/gemma_2_2b_fp16.safetensors",
                "local_path": text_encoder_dir / "gemma_2_2b_fp16.safetensors"
            },
            {
                "name": "VAE",
                "repo_id": "neta-art/Neta-Lumina",
                "filename": "VAE/ae.safetensors", 
                "local_path": vae_dir / "ae.safetensors"
            }
        ]
        
        for model in models_to_download:
            print(f"\n下载 {model['name']}...")
            try:
                # 检查文件是否已存在
                if model['local_path'].exists():
                    size_gb = model['local_path'].stat().st_size / (1024**3)
                    print(f"✅ {model['name']} 已存在 ({size_gb:.2f} GB)")
                    continue
                
                # 下载文件
                downloaded_path = hf_hub_download(
                    repo_id=model['repo_id'],
                    filename=model['filename'],
                    local_dir=model['local_path'].parent,
                    local_dir_use_symlinks=False
                )
                
                # 重命名文件
                if downloaded_path != str(model['local_path']):
                    Path(downloaded_path).rename(model['local_path'])
                
                size_gb = model['local_path'].stat().st_size / (1024**3)
                print(f"✅ {model['name']} 下载完成 ({size_gb:.2f} GB)")
                
            except Exception as e:
                print(f"❌ {model['name']} 下载失败: {e}")
                return False
        
        print("\n✅ 所有模型文件下载完成！")
        return True
    
    def check_models(self):
        """检查模型文件完整性"""
        if not self.comfyui_path:
            return False
        
        models_dir = self.comfyui_path / "models"
        required_files = {
            "UNet": models_dir / "unet" / "neta-lumina-v1.0.safetensors",
            "Text Encoder": models_dir / "text_encoders" / "gemma_2_2b_fp16.safetensors",
            "VAE": models_dir / "vae" / "ae.safetensors"
        }
        
        missing_files = []
        for name, path in required_files.items():
            if path.exists():
                size_gb = path.stat().st_size / (1024**3)
                print(f"✅ {name}: {path} ({size_gb:.2f} GB)")
            else:
                print(f"❌ {name}: 缺失 {path}")
                missing_files.append(name)
        
        if missing_files:
            print(f"\n缺少文件: {', '.join(missing_files)}")
            return False
        
        return True
    
    def start_comfyui(self):
        """启动ComfyUI"""
        if not self.comfyui_path:
            print("❌ ComfyUI路径未设置")
            return False
        
        print("启动ComfyUI...")
        
        def run_comfyui():
            os.chdir(self.comfyui_path)
            subprocess.run([sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(self.comfyui_port)])
        
        thread = threading.Thread(target=run_comfyui, daemon=True)
        thread.start()
        
        # 等待ComfyUI启动
        print("等待ComfyUI启动...")
        start_time = time.time()
        while time.time() - start_time < 60:
            try:
                response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
                if response.status_code == 200:
                    print("✅ ComfyUI已启动")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("❌ ComfyUI启动超时")
        return False
    
    def check_comfyui_status(self):
        """检查ComfyUI状态"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_gpu_memory(self):
        """获取GPU内存使用量"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # 处理多GPU情况，取第一个GPU的内存使用量
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
    
    def load_workflow(self):
        """加载工作流文件"""
        workflow_file = Path("./Neta-Lumina/lumina_workflow.json")
        if not workflow_file.exists():
            print(f"❌ 工作流文件不存在: {workflow_file}")
            return None
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            print(f"✅ 加载工作流: {workflow_file}")
            return workflow
        except Exception as e:
            print(f"❌ 加载工作流失败: {e}")
            return None
    
    def modify_workflow(self, workflow, prompt, negative_prompt="", steps=30, cfg=4.0):
        """修改工作流参数"""
        if "nodes" not in workflow:
            print("❌ 工作流格式不正确")
            return workflow
        
        # 查找文本输入节点
        text_encode_nodes = []
        sampler_nodes = []
        
        for node in workflow["nodes"]:
            if node.get("type") == "CLIPTextEncode":
                text_encode_nodes.append(node)
            elif node.get("type") == "KSampler":
                sampler_nodes.append(node)
        
        # 修改文本编码节点
        for i, node in enumerate(text_encode_nodes):
            if "widgets_values" in node and len(node["widgets_values"]) > 0:
                if i == 0:  # 第一个通常是正面提示词
                    node["widgets_values"][0] = prompt
                elif i == 1:  # 第二个通常是负面提示词
                    node["widgets_values"][0] = negative_prompt
        
        # 修改采样器节点
        for node in sampler_nodes:
            if "widgets_values" in node and len(node["widgets_values"]) >= 7:
                # widgets_values通常包含: [model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise]
                node["widgets_values"][5] = steps  # steps
                node["widgets_values"][6] = cfg    # cfg
                node["widgets_values"][4] = int(time.time()) % 1000000  # seed
        
        # 转换为ComfyUI期望的字典格式，过滤掉Note节点
        workflow_dict = {}
        for node in workflow["nodes"]:
            # 跳过Note节点（注释节点）
            if node.get("type") == "Note":
                continue
                
            node_id = node["id"]
            workflow_dict[str(node_id)] = {
                "class_type": node["type"],
                "inputs": node.get("inputs", {}),
                "widgets_values": node.get("widgets_values", [])
            }
        
        return workflow_dict
    
    def send_inference_request(self, prompt, negative_prompt="", steps=30, cfg=4.0):
        """发送推理请求"""
        # 加载工作流
        workflow = self.load_workflow()
        if not workflow:
            return False
        
        # 修改工作流参数
        workflow = self.modify_workflow(workflow, prompt, negative_prompt, steps, cfg)
        
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
        
        print("ComfyUI Neta Lumina批量推理测试")
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
            filename = f"comfyui_neta_lumina_results_{timestamp}.json"
        
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
        print("ComfyUI Neta Lumina测试总结")
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
    
    parser = argparse.ArgumentParser(description="ComfyUI Neta Lumina测试工具")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI端口")
    parser.add_argument("--prompt", default="A beautiful anime character in a magical garden, detailed, high quality", 
                       help="推理提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG值")
    parser.add_argument("--batch", action="store_true", help="运行批量测试")
    parser.add_argument("--download", action="store_true", help="下载模型文件")
    parser.add_argument("--start", action="store_true", help="启动ComfyUI")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ComfyUITester(args.port)
    
    # 查找ComfyUI
    if not tester.find_comfyui():
        print("请安装ComfyUI或指定正确的路径")
        return
    
    # 下载模型文件
    if args.download:
        if not tester.download_models():
            return
    
    # 检查模型文件
    if not tester.check_models():
        print("模型文件不完整，请先下载模型文件:")
        print("python comfyui_test.py --download")
        return
    
    # 启动ComfyUI
    if args.start:
        if not tester.start_comfyui():
            return
    
    # 检查ComfyUI状态
    if not tester.check_comfyui_status():
        print("ComfyUI未运行，请先启动ComfyUI:")
        print("python comfyui_test.py --start")
        return
    
    # 运行测试
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

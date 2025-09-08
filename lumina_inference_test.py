#!/usr/bin/env python3
"""
LUMINA模型推理测试工具
支持GPU集群环境，详细性能监控
"""

import os
import time
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import psutil
from diffusers import Lumina2Pipeline

class LuminaInferenceTester:
    """LUMINA模型推理测试器"""
    
    def __init__(self, gpu_id=0, model_path="./Lumina-Image-2.0"):
        self.gpu_id = gpu_id
        self.model_path = model_path
        # 自动检测可用的GPU设备
        self.device = self._get_available_device()
        self.results = []
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"lumina_output_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.results_dir = self.output_dir / "results"
        self.images_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"输出目录: {self.output_dir}")
        print(f"图片目录: {self.images_dir}")
        print(f"结果目录: {self.results_dir}")
        print(f"使用GPU ID: {self.gpu_id}")
        print(f"设备: {self.device}")
        
        # 加载模型
        self.pipe = None
        self.load_model()
    
    def _get_available_device(self):
        """自动检测可用的GPU设备"""
        try:
            if torch.cuda.is_available():
                # 检查指定的GPU ID是否可用
                if self.gpu_id < torch.cuda.device_count():
                    device = f"cuda:{self.gpu_id}"
                    print(f"✅ 检测到GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}")
                    return device
                else:
                    print(f"⚠️ GPU {self.gpu_id} 不存在，使用GPU 0")
                    device = "cuda:0"
                    print(f"✅ 使用GPU 0: {torch.cuda.get_device_name(0)}")
                    return device
            else:
                print("⚠️ 未检测到CUDA，使用CPU")
                return "cpu"
        except Exception as e:
            print(f"⚠️ 设备检测失败: {e}，使用CPU")
            return "cpu"
    
    def load_model(self):
        """加载LUMINA模型"""
        try:
            print("正在加载LUMINA模型...")
            # 使用更兼容的device_map策略
            if "cuda" in self.device:
                self.pipe = Lumina2Pipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"  # 使用auto让diffusers自动分配
                )
            else:
                self.pipe = Lumina2Pipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
            print("✅ LUMINA模型加载成功")
        except Exception as e:
            print(f"❌ LUMINA模型加载失败: {e}")
            print("尝试使用CPU加载...")
            try:
                self.pipe = Lumina2Pipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # CPU使用float32
                    device_map="cpu"
                )
                print("✅ LUMINA模型在CPU上加载成功")
            except Exception as e2:
                print(f"❌ CPU加载也失败: {e2}")
                self.pipe = None
    
    def get_gpu_memory(self):
        """获取当前GPU的内存使用量"""
        try:
            # 获取实际使用的GPU ID
            actual_gpu_id = self._get_actual_gpu_id()
            if actual_gpu_id is None:
                return 0.0
                
            result = subprocess.run([
                'nvidia-smi',
                f'--id={actual_gpu_id}',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    used_mb, total_mb = lines[0].split(',')
                    used_gb = float(used_mb) / 1024.0
                    total_gb = float(total_mb) / 1024.0
                    print(f"GPU {actual_gpu_id} 内存: {used_gb:.2f}GB / {total_gb:.2f}GB")
                    return used_gb
        except Exception as e:
            print(f"获取GPU内存失败: {e}")
        
        return 0.0
    
    def _get_actual_gpu_id(self):
        """获取实际使用的GPU ID"""
        try:
            if "cuda" in self.device:
                # 从device字符串中提取GPU ID
                if ":" in self.device:
                    return int(self.device.split(":")[1])
                else:
                    return 0
            return None
        except:
            return None
    
    def get_detailed_gpu_memory(self):
        """获取当前GPU的详细内存信息"""
        try:
            # 获取实际使用的GPU ID
            actual_gpu_id = self._get_actual_gpu_id()
            if actual_gpu_id is None:
                return {}
                
            result = subprocess.run([
                'nvidia-smi',
                f'--id={actual_gpu_id}',
                '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
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
                        'gpu_id': actual_gpu_id,
                        'used_gb': used_gb,
                        'total_gb': total_gb,
                        'free_gb': free_gb,
                        'utilization_percent': utilization,
                        'temperature_c': temperature,
                        'power_watts': power
                    }
        except Exception as e:
            print(f"获取GPU详细内存失败: {e}")
        
        return {}
    
    def get_system_memory(self):
        """获取系统内存使用量"""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024**3)  # 转换为GB
        except Exception as e:
            print(f"获取系统内存失败: {e}")
            return 0.0
    
    def get_model_parameters(self):
        """获取LUMINA模型参数量"""
        try:
            model_info = {}
            
            if self.pipe and hasattr(self.pipe, 'unet'):
                # UNet参数量
                unet_params = sum(p.numel() for p in self.pipe.unet.parameters())
                model_info['unet_parameters'] = unet_params
                
                # Text Encoder参数量
                if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder:
                    te_params = sum(p.numel() for p in self.pipe.text_encoder.parameters())
                    model_info['text_encoder_parameters'] = te_params
                
                # VAE参数量
                if hasattr(self.pipe, 'vae') and self.pipe.vae:
                    vae_params = sum(p.numel() for p in self.pipe.vae.parameters())
                    model_info['vae_parameters'] = vae_params
                
                # 总参数量
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
    
    def run_inference_test(self, prompt, negative_prompt="", steps=30, cfg=4.0, width=1024, height=1024):
        """运行LUMINA推理测试"""
        if not self.pipe:
            print("❌ 模型未加载")
            return None
        
        print(f"\n开始LUMINA推理测试...")
        print(f"提示词: {prompt}")
        print(f"负面提示词: {negative_prompt}")
        print(f"步数: {steps}, CFG: {cfg}, 尺寸: {width}x{height}")
        
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
        
        # 执行推理
        inference_start = time.time()
        try:
            with torch.no_grad():
                image = self.pipe(
                    prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    negative_prompt=negative_prompt,
                    max_sequence_length=256,
                    cfg_trunc_ratio=1.0,
                    cfg_normalization=True,
                    generator=torch.Generator(self.device).manual_seed(int(time.time()) % 1000000)
                ).images[0]
            
            inference_end = time.time()
            inference_time = inference_end - inference_start
            
            # 保存图片到时间戳文件夹
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lumina_{width}x{height}_steps_{steps}_cfg_{cfg}_{safe_prompt}_{timestamp_str}.png"
            image_path = self.images_dir / filename
            image.save(image_path)
            print(f"✅ 图片已保存: {image_path}")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return None
        
        # 记录结束状态
        end_time = time.time()
        print("正在获取结束状态...")
        end_gpu_memory = self.get_gpu_memory()
        end_detailed_gpu = self.get_detailed_gpu_memory()
        end_system_memory = self.get_system_memory()
        
        # 计算统计信息
        total_time = end_time - start_time
        gpu_memory_used = end_gpu_memory - start_gpu_memory
        system_memory_used = end_system_memory - start_system_memory
        
        print(f"结束状态 - GPU内存: {end_gpu_memory:.2f}GB, 系统内存: {end_system_memory:.2f}GB")
        if end_detailed_gpu:
            print(f"GPU利用率: {end_detailed_gpu.get('utilization_percent', 0):.1f}%")
        
        result = {
            'model': 'LUMINA',
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'cfg': cfg,
            'width': width,
            'height': height,
            'total_inference_time': total_time,
            'pure_inference_time': inference_time,
            'start_gpu_memory': start_gpu_memory,
            'end_gpu_memory': end_gpu_memory,
            'gpu_memory_used': gpu_memory_used,
            'start_system_memory': start_system_memory,
            'end_system_memory': end_system_memory,
            'system_memory_used': system_memory_used,
            'model_parameters': model_info,
            'gpu_details_start': start_detailed_gpu,
            'gpu_details_end': end_detailed_gpu,
            'gpu_id': self.gpu_id,
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"推理完成 - 总时间: {total_time:.2f}秒")
        print(f"  - 纯推理时间: {inference_time:.2f}秒")
        print(f"GPU内存变化: {gpu_memory_used:+.2f}GB")
        print(f"系统内存变化: {system_memory_used:+.2f}GB")
        
        # 打印模型参数量信息
        if model_info:
            print(f"模型参数量:")
            if 'unet_parameters' in model_info:
                print(f"  - UNet: {model_info['unet_parameters']:,} 参数")
            if 'text_encoder_parameters' in model_info:
                print(f"  - Text Encoder: {model_info['text_encoder_parameters']:,} 参数")
            if 'vae_parameters' in model_info:
                print(f"  - VAE: {model_info['vae_parameters']:,} 参数")
            if 'total_parameters' in model_info:
                print(f"  - 总计: {model_info['total_parameters']:,} 参数")
        
        return result
    
    def run_batch_tests(self):
        """运行批量测试"""
        test_configs = [
            {
                "prompt": "A beautiful anime character in a magical garden, detailed, high quality",
                "negative_prompt": "",
                "steps": 30,
                "cfg": 4.0,
                "width": 1024,
                "height": 1024
            },
            {
                "prompt": "A futuristic city with flying cars, cyberpunk style",
                "negative_prompt": "blurry, low quality",
                "steps": 30,
                "cfg": 4.0,
                "width": 1024,
                "height": 1024
            },
            {
                "prompt": "A cute cat in a cozy room, warm lighting, detailed",
                "negative_prompt": "",
                "steps": 30,
                "cfg": 4.0,
                "width": 1024,
                "height": 1024
            }
        ]
        
        print("LUMINA批量推理测试")
        print("=" * 50)
        
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
        """保存测试结果到时间戳文件夹"""
        if not self.results:
            print("没有测试结果可保存")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lumina_inference_results_{timestamp}.json"
        
        # 保存到results子目录
        results_path = self.results_dir / filename
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试结果已保存: {results_path}")
        return str(results_path)
    
    def organize_images(self):
        """整理生成的图片"""
        if not self.images_dir.exists():
            print("图片目录不存在")
            return
        
        # 统计图片信息
        image_files = list(self.images_dir.glob("*.png"))
        if not image_files:
            print("没有找到生成的图片")
            return
        
        print(f"\n图片整理总结:")
        print(f"总图片数量: {len(image_files)}")
        print(f"图片目录: {self.images_dir}")
        
        # 按尺寸分组统计
        size_stats = {}
        for img_file in image_files:
            # 从文件名解析尺寸信息
            filename = img_file.stem
            if "_" in filename:
                parts = filename.split("_")
                for part in parts:
                    if "x" in part and part.replace("x", "").isdigit():
                        size = part
                        size_stats[size] = size_stats.get(size, 0) + 1
                        break
        
        if size_stats:
            print(f"尺寸分布: {size_stats}")
        
        # 创建HTML画廊
        self.create_html_gallery(image_files)
    
    def create_html_gallery(self, image_files):
        """创建HTML图片画廊"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LUMINA推理结果画廊</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .image-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
        }}
        .image-card img {{
            width: 100%;
            height: 300px;
            object-fit: cover;
        }}
        .image-info {{
            padding: 15px;
        }}
        .image-title {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }}
        .image-details {{
            font-size: 14px;
            color: #666;
        }}
        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LUMINA推理结果画廊</h1>
        <p>生成时间: {timestamp}</p>
        <p>总图片数量: {total_images}</p>
    </div>
    
    <div class="gallery">
        {image_cards}
    </div>
    
    <div class="stats">
        <h3>统计信息</h3>
        <p>总图片数量: {total_images}</p>
        <p>生成时间: {timestamp}</p>
    </div>
</body>
</html>
        """
        
        # 生成图片卡片
        image_cards = ""
        for img_file in image_files:
            filename = img_file.name
            # 从文件名解析信息
            parts = filename.replace(".png", "").split("_")
            size = "未知"
            steps = "未知"
            cfg = "未知"
            
            for part in parts:
                if "x" in part and part.replace("x", "").isdigit():
                    size = part
                elif part.startswith("steps"):
                    steps = part.replace("steps", "")
                elif part.startswith("cfg"):
                    cfg = part.replace("cfg", "")
            
            image_cards += f"""
            <div class="image-card">
                <img src="{img_file.name}" alt="{filename}">
                <div class="image-info">
                    <div class="image-title">{filename}</div>
                    <div class="image-details">尺寸: {size} | 步数: {steps} | CFG: {cfg}</div>
                </div>
            </div>
            """
        
        # 填充HTML模板
        html_filled = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_images=len(image_files),
            image_cards=image_cards
        )
        
        # 保存HTML文件
        html_path = self.output_dir / "gallery.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_filled)
        
        print(f"✅ HTML画廊已生成: {html_path}")
    
    def print_summary(self):
        """打印测试总结"""
        if not self.results:
            print("没有测试结果")
            return
        
        print(f"\n{'='*50}")
        print("LUMINA推理测试总结")
        print(f"{'='*50}")
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r])
        
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"失败测试: {total_tests - successful_tests}")
        
        if successful_tests > 0:
            # 基本性能统计
            avg_total_time = sum(r.get('total_inference_time', 0) for r in self.results if r) / successful_tests
            avg_inference_time = sum(r.get('pure_inference_time', 0) for r in self.results if r) / successful_tests
            
            print(f"\n时间统计:")
            print(f"  平均总时间: {avg_total_time:.2f}秒")
            print(f"  平均纯推理时间: {avg_inference_time:.2f}秒")
            
            # 内存统计
            avg_gpu_memory = sum(r.get('gpu_memory_used', 0) for r in self.results if r) / successful_tests
            avg_system_memory = sum(r.get('system_memory_used', 0) for r in self.results if r) / successful_tests
            
            print(f"\n内存统计:")
            print(f"  平均GPU内存使用: {avg_gpu_memory:.2f}GB")
            print(f"  平均系统内存使用: {avg_system_memory:.2f}GB")
            
            # 模型参数量信息
            first_result = self.results[0]
            if 'model_parameters' in first_result and first_result['model_parameters']:
                model_info = first_result['model_parameters']
                print(f"\n模型参数量:")
                if 'unet_parameters' in model_info:
                    print(f"  UNet: {model_info['unet_parameters']:,} 参数")
                if 'text_encoder_parameters' in model_info:
                    print(f"  Text Encoder: {model_info['text_encoder_parameters']:,} 参数")
                if 'vae_parameters' in model_info:
                    print(f"  VAE: {model_info['vae_parameters']:,} 参数")
                if 'total_parameters' in model_info:
                    print(f"  总计: {model_info['total_parameters']:,} 参数")
        
        print("=" * 50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LUMINA模型推理测试工具")
    parser.add_argument("--gpu-id", type=int, default=0, help="指定使用的GPU ID")
    parser.add_argument("--model-path", default="./Lumina-Image-2.0", help="LUMINA模型路径")
    parser.add_argument("--prompt", default="A beautiful anime character in a magical garden, detailed, high quality", 
                       help="推理提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG值")
    parser.add_argument("--width", type=int, default=1024, help="图片宽度")
    parser.add_argument("--height", type=int, default=1024, help="图片高度")
    parser.add_argument("--batch", action="store_true", help="运行批量测试")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = LuminaInferenceTester(args.gpu_id, args.model_path)
    
    if args.batch:
        # 批量测试
        results = tester.run_batch_tests()
        tester.print_summary()
        tester.save_results()
        tester.organize_images()
    else:
        # 单次测试
        result = tester.run_inference_test(
            args.prompt, 
            args.negative_prompt, 
            args.steps, 
            args.cfg, 
            args.width, 
            args.height
        )
        if result:
            tester.results.append(result)
            tester.print_summary()
            tester.save_results()
            tester.organize_images()

if __name__ == "__main__":
    main()

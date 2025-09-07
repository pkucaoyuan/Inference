#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI版本的Neta Lumina推理测试
基于官方lumina_workflow.json工作流
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import torch

class ComfyUINetaLuminaTester:
    """ComfyUI版本的Neta Lumina测试器"""
    
    def __init__(self, comfyui_path: str = None):
        self.comfyui_path = Path(comfyui_path) if comfyui_path else self._find_comfyui()
        self.workflow_file = Path("./Neta-Lumina/lumina_workflow.json")
        self.output_dir = None
        
    def _find_comfyui(self) -> Path:
        """自动查找ComfyUI安装路径"""
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
                print(f"找到ComfyUI: {path}")
                return path
        
        print("未找到ComfyUI安装路径")
        print("请手动指定ComfyUI路径:")
        print("python comfyui_neta_lumina_test.py --comfyui-path /path/to/ComfyUI")
        return None
    
    def check_comfyui_setup(self) -> bool:
        """检查ComfyUI设置"""
        if not self.comfyui_path or not self.comfyui_path.exists():
            print("❌ ComfyUI路径不存在")
            return False
        
        if not (self.comfyui_path / "main.py").exists():
            print("❌ ComfyUI主程序不存在")
            return False
        
        print(f"✅ ComfyUI路径: {self.comfyui_path}")
        
        # 检查模型文件
        models_dir = self.comfyui_path / "models"
        if not models_dir.exists():
            print("❌ ComfyUI models目录不存在")
            return False
        
        # 检查Neta Lumina模型文件
        unet_dir = models_dir / "unet"
        text_encoder_dir = models_dir / "text_encoders"
        vae_dir = models_dir / "vae"
        
        required_files = {
            "UNet": unet_dir / "neta-lumina-v1.0.safetensors",
            "Text Encoder": text_encoder_dir / "gemma_2_2b_fp16.safetensors",
            "VAE": vae_dir / "ae.safetensors"
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
            print("请下载Neta Lumina模型文件到ComfyUI对应目录:")
            print("1. UNet: ComfyUI/models/unet/neta-lumina-v1.0.safetensors")
            print("2. Text Encoder: ComfyUI/models/text_encoders/gemma_2_2b_fp16.safetensors")
            print("3. VAE: ComfyUI/models/vae/ae.safetensors")
            return False
        
        return True
    
    def load_workflow(self) -> dict:
        """加载工作流配置"""
        if not self.workflow_file.exists():
            print(f"❌ 工作流文件不存在: {self.workflow_file}")
            return None
        
        try:
            with open(self.workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            print(f"✅ 加载工作流: {self.workflow_file}")
            return workflow
        except Exception as e:
            print(f"❌ 加载工作流失败: {e}")
            return None
    
    def modify_workflow(self, workflow: dict, prompt: str, negative_prompt: str = "") -> dict:
        """修改工作流参数"""
        # 查找文本输入节点
        for node_id, node in workflow.items():
            if isinstance(node, dict):
                # 查找文本输入节点
                if node.get("class_type") == "CLIPTextEncode":
                    if "text" in node.get("inputs", {}):
                        if "positive" in str(node.get("inputs", {}).get("text", "")).lower():
                            node["inputs"]["text"] = prompt
                        elif "negative" in str(node.get("inputs", {}).get("text", "")).lower():
                            node["inputs"]["text"] = negative_prompt
                
                # 修改采样参数
                elif node.get("class_type") == "KSampler":
                    inputs = node.get("inputs", {})
                    inputs["steps"] = 30
                    inputs["cfg"] = 4.0
                    inputs["seed"] = int(time.time()) % 1000000
                
                # 修改图像尺寸
                elif node.get("class_type") == "EmptySD3LatentImage":
                    inputs = node.get("inputs", {})
                    inputs["width"] = 1024
                    inputs["height"] = 1024
        
        return workflow
    
    def run_comfyui_inference(self, workflow: dict) -> str:
        """运行ComfyUI推理"""
        # 创建输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"comfyui_neta_lumina_test_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # 保存修改后的工作流
        workflow_file = self.output_dir / "workflow.json"
        with open(workflow_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 工作流已保存: {workflow_file}")
        print("请在ComfyUI中加载此工作流文件进行推理")
        print(f"输出目录: {self.output_dir}")
        
        return str(workflow_file)
    
    def create_comfyui_script(self, workflow_file: str):
        """创建ComfyUI启动脚本"""
        script_content = f'''#!/bin/bash
# ComfyUI Neta Lumina测试脚本
# 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}

echo "启动ComfyUI Neta Lumina测试..."
echo "工作流文件: {workflow_file}"
echo "输出目录: {self.output_dir}"

# 切换到ComfyUI目录
cd "{self.comfyui_path}"

# 启动ComfyUI
python main.py --listen 0.0.0.0 --port 8188

echo "ComfyUI已启动，请在浏览器中访问: http://localhost:8188"
echo "然后加载工作流文件: {workflow_file}"
'''
        
        script_file = self.output_dir / "start_comfyui.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 在Windows上创建bat文件
        bat_content = f'''@echo off
echo 启动ComfyUI Neta Lumina测试...
echo 工作流文件: {workflow_file}
echo 输出目录: {self.output_dir}

cd /d "{self.comfyui_path}"
python main.py --listen 0.0.0.0 --port 8188

echo ComfyUI已启动，请在浏览器中访问: http://localhost:8188
echo 然后加载工作流文件: {workflow_file}
pause
'''
        
        bat_file = self.output_dir / "start_comfyui.bat"
        with open(bat_file, 'w', encoding='utf-8') as f:
            f.write(bat_content)
        
        print(f"✅ 启动脚本已创建:")
        print(f"   Linux/Mac: {script_file}")
        print(f"   Windows: {bat_file}")
    
    def test_inference(self, prompt: str, negative_prompt: str = ""):
        """执行推理测试"""
        print("ComfyUI Neta Lumina推理测试")
        print("=" * 50)
        
        # 检查ComfyUI设置
        if not self.check_comfyui_setup():
            return False
        
        # 加载工作流
        workflow = self.load_workflow()
        if not workflow:
            return False
        
        # 修改工作流参数
        workflow = self.modify_workflow(workflow, prompt, negative_prompt)
        
        # 运行推理
        workflow_file = self.run_comfyui_inference(workflow)
        
        # 创建启动脚本
        self.create_comfyui_script(workflow_file)
        
        print("\n" + "=" * 50)
        print("测试准备完成！")
        print("下一步操作:")
        print("1. 运行启动脚本启动ComfyUI")
        print("2. 在浏览器中访问 http://localhost:8188")
        print("3. 加载生成的工作流文件")
        print("4. 点击Queue Prompt开始推理")
        print("5. 查看生成的图片")
        
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ComfyUI Neta Lumina推理测试")
    parser.add_argument("--comfyui-path", help="ComfyUI安装路径")
    parser.add_argument("--prompt", default="A beautiful anime character in a magical garden, detailed, high quality", 
                       help="推理提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ComfyUINetaLuminaTester(args.comfyui_path)
    
    # 执行测试
    success = tester.test_inference(args.prompt, args.negative_prompt)
    
    if not success:
        print("\n❌ 测试准备失败")
        sys.exit(1)
    
    print("\n✅ 测试准备成功！")

if __name__ == "__main__":
    main()

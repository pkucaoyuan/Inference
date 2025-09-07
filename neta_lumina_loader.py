#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neta Lumina模型加载器
支持ComfyUI格式的Neta Lumina模型
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL库不可用，无法处理图片")

class NetaLuminaLoader:
    """Neta Lumina模型加载器"""
    
    def __init__(self, model_dir: str = "./Neta-Lumina"):
        self.model_dir = Path(model_dir)
        self.workflow_file = self.model_dir / "lumina_workflow.json"
        
    def check_model_files(self) -> bool:
        """检查模型文件是否完整"""
        required_files = [
            "lumina_workflow.json",
            "README.md"
        ]
        
        for file in required_files:
            if not (self.model_dir / file).exists():
                print(f"缺少文件: {file}")
                return False
        
        # 检查是否有模型权重文件
        model_files = list(self.model_dir.rglob("*.safetensors"))
        if not model_files:
            print("未找到模型权重文件(.safetensors)")
            return False
        
        print(f"找到 {len(model_files)} 个模型文件")
        return True
    
    def load_workflow(self) -> Dict:
        """加载ComfyUI工作流"""
        try:
            with open(self.workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            print("成功加载ComfyUI工作流")
            return workflow
        except Exception as e:
            print(f"加载工作流失败: {e}")
            return {}
    
    def analyze_workflow(self, workflow: Dict) -> Dict:
        """分析工作流配置"""
        analysis = {
            "nodes": len(workflow.get("nodes", [])),
            "links": len(workflow.get("links", [])),
            "sampler_info": {},
            "model_info": {},
            "vae_info": {},
            "text_encoder_info": {}
        }
        
        # 分析节点
        for node in workflow.get("nodes", []):
            node_type = node.get("class_type", "")
            node_id = node.get("id", "")
            
            if "sampler" in node_type.lower():
                analysis["sampler_info"][node_id] = node.get("inputs", {})
            elif "model" in node_type.lower():
                analysis["model_info"][node_id] = node.get("inputs", {})
            elif "vae" in node_type.lower():
                analysis["vae_info"][node_id] = node.get("inputs", {})
            elif "text" in node_type.lower() or "clip" in node_type.lower():
                analysis["text_encoder_info"][node_id] = node.get("inputs", {})
        
        return analysis
    
    def create_simple_interface(self) -> Dict:
        """创建简化的接口配置"""
        return {
            "model_name": "Neta Lumina",
            "format": "ComfyUI",
            "recommended_settings": {
                "sampler": "res_multistep",
                "scheduler": "linear_quadratic", 
                "steps": 30,
                "guidance_scale": 4.0,
                "resolution": "1024x1024"
            },
            "supported_resolutions": [
                "1024x1024",
                "768x1532", 
                "968x1322"
            ],
            "note": "需要ComfyUI环境运行"
        }
    
    def generate_comfyui_script(self, prompt: str, negative_prompt: str = "", 
                               width: int = 1024, height: int = 1024, 
                               steps: int = 30, guidance_scale: float = 4.0) -> str:
        """生成ComfyUI运行脚本"""
        script = f"""
# Neta Lumina ComfyUI 运行脚本
# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

import json
import requests

# 工作流配置
workflow_config = {{
    "prompt": "{prompt}",
    "negative_prompt": "{negative_prompt}",
    "width": {width},
    "height": {height},
    "steps": {steps},
    "guidance_scale": {guidance_scale}
}}

# ComfyUI API调用
def run_neta_lumina():
    url = "http://localhost:8188/prompt"
    
    # 加载工作流
    with open("lumina_workflow.json", "r") as f:
        workflow = json.load(f)
    
    # 更新参数
    # 这里需要根据具体的工作流结构来更新参数
    
    # 发送请求
    response = requests.post(url, json={{"prompt": workflow}})
    return response.json()

# 运行
if __name__ == "__main__":
    result = run_neta_lumina()
    print("生成完成:", result)
"""
        return script
    
    def create_alternative_loader(self) -> str:
        """创建替代加载器代码"""
        code = '''
# 替代方案1: 使用diffusers尝试加载
def try_diffusers_load():
    try:
        from diffusers import Lumina2Pipeline
        # 尝试直接加载
        pipe = Lumina2Pipeline.from_pretrained(
            "./Neta-Lumina",
            torch_dtype=torch.bfloat16
        )
        return pipe
    except Exception as e:
        print(f"diffusers加载失败: {e}")
        return None

# 替代方案2: 手动加载组件
def manual_load_components():
    components = {}
    model_dir = Path("./Neta-Lumina")
    
    # 查找模型文件
    for component_type in ["unet", "vae", "text_encoder"]:
        files = list(model_dir.rglob(f"*{component_type}*.safetensors"))
        if files:
            components[component_type] = files[0]
    
    return components

# 替代方案3: 使用ComfyUI API
def comfyui_api_call(prompt, **kwargs):
    import requests
    import json
    
    # 构建API请求
    api_data = {
        "prompt": prompt,
        "workflow": "lumina_workflow.json",
        **kwargs
    }
    
    response = requests.post(
        "http://localhost:8188/prompt",
        json=api_data
    )
    
    return response.json()
'''
        return code

def main():
    """主函数"""
    print("Neta Lumina模型加载器")
    print("=" * 40)
    
    loader = NetaLuminaLoader()
    
    # 检查模型文件
    if not loader.check_model_files():
        print("模型文件不完整，请检查下载")
        return
    
    # 加载工作流
    workflow = loader.load_workflow()
    if not workflow:
        return
    
    # 分析工作流
    analysis = loader.analyze_workflow(workflow)
    print(f"工作流分析:")
    print(f"  节点数量: {analysis['nodes']}")
    print(f"  连接数量: {analysis['links']}")
    print(f"  采样器: {len(analysis['sampler_info'])} 个")
    print(f"  模型: {len(analysis['model_info'])} 个")
    print(f"  VAE: {len(analysis['vae_info'])} 个")
    print(f"  文本编码器: {len(analysis['text_encoder_info'])} 个")
    
    # 创建简化接口
    interface = loader.create_simple_interface()
    print(f"\\n推荐设置:")
    for key, value in interface["recommended_settings"].items():
        print(f"  {key}: {value}")
    
    # 生成ComfyUI脚本
    script = loader.generate_comfyui_script(
        "A beautiful landscape with mountains and lakes",
        "blurry, low quality",
        1024, 1024, 30, 4.0
    )
    
    script_path = Path("./neta_lumina_comfyui_script.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"\\n已生成ComfyUI脚本: {script_path}")
    
    # 生成替代加载器代码
    alt_code = loader.create_alternative_loader()
    alt_path = Path("./neta_lumina_alternative_loader.py")
    with open(alt_path, 'w', encoding='utf-8') as f:
        f.write(alt_code)
    
    print(f"已生成替代加载器: {alt_path}")
    
    print("\\n解决方案:")
    print("1. 使用ComfyUI环境运行")
    print("2. 尝试手动加载模型组件")
    print("3. 使用ComfyUI API调用")
    print("4. 等待官方diffusers支持")

if __name__ == "__main__":
    main()

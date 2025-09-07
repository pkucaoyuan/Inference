#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ComfyUI工作流加载和修改
"""

import json
from pathlib import Path

def test_workflow_loading():
    """测试工作流加载"""
    workflow_file = Path("./Neta-Lumina/lumina_workflow.json")
    
    if not workflow_file.exists():
        print(f"❌ 工作流文件不存在: {workflow_file}")
        return False
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        print(f"✅ 工作流加载成功")
        
        # 分析工作流结构
        print(f"工作流结构:")
        print(f"  - 节点数量: {len(workflow.get('nodes', []))}")
        print(f"  - 链接数量: {len(workflow.get('links', []))}")
        
        # 查找关键节点
        text_encode_nodes = []
        sampler_nodes = []
        
        for node in workflow.get("nodes", []):
            node_type = node.get("type")
            if node_type == "CLIPTextEncode":
                text_encode_nodes.append(node)
            elif node_type == "KSampler":
                sampler_nodes.append(node)
        
        print(f"  - 文本编码节点: {len(text_encode_nodes)}")
        print(f"  - 采样器节点: {len(sampler_nodes)}")
        
        # 显示文本编码节点信息
        for i, node in enumerate(text_encode_nodes):
            print(f"  文本编码节点 {i+1}:")
            print(f"    ID: {node.get('id')}")
            print(f"    位置: {node.get('pos')}")
            if "widgets_values" in node:
                print(f"    当前文本: {node['widgets_values'][0] if node['widgets_values'] else 'None'}")
        
        # 显示采样器节点信息
        for i, node in enumerate(sampler_nodes):
            print(f"  采样器节点 {i+1}:")
            print(f"    ID: {node.get('id')}")
            print(f"    位置: {node.get('pos')}")
            if "widgets_values" in node:
                values = node['widgets_values']
                print(f"    参数: steps={values[5] if len(values) > 5 else 'N/A'}, cfg={values[6] if len(values) > 6 else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流加载失败: {e}")
        return False

def test_workflow_modification():
    """测试工作流修改"""
    workflow_file = Path("./Neta-Lumina/lumina_workflow.json")
    
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # 修改工作流
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
                    node["widgets_values"][0] = "A beautiful anime character in a magical garden, detailed, high quality"
                elif i == 1:  # 第二个通常是负面提示词
                    node["widgets_values"][0] = "blurry, low quality"
        
        # 修改采样器节点
        for node in sampler_nodes:
            if "widgets_values" in node and len(node["widgets_values"]) >= 7:
                node["widgets_values"][5] = 30  # steps
                node["widgets_values"][6] = 4.0  # cfg
                node["widgets_values"][4] = 12345  # seed
        
        print("✅ 工作流修改成功")
        
        # 保存修改后的工作流
        test_workflow_file = Path("test_workflow.json")
        with open(test_workflow_file, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试工作流已保存: {test_workflow_file}")
        return True
        
    except Exception as e:
        print(f"❌ 工作流修改失败: {e}")
        return False

def main():
    """主函数"""
    print("ComfyUI工作流测试")
    print("=" * 30)
    
    # 测试工作流加载
    if not test_workflow_loading():
        return
    
    print("\n" + "=" * 30)
    
    # 测试工作流修改
    if not test_workflow_modification():
        return
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    main()

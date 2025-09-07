#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速图片整理脚本
用于快速整理现有的推理结果图片
"""

import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent))

from image_organizer import ImageOrganizer

def main():
    """主函数"""
    print("快速图片整理工具")
    print("=" * 30)
    
    # 检查可能的图片目录
    possible_dirs = [
        "./output_images",
        "./images", 
        "./results",
        "./benchmark_report",
        "./real_inference_report"
    ]
    
    found_dirs = []
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            # 检查是否包含图片文件
            image_files = list(Path(dir_path).rglob("*.png"))
            if image_files:
                found_dirs.append((dir_path, len(image_files)))
    
    if not found_dirs:
        print("未找到包含图片的目录")
        print("请确保图片文件位于以下目录之一:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        return
    
    print("找到以下包含图片的目录:")
    for i, (dir_path, count) in enumerate(found_dirs, 1):
        print(f"  {i}. {dir_path} ({count} 张图片)")
    
    # 选择目录
    if len(found_dirs) == 1:
        selected_dir = found_dirs[0][0]
        print(f"\n自动选择目录: {selected_dir}")
    else:
        try:
            choice = int(input(f"\n请选择要整理的目录 (1-{len(found_dirs)}): ")) - 1
            if 0 <= choice < len(found_dirs):
                selected_dir = found_dirs[choice][0]
            else:
                print("无效选择")
                return
        except ValueError:
            print("无效输入")
            return
    
    # 设置输出目录
    output_dir = f"{selected_dir}_organized"
    
    print(f"\n开始整理图片...")
    print(f"输入目录: {selected_dir}")
    print(f"输出目录: {output_dir}")
    
    # 创建整理器并执行整理
    organizer = ImageOrganizer(selected_dir)
    organizer.organized_dir = Path(output_dir)
    organizer.comparison_dir = organizer.organized_dir / "comparisons"
    organizer.individual_dir = organizer.organized_dir / "individual"
    
    # 重新创建目录结构
    organizer._create_directories()
    
    # 执行整理
    organizer.organize_all()
    
    print(f"\n整理完成！")
    print(f"查看HTML画廊: {organizer.organized_dir / 'gallery.html'}")

if __name__ == "__main__":
    main()

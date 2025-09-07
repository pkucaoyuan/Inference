#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片整理工具
用于整理三个模型输出的图片结果，按模型、参数、时间等维度进行分类和展示
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL库不可用，将跳过图片处理功能")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib库不可用，将跳过图表生成功能")

class ImageOrganizer:
    """图片整理器"""
    
    def __init__(self, base_dir: str = "./output_images"):
        self.base_dir = Path(base_dir)
        self.models = ["FLUX", "Lumina", "Neta_Lumina"]
        self.organized_dir = self.base_dir / "organized"
        self.comparison_dir = self.organized_dir / "comparisons"
        self.individual_dir = self.organized_dir / "individual"
        
        # 创建目录结构
        self._create_directories()
    
    def _create_directories(self):
        """创建整理后的目录结构"""
        directories = [
            self.organized_dir,
            self.comparison_dir,
            self.individual_dir
        ]
        
        # 为每个模型创建子目录
        for model in self.models:
            directories.extend([
                self.individual_dir / model,
                self.individual_dir / model / "by_size",
                self.individual_dir / model / "by_steps",
                self.individual_dir / model / "by_prompt"
            ])
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def scan_images(self) -> Dict[str, List[Dict]]:
        """扫描所有图片文件"""
        images = {model: [] for model in self.models}
        
        if not self.base_dir.exists():
            print(f"输出目录不存在: {self.base_dir}")
            return images
        
        # 扫描所有图片文件
        for image_file in self.base_dir.rglob("*.png"):
            if image_file.is_file():
                # 尝试从文件名推断模型信息
                model_info = self._parse_filename(image_file.name)
                if model_info:
                    model = model_info["model"]
                    if model in images:
                        images[model].append({
                            "path": image_file,
                            "filename": image_file.name,
                            "size": image_file.stat().st_size,
                            "modified": datetime.fromtimestamp(image_file.stat().st_mtime),
                            **model_info
                        })
        
        return images
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """从文件名解析模型信息"""
        filename_lower = filename.lower()
        
        # 模型识别
        if "flux" in filename_lower:
            model = "FLUX"
        elif "lumina" in filename_lower and "neta" not in filename_lower:
            model = "Lumina"
        elif "neta" in filename_lower:
            model = "Neta_Lumina"
        else:
            return None
        
        # 解析其他信息
        info = {"model": model}
        
        # 尝试解析尺寸 (例如: 1024x1024)
        import re
        size_match = re.search(r'(\d+)x(\d+)', filename)
        if size_match:
            info["width"] = int(size_match.group(1))
            info["height"] = int(size_match.group(2))
        
        # 尝试解析步数 (例如: steps_30)
        steps_match = re.search(r'steps[_-]?(\d+)', filename)
        if steps_match:
            info["steps"] = int(steps_match.group(1))
        
        # 尝试解析guidance scale (例如: cfg_4.5)
        cfg_match = re.search(r'cfg[_-]?([\d.]+)', filename)
        if cfg_match:
            info["guidance_scale"] = float(cfg_match.group(1))
        
        return info
    
    def organize_by_model(self, images: Dict[str, List[Dict]]):
        """按模型整理图片"""
        print("按模型整理图片...")
        
        for model, model_images in images.items():
            if not model_images:
                continue
            
            model_dir = self.individual_dir / model
            print(f"整理 {model} 模型图片: {len(model_images)} 张")
            
            for img_info in model_images:
                # 复制到模型目录
                dest_path = model_dir / img_info["filename"]
                shutil.copy2(img_info["path"], dest_path)
                
                # 按尺寸分类
                if "width" in img_info and "height" in img_info:
                    size_dir = model_dir / "by_size" / f"{img_info['width']}x{img_info['height']}"
                    size_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_info["path"], size_dir / img_info["filename"])
                
                # 按步数分类
                if "steps" in img_info:
                    steps_dir = model_dir / "by_steps" / f"steps_{img_info['steps']}"
                    steps_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_info["path"], steps_dir / img_info["filename"])
    
    def create_comparison_grids(self, images: Dict[str, List[Dict]]):
        """创建对比网格图"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib不可用，跳过对比网格生成")
            return
        
        print("创建对比网格图...")
        
        # 按尺寸和步数分组
        comparisons = {}
        for model, model_images in images.items():
            for img_info in model_images:
                if "width" in img_info and "height" in img_info and "steps" in img_info:
                    key = f"{img_info['width']}x{img_info['height']}_steps_{img_info['steps']}"
                    if key not in comparisons:
                        comparisons[key] = {}
                    comparisons[key][model] = img_info
        
        # 为每个组合创建对比图
        for key, model_images in comparisons.items():
            if len(model_images) >= 2:  # 至少有两个模型
                self._create_single_comparison(key, model_images)
    
    def _create_single_comparison(self, key: str, model_images: Dict[str, Dict]):
        """创建单个对比图"""
        try:
            fig, axes = plt.subplots(1, len(model_images), figsize=(5 * len(model_images), 5))
            if len(model_images) == 1:
                axes = [axes]
            
            for i, (model, img_info) in enumerate(model_images.items()):
                try:
                    img = Image.open(img_info["path"])
                    axes[i].imshow(img)
                    axes[i].set_title(f"{model}\n{img_info['filename']}")
                    axes[i].axis('off')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"加载失败\n{str(e)}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f"{model} (错误)")
                    axes[i].axis('off')
            
            plt.tight_layout()
            comparison_path = self.comparison_dir / f"comparison_{key}.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"创建对比图: {comparison_path}")
            
        except Exception as e:
            print(f"创建对比图失败 {key}: {e}")
    
    def generate_summary_report(self, images: Dict[str, List[Dict]]):
        """生成总结报告"""
        report_path = self.organized_dir / "summary_report.json"
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_images": sum(len(imgs) for imgs in images.values()),
            "models": {}
        }
        
        for model, model_images in images.items():
            if not model_images:
                continue
            
            model_summary = {
                "count": len(model_images),
                "sizes": {},
                "steps": {},
                "guidance_scales": {},
                "total_size_mb": sum(img["size"] for img in model_images) / (1024 * 1024)
            }
            
            # 统计尺寸分布
            for img in model_images:
                if "width" in img and "height" in img:
                    size_key = f"{img['width']}x{img['height']}"
                    model_summary["sizes"][size_key] = model_summary["sizes"].get(size_key, 0) + 1
            
            # 统计步数分布
            for img in model_images:
                if "steps" in img:
                    steps_key = str(img["steps"])
                    model_summary["steps"][steps_key] = model_summary["steps"].get(steps_key, 0) + 1
            
            # 统计guidance scale分布
            for img in model_images:
                if "guidance_scale" in img:
                    cfg_key = str(img["guidance_scale"])
                    model_summary["guidance_scales"][cfg_key] = model_summary["guidance_scales"].get(cfg_key, 0) + 1
            
            summary["models"][model] = model_summary
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"生成总结报告: {report_path}")
        
        # 打印简要统计
        self._print_summary(summary)
    
    def _print_summary(self, summary: Dict):
        """打印简要统计信息"""
        print("\n" + "="*50)
        print("图片整理总结")
        print("="*50)
        print(f"总图片数量: {summary['total_images']}")
        print(f"生成时间: {summary['generated_at']}")
        print()
        
        for model, model_summary in summary["models"].items():
            print(f"{model}:")
            print(f"  图片数量: {model_summary['count']}")
            print(f"  总大小: {model_summary['total_size_mb']:.2f} MB")
            
            if model_summary["sizes"]:
                print(f"  尺寸分布: {model_summary['sizes']}")
            if model_summary["steps"]:
                print(f"  步数分布: {model_summary['steps']}")
            if model_summary["guidance_scales"]:
                print(f"  Guidance Scale分布: {model_summary['guidance_scales']}")
            print()
    
    def create_html_gallery(self, images: Dict[str, List[Dict]]):
        """创建HTML图片画廊"""
        html_path = self.organized_dir / "gallery.html"
        
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型推理结果画廊</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header { text-align: center; margin-bottom: 30px; }
        .model-section { margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .model-title { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .image-item { text-align: center; }
        .image-item img { max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .image-info { margin-top: 10px; font-size: 14px; color: #666; }
        .comparison-section { margin-top: 40px; }
        .comparison-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>模型推理结果画廊</h1>
        <p>生成时间: {timestamp}</p>
        <p>总图片数量: {total_images}</p>
    </div>
"""
        
        # 添加各模型图片
        for model, model_images in images.items():
            if not model_images:
                continue
            
            html_content += f"""
    <div class="model-section">
        <h2 class="model-title">{model} 模型</h2>
        <div class="image-grid">
"""
            
            for img_info in model_images:
                relative_path = os.path.relpath(img_info["path"], self.organized_dir)
                info_text = f"文件: {img_info['filename']}<br>"
                if "width" in img_info and "height" in img_info:
                    info_text += f"尺寸: {img_info['width']}x{img_info['height']}<br>"
                if "steps" in img_info:
                    info_text += f"步数: {img_info['steps']}<br>"
                if "guidance_scale" in img_info:
                    info_text += f"Guidance Scale: {img_info['guidance_scale']}<br>"
                info_text += f"大小: {img_info['size']/1024:.1f} KB"
                
                html_content += f"""
            <div class="image-item">
                <img src="{relative_path}" alt="{img_info['filename']}">
                <div class="image-info">{info_text}</div>
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        # 添加对比图
        comparison_images = list(self.comparison_dir.glob("*.png"))
        if comparison_images:
            html_content += """
    <div class="comparison-section">
        <h2 class="model-title">模型对比</h2>
        <div class="comparison-grid">
"""
            for comp_img in comparison_images:
                relative_path = os.path.relpath(comp_img, self.organized_dir)
                html_content += f"""
            <div class="image-item">
                <img src="{relative_path}" alt="{comp_img.name}">
                <div class="image-info">{comp_img.stem}</div>
            </div>
"""
            html_content += """
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # 替换占位符
        html_content = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_images=sum(len(imgs) for imgs in images.values())
        )
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"创建HTML画廊: {html_path}")
    
    def organize_all(self):
        """执行完整的图片整理流程"""
        print("开始整理图片...")
        
        # 扫描图片
        images = self.scan_images()
        
        if not any(images.values()):
            print("未找到任何图片文件")
            return
        
        # 按模型整理
        self.organize_by_model(images)
        
        # 创建对比图
        self.create_comparison_grids(images)
        
        # 生成报告
        self.generate_summary_report(images)
        
        # 创建HTML画廊
        self.create_html_gallery(images)
        
        print(f"\n图片整理完成！")
        print(f"整理后的文件位于: {self.organized_dir}")
        print(f"HTML画廊: {self.organized_dir / 'gallery.html'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图片整理工具")
    parser.add_argument("--input-dir", "-i", default="./output_images", 
                       help="输入图片目录 (默认: ./output_images)")
    parser.add_argument("--output-dir", "-o", default="./organized_images",
                       help="输出整理目录 (默认: ./organized_images)")
    
    args = parser.parse_args()
    
    # 创建整理器
    organizer = ImageOrganizer(args.input_dir)
    
    # 执行整理
    organizer.organize_all()


if __name__ == "__main__":
    main()

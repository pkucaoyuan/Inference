#!/bin/bash
echo "启动ComfyUI..."

# 检查ComfyUI路径
if [ ! -f "../ComfyUI/main.py" ]; then
    echo "❌ ComfyUI路径不存在: ../ComfyUI"
    echo "请确保ComfyUI已正确安装"
    exit 1
fi

echo "✅ 找到ComfyUI: ../ComfyUI"
echo "启动ComfyUI..."

# 启动ComfyUI
cd ../ComfyUI
python main.py --listen 127.0.0.1 --port 8188

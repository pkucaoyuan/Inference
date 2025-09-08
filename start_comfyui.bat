@echo off
echo 启动ComfyUI...
cd /d "%~dp0"

REM 检查ComfyUI路径
if not exist "..\ComfyUI\main.py" (
    echo ❌ ComfyUI路径不存在: ..\ComfyUI
    echo 请确保ComfyUI已正确安装
    pause
    exit /b 1
)

echo ✅ 找到ComfyUI: ..\ComfyUI
echo 启动ComfyUI...

REM 启动ComfyUI
cd ..\ComfyUI
python main.py --listen 127.0.0.1 --port 8188

pause

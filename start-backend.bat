@echo off
title 旅迹AI - 后端服务 8001
cd /d "%~dp0backend"

if not exist ".venv-ocr\Scripts\python.exe" (
    echo.
    echo [错误] 找不到后端 Python 环境：
    echo        %~dp0backend\.venv-ocr\Scripts\python.exe
    echo.
    echo        请确认这个脚本放在 travel-agent-platform 文件夹里。
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   旅迹AI - 后端服务
echo.
echo   接口地址： http://127.0.0.1:8001
echo   接口文档： http://127.0.0.1:8001/docs
echo   健康检查： http://127.0.0.1:8001/api/v1/health
echo.
echo   启动需要几秒，看到 Application startup complete 就是好了。
echo   关闭这个窗口即可停止后端。
echo ============================================================
echo.

".venv-ocr\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8001

echo.
echo 后端已停止。
pause

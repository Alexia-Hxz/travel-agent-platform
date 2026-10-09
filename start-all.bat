@echo off
title 旅迹AI - 一键启动
cd /d "%~dp0"

echo ============================================================
echo   旅迹AI  一键启动
echo   后端 8001      前端 5173
echo ============================================================
echo.

if not exist "backend\.venv-ocr\Scripts\python.exe" (
    echo [错误] 找不到后端 Python 环境 backend\.venv-ocr
    echo        请确认本项目文件夹没有被移动过。
    echo.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [错误] 找不到前端依赖 frontend\node_modules
    echo        请先执行一次 npm ci
    echo.
    pause
    exit /b 1
)

echo 正在打开后端窗口 ...
start "旅迹AI - 后端 8001" cmd /k start-backend.bat

echo 正在打开前端窗口 ...
start "旅迹AI - 前端 5173" cmd /k start-frontend.bat

echo.
echo 等待后端就绪 ...
set n=0
:waitbe
set /a n+=1
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if not errorlevel 1 goto be_ready
if %n% geq 45 goto be_fail
ping -n 2 127.0.0.1 >nul
goto waitbe

:be_fail
echo.
echo [失败] 等待 45 秒后，后端仍未在 8001 端口监听。
echo        请查看【旅迹AI - 后端 8001】那个窗口里的报错信息。
echo.
pause
exit /b 1

:be_ready
echo 后端已就绪，继续等待前端 ...
set n=0
:waitfe
set /a n+=1
netstat -ano | findstr ":5173" | findstr "LISTENING" >nul
if not errorlevel 1 goto fe_ready
if %n% geq 45 goto fe_timeout
ping -n 2 127.0.0.1 >nul
goto waitfe

:fe_timeout
echo.
echo [提示] 前端 45 秒内还没起来，仍然为你打开浏览器。
goto open

:fe_ready
echo 前端已就绪。
echo.

:open
start "" http://127.0.0.1:5173/

echo 浏览器已打开 http://127.0.0.1:5173
echo 首次使用请先注册一个账号。
echo.
echo 这个窗口可以关闭，两个服务会继续运行。
pause
exit /b 0

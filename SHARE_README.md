# 旅迹 AI：分享版使用说明

这份压缩包供另一台电脑部署。它包含项目源码、依赖锁文件、OCR 模型、占位图和已下载的公共地点照片；不包含原作者的 API 密钥、账号、行程、收藏地点或上传文件。`backend/travel_agent.db` 是空文件，首次启动后会自动建立数据库表。

## 本地启动（Windows PowerShell）

先安装 Python 3.11、[uv](https://docs.astral.sh/uv/) 和 Node.js 20 或更新版本。在 `backend` 目录复制 `.env.example` 为 `.env`，填入自己的 `DASHSCOPE_API_KEY` 和 `PEXELS_API_KEY`。没有密钥时，账号和本地行程功能仍可使用，但 AI 规划及在线配图会受限。

打开第一个终端：

```powershell
cd backend
Copy-Item .env.example .env
# 编辑 .env，填写自己的密钥
$env:UV_PROJECT_ENVIRONMENT='.venv-ocr'
uv sync
uv run uvicorn app.main:app --reload --port 8001
```

打开第二个终端：

```powershell
cd frontend
npm ci
npm run dev
```

访问 http://127.0.0.1:5173/ 。后端健康检查地址是 http://127.0.0.1:8001/api/v1/health 。首次启动请在页面注册自己的账号。

## Docker 启动

先在 `backend` 目录把 `.env.example` 复制为 `.env` 并填写自己的密钥，然后在项目根目录运行 `docker compose up --build`。包内的空数据库文件用于 Docker 的文件挂载；容器首次启动会创建数据库表。

不要把填好密钥的 `.env` 再发给别人。原项目根目录的历史交付说明不在这份分享包中；以本文件和 `README.md` 的启动命令为准。

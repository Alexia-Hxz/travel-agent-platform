# 智能旅行规划平台

这是一个使用 Python、FastAPI、Vue 3、LangChain/LangGraph、Tavily 与百炼视觉模型构建的本地旅行 Agent 项目。它支持旅行项目管理、衣食住行地点收藏、每日行程、文件解析、联网规划，以及图片识别后的人工确认保存。

## 功能

- 创建、查看和删除旅行项目。
- 收藏景点、酒店、餐厅、机场和车站，并添加每日行程。
- 上传机票、酒店订单、菜单或攻略图片，使用 OCR 解析内容。
- 使用多模态 Agent 识别旅行图片，预览并编辑地点与行程建议后再保存。
- 使用 LangGraph 旅行规划 Agent 汇总已有行程与联网搜索结果，并提示预算和时间冲突。
- 通过自然语言对话调整行程（替换、删除、新增项目、改预算）。

## 前置条件

- Python 3.11 或更高版本。
- [uv](https://docs.astral.sh/uv/)。
- Node.js 20 或更高版本。
- Docker Desktop，用于容器化演示。

## 环境变量

在 `backend` 目录下创建 `.env`，复制 `backend/.env.example` 中的变量名并填入自己的 Key：

```dotenv
DASHSCOPE_API_KEY=
PEXELS_API_KEY=
```

`DASHSCOPE_API_KEY` 用于百炼联网检索、语言模型与视觉分析；`PEXELS_API_KEY` 用于给推荐卡片配图（免费图库，使用时按图库要求署名）。这两个是项目**全部**的外部服务，不要再加计量型搜索 API：原来的 `TAVILY_API_KEY` 已经因为对每个查询都返回 HTTP 432 而被整体删除。不要将 `.env` 提交到 Git。

## Docker 启动

在项目根目录运行：

```powershell
docker compose up --build
```

启动后访问：

- 前端：http://127.0.0.1:5173
- 后端健康检查：http://127.0.0.1:8001/api/v1/health
- 后端 API 文档：http://127.0.0.1:8001/docs

停止容器：

```powershell
docker compose down
```

Compose 会保留 `data/` 和 `backend/travel_agent.db`，所以重启容器不会清空本地旅行数据、上传文件或向量库。

## 本地开发启动

后端使用 OCR 专用虚拟环境：

```powershell
cd backend
$env:UV_PROJECT_ENVIRONMENT='.venv-ocr'
uv run uvicorn app.main:app --reload --port 8001
```

另开一个终端启动前端：

```powershell
cd frontend
npm install
npm run dev
```

前端开发地址默认为 http://127.0.0.1:5173。

## 测试与构建

```powershell
cd backend
$env:UV_PROJECT_ENVIRONMENT='.venv-ocr'
uv run pytest tests -v
```

```powershell
cd frontend
npm test
npm run build
```

## 项目文档

- [架构说明](docs/architecture.md)
- [演示脚本](docs/demo-script.md)

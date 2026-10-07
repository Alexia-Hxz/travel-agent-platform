# 系统架构

## 总览

系统采用 Vue 3 单页应用和 FastAPI REST API 的前后端分离架构。浏览器负责表单、时间线和 Agent 结果展示；FastAPI 负责数据校验、SQLite 持久化、文件解析、Agent 工具调用和外部模型访问。

```text
Vue 3 + Element Plus
        |
        | HTTP /api/v1
        v
FastAPI + SQLAlchemy + SQLite
   |        |          |
   |        |          +-- 旅行项目、地点、行程、文件
   |        +-- 本地文件与 OCR
   |
   +-- LangGraph 规划工作流
          |-- 百炼联网检索（文本资料）
          |-- Tavily 联网搜索（图片）
          |-- 预算与行程冲突工具
          +-- 百炼视觉分析
```

> 2026-09 变更：原先还有一层"城市知识库 + 本地 RAG 检索"（`app/rag/`、ChromaDB、
> `knowledge_documents`）。因为实测检索恒为 0 条、对规划零贡献，已整体删除。

## 数据与业务边界

- `Trip` 是旅行项目根实体，包含目的地、日期范围和预算。
- `Place` 与 `ItineraryItem` 分别存储收藏地点和每日行程。后端拒绝日期越界或跨旅行项目地点。
- `TripFile` 保存上传文件的元数据和解析结果；文件内容只保存在本地 `data/uploads`。

## Agent 边界

规划 Agent 使用 LangGraph 编排：读取旅行上下文、可选联网搜索、提取推荐地点并配图、计算预算与检测冲突，最后返回结构化结果。联网不可用时会降级，不阻塞已有旅行数据的规划。

多模态 Agent 仅在用户上传图片后调用百炼视觉模型。模型输出先转为可编辑预览；地点或行程只会在用户点击确认保存后写入数据库。行程日期越界时不会留下部分地点数据。

## 密钥与部署

`DASHSCOPE_API_KEY` 与 `PEXELS_API_KEY` 只从后端 `backend/.env` 读取。前端代码、浏览器网络请求、Docker 镜像和文档均不包含密钥。

Docker Compose 包含两个服务：后端运行在宿主机端口 8001；前端由 Nginx 提供静态文件并运行在端口 5173。Nginx 将 `/api/` 转发给后端服务。`data/` 与 SQLite 文件以挂载方式持久化。

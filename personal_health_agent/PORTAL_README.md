# PHA Web 门户开发指南

本文介绍 Web 门户的结构和配置。启动方法见[项目 README](README.md)。

## 架构

React 前端默认使用端口 3000，调用运行在端口 8000 的 FastAPI 服务。后端根据会话配置运行 PHA 编排架构、Parallel 并行基线或 PHIA 单 Agent ReAct 基线。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `api/main.py` | FastAPI 应用和会话管理 |
| `frontend/src/App.jsx` | React 主组件 |
| `frontend/src/components/SplashScreen.jsx` | 配置页面 |
| `frontend/src/components/ChatInterface.jsx` | 聊天界面 |
| `frontend/src/components/*.css` | 组件样式 |
| `pha/agents/` | Agent 实现 |
| `pha/llm/` | 大模型后端抽象层 |
| `pha/utils/` | 模型发现及其他工具 |
| `data/sample/` | 默认合成档案 |
| `data/*/` | 其他档案目录 |

## API 端点

### 获取配置

```text
GET /config
```

返回可用的模型提供商、模型、架构和用户档案。

### 管理会话

```text
POST /sessions
Body: {
  "provider": "gemini",
  "model_id": "models/gemini-2.0-flash",
  "baseline": "pha",
  "persona_id": "sample",
  "gemini_api_key": "...",     // 也可以使用 openai_api_key
  "tavily_api_key": "..."      // 可选
}
Response: { "session_id": "uuid", ... }

GET /sessions/{id}
DELETE /sessions/{id}

POST /sessions/{id}/chat
Body: { "message": "How is my sleep?" }
Response: { "response": "...", "processing_time": 1.23 }

GET /sessions/{id}/messages
Response: { "messages": [...] }
```

## 添加用户档案

1. 创建目录 `data/my_persona/`。
2. 添加 `summary.csv`（每日健康指标）、`activities.csv`（活动记录）、`profile.csv`（人口学信息）和 `population_percentiles.csv`（参考人群数据，可从样本复制）。
3. 如需添加档案介绍，可创建 `description.txt`。
4. 重启 API 服务。门户会自动发现新档案。

## 模型提供商

| 提供商 | 支持情况 | 说明 |
| --- | --- | --- |
| Gemini | 已支持 | 支持全部架构。 |
| OpenAI | 已支持 | 通过 OneTwo 支持全部架构。 |
| Anthropic | 即将支持 | 后端已就绪，界面入口暂未启用。 |

## 环境变量

API 密钥可以在门户界面填写，也可以通过服务端环境变量设置。

```bash
# 模型提供商
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# 可选服务：供领域专家 Agent 搜索网页
TAVILY_API_KEY=...
```

## 本地开发

### 前端

```bash
cd frontend
npm install
npm run dev          # 启动支持热重载的开发服务器
npm run build        # 构建生产版本
npm run preview      # 预览生产版本
```

### 后端

```bash
# 启动支持自动重载的服务
uvicorn api.main:app --reload --port 8000

# 运行测试
pytest tests/

# 运行类型检查
mypy pha/
```

### 添加新模型

提供密钥后，应用会调用对应提供商的 API 自动发现模型。备用模型列表定义在 `pha/utils/model_discovery.py` 中。

## 用户研究模式

当 `api/main.py` 中的 `GLOBAL_USER_STUDY_FLAG = True` 时，配置页会隐藏提供商、模型和密钥选项。服务端从 Git 忽略的 `config/user_study_defaults.json` 读取默认值。

1. 将 `config/user_study_defaults.example.json` 复制为 `config/user_study_defaults.json`。
2. 填写所用提供商的 `provider`、`model` 和 `api_keys`。
3. 密钥仅存于服务端。开启用户研究模式后，`GET /user-study-defaults` 只返回 `{ provider, model }`。

## 开展用户研究

门户支持不同 Agent 架构的 A/B 测试。

- 使用不同的 `baseline` 值创建会话，便于随机分组。
- 消息保存在各自的会话对象，可通过 `/sessions/{id}/messages` 读取。
- 每条回复带有 `processing_time`，便于比较响应耗时。
- 会话彼此独立，可在研究结束后删除。

## 常见问题

### 提示 API 密钥无效

确认密钥格式与所选提供商匹配。Gemini 密钥形如 `AIza...`，OpenAI 密钥形如 `sk-...`。可刷新页面后重新输入。

### PHIA 基线无法运行

PHIA 使用 OneTwo ReAct，支持 Gemini 和 OpenAI。使用 Anthropic 模型时，选择 PHA 或 Parallel。

### 前端连不上 API

确认 API 服务正在监听端口 8000。前后端域名不同时，检查 CORS 设置，并确认防火墙允许本地连接。

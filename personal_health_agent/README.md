# 个人健康 Agent（PHA）框架

本目录是 Google 为论文《The Anatomy of a Personal Health Agent》开源的个人健康 Agent（Personal Health Agent，PHA）框架的简体中文版。如果你在研究中使用这项工作，请[引用论文](#scroll-citation)。英文源码见 [Google 官方仓库](https://github.com/Google-Health/consumer-health-research/tree/main/personal_health_agent)。

[Google Research 博客](https://research.google/blog/the-anatomy-of-a-personal-health-agent/) · [论文预印本](https://www.arxiv.org/abs/2508.20148)

![个人健康 Agent（PHA）框架](docs/header.png)

## 概览

PHA 是一个多 Agent 框架，用于分析可穿戴传感器数据流、活动记录和人口学信息等多模态健康数据，在日常非临床场景给出个性化的分析和建议。

编排器协调三个专用子 Agent。

- **数据科学 Agent**（Data Science Agent）分析个人健康时间序列，识别趋势、比较不同时间段或人群，并按需计算统计指标。
- **领域专家 Agent**（Domain Expert Agent）结合医学参考范围和健康背景解读分析结果，使用网页搜索和 Data Commons 获取信息。
- **健康教练 Agent**（Health Coach Agent）综合上述信息，通过对话给出可执行的建议。

项目还提供两种用于比较的基线架构。**Parallel** 让同样的三个专用 Agent 并行工作。**PHIA** 使用一个通用的 ReAct Agent。论文在 10 项基准任务中评估这些架构，覆盖健康专家和终端用户提供的 7,000 多条标注。

## 环境搭建

需要 Python 3.11 或更新版本以及 conda。运行 `bash setup.sh` 可自动创建并配置 `pha_framework` 环境。

```bash
# 创建 conda 环境并安装所有依赖
bash setup.sh

# 激活环境
conda activate pha_framework
```

## Web 门户（推荐）

Web 门户可以配置模型和用户档案，并直接与健康 Agent 对话。

### 快速上手

```bash
# 终端 1：启动 API 服务
conda activate pha_framework
uvicorn api.main:app --reload --port 8000

# 终端 2：启动前端
cd frontend
npm install  # 首次启动时安装依赖
npm run dev
```

在浏览器打开 <http://localhost:3000>，按页面提示填写 API 密钥。

### 门户功能

- **模型提供商**支持 Gemini 和 OpenAI。
- **架构对比**支持在 PHA、Parallel 和 PHIA 之间切换。
- **用户档案**支持切换不同的健康数据样本。
- **设置保存**将 API 密钥和偏好设置保存在本地。

### 系统架构

| 架构 | 组成 | 运行方式 |
| --- | --- | --- |
| **PHA** | 3 个专用 Agent 和 1 个编排器 | 编排器按问题选择合适的 Agent，并协调生成回复。 |
| **Parallel** | 3 个专用 Agent | 所有 Agent 并行处理同一问题，再汇总结果。 |
| **PHIA** | 1 个通用 Agent | 原始单 Agent ReAct 架构，支持 Gemini 和 OpenAI。 |

## 数据格式

PHA 读取四个 CSV 文件。仓库在 `data/sample/` 提供了合成样本数据。字段定义请见[数据说明](data/README.md)。

| 文件 | 用途 |
| --- | --- |
| `summary.csv` | 每日健康指标，如步数、睡眠和心率。 |
| `activities.csv` | 每次运动或活动的记录。 |
| `profile.csv` | 用户档案，如年龄、性别、身高和体重。 |
| `population_percentiles.csv` | 与同类人群比较时使用的参考数据。 |

可以采用以下任一方式加载自己的数据。

1. 按相同字段结构替换 `data/sample/` 下的 CSV 文件。
2. 在 `data/` 下创建新的用户档案目录并放入文件。
3. 初始化时通过 `Settings(data_dir="path/to/your/data")` 指定位置。

## 在代码中使用

Python 脚本和 Jupyter Notebook 都可以直接调用各个 Agent。

```python
import pandas as pd
from pha.agents import (
    DataScienceAgent,
    DomainExpertAgent,
    HealthCoachAgent,
    MultiAgentOrchestrator,
)

# 加载健康数据
summary_df = pd.read_csv('data/sample/summary.csv')
activities_df = pd.read_csv('data/sample/activities.csv')
profile_df = pd.read_csv('data/sample/profile.csv')
population_df = pd.read_csv('data/sample/population_percentiles.csv')

# 数据科学 Agent：分析 DataFrame 并计算统计指标
ds_agent = DataScienceAgent()
ds_agent.configure(api_key='your-key', provider='gemini')  # 也可以用 'openai'
ds_agent.load_dataframes({
    'summary': summary_df,
    'activities': activities_df,
    'profile': profile_df,
    'population': population_df,
})

# 领域专家 Agent：利用网页搜索和参考范围解读分析结果
de_agent = DomainExpertAgent(tavily_api_key='your-tavily-key')
de_agent.get_agent(api_key='your-key', provider='gemini')
health_context = (
    f"## User Profile\n{profile_df.to_string()}\n\n"
    f"## Recent Health Summary\n{summary_df.head(30).to_string()}"
)
de_agent.set_user_health_data(health_context)

# 健康教练 Agent：汇总信息并以对话形式给出建议
coach = HealthCoachAgent(simple_mode=True)
coach.configure(api_key='your-key', provider='gemini')

# 编排器：把问题交给合适的专用 Agent
pha = MultiAgentOrchestrator()
pha.configure(api_key='your-key', provider='gemini')
pha.set_agents(
    data_science_agent=ds_agent,
    domain_expert_agent=de_agent,
    health_coach_agent=coach,
)

# 提出健康问题
response = pha.respond("我最近睡得怎么样？")
print(response)
```

### 教程 Notebook

[`notebooks/`](notebooks/) 目录包含各 Agent 的独立示例，以及完整的多 Agent 流程。

| Notebook | 内容 |
| --- | --- |
| [01 快速上手](notebooks/01_quickstart.ipynb) | 约 5 分钟体验基本用法。 |
| [02 数据科学 Agent](notebooks/02_data_science_agent.ipynb) | 分析数据并识别趋势。 |
| [03 领域专家 Agent](notebooks/03_domain_expert_agent.ipynb) | 使用 ReAct 解读健康指标。 |
| [04 健康教练 Agent](notebooks/04_health_coach_agent.ipynb) | 通过多轮对话提供健康指导。 |
| [05 完整流程](notebooks/05_full_pipeline.ipynb) | 从问题路由到回复的完整编排流程。 |
| [06 Parallel 基线](notebooks/06_parallel_baseline.ipynb) | 多 Agent 并行基线。 |
| [07 PHIA 基线](notebooks/07_phia_baseline.ipynb) | 单 Agent ReAct 基线。 |

[领域专家少样本导读](few_shots/EXAMPLES.zh-CN.md)和 [PHIA 少样本导读](few_shots/phia/EXAMPLES.zh-CN.md)提供 57 个原问题的中文翻译和答复摘要。完整的 ReAct 示例仍由模型从英文 Notebook 读取。

中文版门户和教程可直接使用中文提问，运行提示词会引导模型沿用用户的语言。API 路径、CSV 字段、工具名称和 ReAct 标签沿用原有格式。若要复现论文的英文提示词实验，请使用 [Google 原版](https://github.com/Google-Health/consumer-health-research/tree/3e9d1e4877a54cf81ce8ef34085006952dd19601/personal_health_agent)。

### 环境变量（可选）

也可以通过环境变量提供密钥。

```bash
export GEMINI_API_KEY='your-gemini-key'
export OPENAI_API_KEY='your-openai-key'
export TAVILY_API_KEY='your-tavily-key'  # 可选，用于网页搜索
```

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `pha/agents/` | 三个专用 Agent、编排器和两种基线实现 |
| `pha/prompts/` | 各 Agent 的提示词模板 |
| `pha/llm/` | 模型提供商抽象层 |
| `pha/tools/` | 网页搜索和 Python 沙箱 |
| `pha/utils/` | 数据加载、用户档案、解析和模型发现 |
| `pha/streaming.py` | 为 Web 门户发送服务端事件（SSE） |
| `api/` | FastAPI 后端 |
| `frontend/` | React 和 Vite 前端 |
| `notebooks/` | 教程 Notebook |
| `few_shots/` | 健康解读少样本示例 |
| `data/` | 合成档案及健康样本 |
| `config/` | 应用设置和用户研究配置 |
| `tests/` | 单元测试和端到端测试 |

Web 门户的实现细节见[开发指南](PORTAL_README.md)。

## API 参考

| 端点 | 方法 | 功能 |
| --- | --- | --- |
| `/config` | GET | 获取所有配置选项 |
| `/sessions` | POST | 创建对话会话 |
| `/sessions/{id}` | GET | 获取会话状态 |
| `/sessions/{id}` | DELETE | 结束会话 |
| `/sessions/{id}/chat` | POST | 发送消息 |
| `/sessions/{id}/messages` | GET | 获取消息历史 |

后端运行后，可以在 <http://localhost:8000/docs> 查看完整 API 文档。

<a id="scroll-citation"></a>

## 引用方式

如果你在研究中使用了[论文](https://www.arxiv.org/abs/2508.20148)或这份代码，请引用以下条目。

```bibtex
@article{heydari2025anatomy,
  title={The anatomy of a personal health agent},
  author={Heydari, A Ali and Gu, Ken and Srinivas, Vidya and Yu, Hong and Zhang, Zhihan and Zhang, Yuwei and Paruchuri, Akshay and He, Qian and Palangi, Hamid and Hammerquist, Nova and others},
  journal={arXiv preprint arXiv:2508.20148},
  year={2025}
}
```

## 参与贡献

详见[贡献指南](CONTRIBUTING.md)。

## 许可证

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## 免责声明

本项目不是 Google 官方提供支持的产品，也不属于 [Google 开源软件漏洞奖励计划](https://bughunters.google.com/open-source-security)的覆盖范围。代码仅用于演示，不适合部署在生产环境。

此研究代码不属于医疗器械，也不适用于任何临床用途，包括诊断、预后判断和治疗建议。

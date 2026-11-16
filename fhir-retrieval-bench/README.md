# FHIR 检索基准（FHIR Retrieval Bench）

FHIR Retrieval Bench 是面向临床场景的评估框架。它以按患者组织成 FHIR Bundle 的纵向电子健康记录（EHR）为输入，评估问答系统和检索增强生成（RAG）策略。

本仓库提供论文对应的官方实现。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `LICENSE` | 项目的许可证 |
| `README.md` | 安装和使用说明 |
| `pyproject.toml` | PEP 621 包元数据和依赖 |
| `config/` | 可组合的 YAML 配置，包括模型、数据集和策略 |
| [`docs/`](docs/) | 详细文档 |
| [`notebooks/`](notebooks/) | 快速上手、数据预处理和结果分析示例 |
| [`src/fhir_retrieval_bench/`](src/fhir_retrieval_bench/) | 基准测试主体 |
| [`src/ns_agent_adk/`](src/ns_agent_adk/) | 神经符号临床推理 Agent |
| `tests/` | Pytest 测试 |
| `scripts/` | 命令行执行脚本 |
| `data/` | 本地数据集，由 Git 忽略 |

## 安装和配置

项目按标准 Python 包组织，支持 Python 3.11 及更高版本，并针对 Python 3.13 做了适配。

### 1. 创建并激活虚拟环境

在项目目录下创建虚拟环境，以隔离依赖。

```bash
python3 -m venv fhir-hopper-env
source fhir-hopper-env/bin/activate
```

### 2. 以可编辑模式安装

以下命令会安装项目及全部依赖，包括 `numpy`、`pandas`、`pyarrow`、`litellm`、`google-genai` 和 `google-adk`。

```bash
pip install -e .
```

## 快速上手

### 1. 检查数据集配置

将 Parquet 或 pickle 数据文件放到 `data/`，然后检查数据能否正确解析、加载。

```bash
python scripts/check_dataset.py
```

### 2. 测试模型 API 连接

支持 Gemini（通过 Google GenAI）、OpenAI 和 Anthropic 等大模型后端。设置对应的 API 密钥后，运行以下命令进行连通性测试。

```bash
# 测试 Gemini 模型 API 连通性
export GENAI_API_KEYS="your_gemini_api_key"
python scripts/check_model_api.py --model gemini_3_flash_preview
```

### 3. 运行多组合评估

命令行工具可对模型、检索策略和数据集做笛卡尔积，批量运行各组评估。

```bash
export OUTPUT_DIR=./outputs/current_run

python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --num_workers 4
```

## 文档和进阶用法

- [完整操作指南](docs/index.md)
- [数据集和目录结构](docs/docs/datasets.md)
- [可用的大模型和嵌入模型](docs/docs/models.md)
- [检索和推理策略，包括神经符号 Agent](docs/docs/strategies.md)

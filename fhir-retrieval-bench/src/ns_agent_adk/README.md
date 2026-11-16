# 与临床相协调:预算受限制的制剂实施

`ns_agent_adk` 包提供了一种用于纵向电子健康记录的推理的神经系统LLM剂，其结构为FHIR捆绑.该剂通过语义搜索，提取上下文，采取预算推理步骤，并使用图传工具来制作全面的临床摘要。

这个执行是使用Google代理开发工具包(ADK)和`google-genai`库构建的。

## 关键特性

- **编年史超图**:变形标准 FHIR JSON 捆绑成互联`ChronologicalHypergraph`结构，无缝地解决引用，时间事件，以及断然的数据。
- ** Causal Salienity Linearization**:嵌入式和时间逻辑用于构造FHIR数据(一个"skeleton view")的受限子集，严格保证与模型上下文窗口相匹配。
- **海关工具执行**:代理机配备了一套自定义图轨工具逻辑，以渐进探索患者图的"隐藏"区域：
    - ** `inspect_node`**:读取具体FHIR资源的详细临床描述。
    - ** `search_graph`**:使用矢量嵌入物进行语义搜索，以发现特定的条件或症状。
    - ** `follow_links`**:Traversestructure FHIR 引用链接。
    - ** `filter_graph_by_time`**:基于具体日期限制的查询子图。
    
## 项目结构

```text
ns_agent_adk/
├── README.md
├── __init__.py
├── engine.py       # Defines the NeuroSymbolicAgent, ADK Agent wrapper, and budgeting
├── tools.py        # Exposes ClinicalGraphTools for LLM graph reasoning and traversal
├── test_run.py     # Entrypoint script demonstrating setup, data ingestion, and querying
├── core/           # Fundamental data structures and timeline logic 
│   ├── __init__.py
│   ├── fhir_time.py
│   ├── graph.py    # Defines ChronologicalHypergraph for linking FHIR dependencies
│   └── linearizer.py # Implements CausalSaliencyLinearizer for LLM context generation
├── utils/          # General utility scripts
│   ├── __init__.py
│   ├── embeddings.py
│   ├── fhir_utils.py
│   └── temporal_parser.py
├── config/         # Configuration setup for Gemini models and embedding API
│   └── config.py
├── templates/      # Jinja system preamble templates
│   └── system_preamble.jinja2
└── tests/          # Unit tests for the agent and time-handling functionality
    ├── test_engine.py
    └── test_time_handling.py
```

## 开始

### 先决条件

要测试该代理机使用 Gemini 模型，请确保您有适当的环境变量集：

```bash
export GENAI_API_KEYS="your_api_key_here"
```

您可以通过修改 `config/config.py` 更改后端 LLM 和嵌入模型。默认情况下，`gemini-3-flash-preview` 用于 LLM 和 `gemini-embedding-001` 用于嵌入。

### 运行示例

执行python脚本以即时化图，计算嵌入，初始化`NeuroSymbolicAgent`,并运行一般诊断查询。

```bash
PYTHONPATH=src/ python src/ns_agent_adk/test_run.py
```

默认情况下，剧本要求代理商_"在过去5年中放大病人的病史记录"_ 剧本将同时丢弃LLM最终生成的输出和完整的ADK推理步骤跟踪，包括图中转录过程中所做的工具调用。

### 评估 基准测试

```bash
python scripts/run_evals.py \
  --dataset medagentbench \
  --strategy ns_agent \
  --model gemini_3_flash_preview \
  --shuffle --max_rows 10
```



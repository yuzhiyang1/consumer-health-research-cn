# FHIR 检索 法官席

[链接到代码](../src/fhir_retrieval_bench/)

## 概览

FHIR 检索 座席评价问答系统比病人FHIR.
(快速保健互操作性资源)数据。
**模型**、一个**数据集**和一个**战略**碎片。

每个片段的配置定义为YAML文件，在`config/`下，由CLI旗选中：

```text
config/
  model/      # answer, judge, and embedding model names
  dataset/    # which benchmark to load and where it lives
  strategy/   # how FHIR data is turned into context
```

## 如何运行 基准测试

### 1. 选择一个模型并设置证书

模型碎片在`config/model/`下存在。
** 回答**、** 法官** 和** 将** 模型纳入运行；
判断模型是独立的，可以来自不同的提供者。

最简单的路径是默认 Google GenAI 后端。导出一个或多个 API
键作为逗号分隔的列表。如果指定多个 API 键，则框架会跨键旋转，以更好地处理配额问题：

```bash
export GENAI_API_KEYS="genai_key1,genai_key2,genai_key3"
```

对于 OpenAI - 备份的模型碎片，导出 OpenAI 密钥的方式相同：

```bash
export OPENAI_API_KEYS="openai-key-1,openai-key-2"
```

要运行模型与 Vertex AI 相对应，请确保您的机器被 gloud 授权到您的 GCP项目，并导出 `GCP_PROJECT_LOCATIONS` 环境变量：

```bash
export GCP_PROJECT_LOCATIONS="project_id_1:location_1,project_id_2:location_2"
```

`config/model/`下的每个模型YAML片段，需要为每个模型(答案，判断，嵌入)有一个明确的`backend: "vertex"`或`backend: "public"`参数来控制其主机路线。

见
[docs/models.md](docs/models.md),为运出的模型碎片的完整列表，提供商-路线规则,Vertex
模范花园设置，和`thinking_config`语义跨供应商。

#### 核查证书的设置

```bash
# 测试 Vertex API 用于 Gemini Pro 3.1 测试 高清 高清 高清 高清
export GCP_PROJECT_LOCATIONS="project_id_1:location_1"
python scripts/check_model_api.py --model gemini_3_flash_preview

# 测试GenAI API,用于Gemini Pro 3.1.
export GENAI_API_KEYS="genai_api_key"
python scripts/check_model_api.py --model gemini_3_flash_preview

# 测试 OpenAI API为 GPT-5 - mini.
export GENAI_API_KEYS="genai_api_key" # You still need this because `config/model/gpt_5_mini` is set to use Gemini embeddings
export OPENAI_API_KEYS="oai_api_key"
python scripts/check_model_api.py --model gpt_5_mini

# 测试 AnthropicVertex API为松网4.6.
export GENAI_API_KEYS="genai_api_key" # You still need this because `config/model/claude_sonnet_4_6` is set to use Gemini embeddings
export GCP_PROJECT_LOCATIONS="project_id_1:location_1"
python scripts/check_model_api.py --model claude_sonnet_4_6
```


### 2. 选择一个数据集

数据集碎片在 `config/dataset/` 下存在。
数据集 root。默认情况下，它被设定为带有预处理数据集的本地 `data/`目录。

已装运的数据集：

| 数据集          | 说明              |
| ---------------- | ------------------------ |
| `ehrqa`          | EHRQA 基准测试          |
| `fhiragentbench` | FHIRAgentBench 基准测试 |
| `medagentbench`  | MedAgentBench 基准测试  |
| `fhirpathqa`     | FHIRPathQA 基准测试     |

确切的文件预期值见
[docs/datasets.md](docs/datasets.md) .

#### 校验数据集设置

```bash
python scripts/check_dataset.py --dataset fhiragentbench --limit 3 # check only first records for FHIRAgentBench dataset
python scripts/check_dataset.py # check every fragment
```

此验证已配置路径存在、 记录负载和样本
输出看起来很合理。

### 3. 选择策略

战略碎片在`config/strategy/`下存在。
FHIR 捆绑变为LLM上下文。

| 战略        | 说明                                               |
| --------------- | --------------------------------------------------------- |
| `ontology_guided_retrieval` | 基于关键词的评分与医学肿瘤学术语扩展 |
| `fhir2text`     | 将 FHIR 资源转换为自然语言文本块 |
| `prefiltered`   | 基于状态的过滤和按概念代码的复制  |
| `embedding_csl` | 基于嵌入的序列化                 |
| `ns_agent` | 基于图表的FHIR + 工具使用 |
| `flowsheet` | 表格视图 FHIR 使用数据 FHIRPath |
| `flowsheet_agent` | 表格FHIR数据查询代理代表 |

详情(包括如何预先计算嵌入
`embedding_csl`,参见[docs/strategies.md](docs/strategies.md).

### 4. 运行评价

主要的CLI入口是`scripts/run_evals.py`.
指定策略、数据集和模型片段组成一个完整的配置，
然后运行评价，并将结果写入输出目录。

#### 微量运行(建议先行)

从小跑开始，你就可以确认一切工作到最后
将时间或配额花在满额基准测试:

```bash
export OUTPUT_DIR=./outputs/current_run
python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --shuffle --max_rows 3

python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet,flowsheet_agent,embedding_csl,ns_agent \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --shuffle --max_rows 3
```

#### 全程运行

```bash
python scripts/run_evals.py \
  --dataset medagentbench \
  --strategy ontology_guided_retrieval \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR
```

#### 网格运行

任何一项 `--strategy` , `--dataset` ,以及 `--model` 可以采取逗号分隔
CLI运行着笛卡尔产品：

```bash
python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --num_workers 4
```

<!-- * 
* MedAgentBench : 148 ()
* FHRAgentBench : 2,931
* EHRQA : 5,133
* FHIRPathQA : 2,025

( 148 + 2931 + 5133 + 2025 ) x ( 1 型号 ) x ( 3 战略) x 2 秒 -> 12 小时
NS 代理+嵌入=5x
-->

> 提示 :此基准测试 主要是被 API 的位数和配额，而不是本地计算。您通常可以在 gcloud 实例中运行它，而不是提交 XManager 任务来杠杆计算。
> 要同时执行基准测试,您只需使用`--num_workers`旗向每个模型提交一个单一命令，以最大限度地利用您的配额。最佳工人人数取决于您特定的API配额和战略，所以您可能只需要给它几条尝试并监控终端上打印的配额错误信息或者检查API的使用仪表板。

> 提示 :如 `ns_agent` 等策略，引用多个 LLM 的调用;对于这些策略，请使用一个较低的工人计数来避免费率限制。我们建议运行 `ns_agent`作为单独的指令，比其他策略低 `--num_workers`。

> 提示 :您可以设置 `cache_dir` 来保存嵌入缓存。如果您使用相同的嵌入模型进行多个实验，这样做特别有用。您还可以运行 `precompute_embeddings.py` 脚本一次，以预算数据集的嵌入。

#### 所有CLI选项如下：

| 旗帜            | 默认                             | 说明                                                 |
| --------------- | ----------------------------------- | ----------------------------------------------------------- |
| `--strategy`    | 所需经费                            | 战略片段名称，逗号分隔                  |
| `--dataset`     | 所需经费                            | 数据集碎片名称，逗号分隔                   |
| `--model`       | 所需经费                            | 模型碎片名称，逗号分隔                     |
| `--output_dir`  | `/tmp/fhir_retrieval_bench_results` | 成果和检查站目录                 |
| `--cache_dir`  | `/tmp/fhir_retrieval_bench_cache` | 保存缓存的目录(如嵌入)                 |
| `--num_workers` | `1`                                | 每项评价的平行线索                             |
| `--shuffle`     | `false`                             | 取消 eval 实例                                 |
| `--max_rows`    | 所有行                            | 最大记录已装入。当设定 `--shuffle` 时，洗牌先完成 |
| `--dry_run`     | `false`                             | 不执行而打印编组的配置                    |

#### 预计算嵌入

对于需要嵌入计算的战略(例如，`embedding_csl`,`ns_agent`),在评价过程中依次计算它们会成为一个瓶颈。您可以预先为使用`precompute_embeddings.py`脚本的数据集进行嵌入计算：

```bash
python scripts/precompute_embeddings.py \
    --strategy=embedding_csl,ns_agent \
    --dataset=medagentbench,fhiragentbench,ehrqa \
    --embedding_model=gemini-embedding-001 \
    --cache_dir=/tmp/fhir_retrieval_bench_cache
```

* `--strategy`:用于预计算嵌入的策略配置片段名称(默认：`embedding_csl,ns_agent`).
* `--dataset`:用于计算嵌入的数据集名称，用于(例如`ehrqa`,`fhiragentbench`.
* `--embedding_model`:嵌入型号名称。
* `--cache_dir`:用于保存缓存嵌入的目标目录(默认：`/tmp/fhir_retrieval_bench_cache`.

#### 输出目录结构

默认情况下，结果会被写入 `/tmp/fhir_retrieval_bench_results/`
(由 `--output-dir` 覆盖。每个运行会写三个文件，全部由 $ 键
相同：(中文(简体) ).

```text
<output-dir>/<dataset>_<strategy>_<answer-model>_<sha6>/
  results.json          # per-record results
  summary.json  # aggregate metrics
  .checkpoint.jsonl  # per-record checkpoint
```

墙上的时间戳住在摘要内JSON,而不是文件名，所以
重新运行相同的配置，覆盖先前已存在的输出，而不是
累积日期副本。

**为什么`<sha6>` 文摘？**文件名已经载有数据集，
战略，以及答案模式——但不是法官模式，嵌入式
模型，每台调用 LLM knobs(温度，`max_output_tokens`),
`thinking_config`),`dataset.path`,或`use_vertex_ai`. 6 -char SHA是一位美国著名女演员。
没有指纹，两条只不同
判断模型或温度会默默地覆盖同一文件。
拼写所有字面上的东西都会使文件名失去功能;摘要是
契约妥协


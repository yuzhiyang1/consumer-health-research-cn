# 战略

`config/strategy/<name>.yaml` 下的每个战略片段都选择一个
FHIR 捆绑变成了LLM上下文。
只说出战略名称，因为目前的战略不会暴露金枪鱼的可捕性
knowbs通过配置。要改变行为，请编辑执行
`strategies/<name>.py` .

## 运输战略

| 战略 | 说明 | 执行细节 |
|---|---|---|
| `ontology_guided_retrieval` | 基于关键词的评分与医学肿瘤学术语扩展 | 从零开始执行 |
| `fhir2text` | 将 FHIR 资源转换为自然语言文本块 | FHIR2Text 系列化 基线 (英语). |
| `prefiltered` | 基于状态的过滤和按概念代码的复制(https://arxiv.org/pdf/2402.01711)) | 从零开始执行 |
| `embedding_csl` | 基于嵌入的序列化 | 从零开始执行 |
| `ns_agent` | 基于图表的FHIR + 工具使用 | `src/ns_agent_adk/` |
| `flowsheet` | 表格视图 FHIR 使用数据 FHIRPath | https://www.nature.com/articles/s41746-025-01708-w |
| `flowsheet_agent` | 表格 FHIR RAG 具有动态概念的代理 检索. | ADK 使用流程表元数据及表格查询工具的代理，并带有日期过滤和适应性下采样 |

### `embedding_csl` 说明

`embedding_csl`战略需要嵌入以确定相关资源。
重算嵌入到整个运行，你可以做一次预计算
预览 `scripts/precompute_embeddings.py` .

## 添加新战略

1.  执行文件
    `src/fhir_retrieval_bench/strategies/<name>.py` 子分类 `Strategy` (英语).
    或 减 减 `RAGStrategy` 从 `strategies/base.py` .
2. 匹配碎片 YAML在 `config/strategy/<name>.yaml` 下。
3. 登记在`src/fhir_retrieval_bench/strategies/__init__.py`发送中。
# 数据集配置

## 概览

每个数据集的配置文件位于 `config/dataset/`，指定本地问答数据和 FHIR 数据的路径。

```yaml
dataset:
  name: fhiragentbench
  qa_path: /path/to/fhiragentbench/qa_path
  fhir_path: /path/to/fhiragentbench/fhir_path
```

编辑后可运行 `scripts/check_dataset.py`，检查能否正确加载。

## 预处理代码

[预处理 Notebook](../../notebooks/preprocess_data.ipynb)可以验证 FHIR JSON Schema，并检查每条问答是否都有对应的 FHIR Bundle。新增数据集时建议运行它。

各数据集的问答条数如下。

| 数据集 | 问答条数 |
| --- | ---: |
| EHRQA | 5,133 |
| FHRAgentBench | 2,931 |
| MedAgentBench | 148 |
| FHIRPath-QA | 2,095 |

预处理后的 FHIR Parquet 文件包含 `Patient ID` 和 `FHIR Bundle` 两列。新增数据集时，可用 `src/fhir_retrieval_bench/data/fhir_utils.py` 中的 `verify_with_pydantic` 验证 Bundle 的格式。

## `fhiragentbench`

[论文](https://arxiv.org/abs/2509.19319)

问答 CSV 需包含 `patient_fhir_id`、`question` 和 `true_answer`。

## `ehrqa`

加载器使用 `patient_id`、`question`、`correct_answer` 和 `answer_choices`。

## `medagentbench`

[论文](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144)

问答 DataFrame 需包含 `eval_MRN`、`instruction` 和 `sol`。

## `fhirpathqa`

[论文](https://arxiv.org/abs/2602.23479)

`scripts/fhirpathqa/convert_to_pickle` 将 FHIRPath-QA 基准的 2,095 条记录转换为 pickle。项目将 MIMIC-IV on FHIR Demo 装入 HAPI，按患者进行 `$everything` 导出，再由 `scripts/fhirpathqa/build_bundles_parquet` 将 Bundle 写入 Parquet。部署方法见[上游仓库](https://github.com/mooshifrew/fhirpath-qa)的 `setup/`。

### 问答字段

| 字段 | 说明 |
| --- | --- |
| `patient_id` | MIMIC 数字标识符，用于查询 Parquet 中对应的 Bundle。 |
| `question` | 问题。 |
| `answer` | 序列化为 JSON 的列表，加载器将其转为字符串。 |
| `now` | 当前日期时间的基准值，会注入 `question_context`。 |

`source_meta` 原样保存以下可选字段。\
`query`、`perspective`、`split`、`question_template_id`、`question_template`、`patient_id_hash`、`s_placeholders`、`op_placeholders`、`t_placeholders`、`id`、`index`、`holdout`。

### 问题措辞

`clinical` 使用医学术语，共 1,239 条。 `patient` 使用日常语言，共 856 条。加载器会读入两种措辞，可按 `source_meta["perspective"]` 分开分析。

### 数据划分

上游对不同改写版本分层，将数据划分为 `train`（1,629 条）、`val`（193 条）和 `test`（273 条），用于监督微调实验。本基准加载全部划分。如果要与仅使用测试集的上游论文对比，可用 `source_meta["split"] == "test"` 进行筛选。

### 患者过滤

MIMIC-IV on FHIR Demo v2.1.0 的导出覆盖了问答中 100 名患者中的 96 名。缺少 Bundle 的四名患者（`10002430`、`10004113`、`10010058` 和 `10012438`）对应 70 条记录。预处理命令 `scripts/fhirpathqa/convert_to_pickle --fhir_path=...` 会去除这些记录，确保生成的 pickle 仅含可在 Parquet 中找到对应 Bundle 的问答。

## 相关文件

- 数据集配置 `config/dataset/*.yaml`
- 加载器 `src/fhir_retrieval_bench/data/loader.py`
- 数据集检查脚本 `scripts/check_dataset.py`

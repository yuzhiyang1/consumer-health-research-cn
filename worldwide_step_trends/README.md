# 利用消费级可穿戴设备队列估计年轻和中年人的全国步数

本项目评估大规模消费级可穿戴设备队列所推算的全国步数，检验它们与日本国民健康与营养调查（NHNS-J）和英格兰老龄化纵向研究（ELSA）基准的接近程度。研究还覆盖 28 个国家，与世界卫生组织关于身体活动不足的估计进行比较。

## 作者联系人( C)

* 主要作者:布拉姆·肖恩费尔特
* 联络点/代码审查:丹尼尔·罗根(`roggen@google.com`)

## 目录结构

```text
worldwide_step_trends/
├── LICENSE.txt                                # Apache License 2.0
├── README.md                                  # Project documentation and external setup guide
├── stratify.py                                # Core demographic stratification and imputation library
│
└── colab/
    └── analysis/                              # Interactive analysis notebooks
        ├── 0_generate_dummy_data.ipynb        # Generates synthetic datasets in ./data/ for external testing
        ├── 1_demographic_pyramids.ipynb       # Demographic pyramids and wearable sampling bias
        ├── 2_japan_england_eval.ipynb         # Validation against Japan (NHNS-J) and England (ELSA)
        ├── 3_global_eval.ipynb                # 28-country correlation with WHO inactivity prevalence
        ├── 4_filling_gaps_japan.ipynb         # Survey gap-filling and longitudinal imputation in Japan
        └── 5_uk_trends.ipynb                  # Longitudinal UK step trends by age group and sex (2015-2024)
```

## 外部设置和复制结果

### 1. Python 环境要求

安装Python3.10+和所需的科学计算包：

```bash
pip install numpy pandas scipy statsmodels patsy matplotlib seaborn openpyxl pycountry pytz absl-py jupyter
```

### 2. 数据目录布局 (`./data/`).

默认情况下，`stratify.py`在 `./data/` 下读写数据。set
`STEP_TRENDS_DATA_DIR`使用不同的地点：

```text
worldwide_step_trends/
└── data/
    ├── internal/
    │   ├── stratify_steps/
    │   │   ├── checkpoint_grid_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_completed_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_processed_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_grid_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   │   ├── checkpoint_completed_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   │   └── checkpoint_processed_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   └── 2021_2023_uk_50plus_participant_steps.pkl
    │
    └── public/
        ├── WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx
        ├── WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx
        ├── who_trends.csv
        └── 2015_2024_japan_steps.pkl
```

两个预先计算的`StratifySteps`检查站MD5
以下汇总参数集：

*   `a36731bea99396ebe673f464d73ada56` (**全球28 - 国家电网**):
    `start_date='2015-01-01'` , `end_date='2026-01-01'` , `n_shards=100` ,
    `min_days_per_user=1`(整个学习窗口的一面),`min_age=20`,
    `max_age=104` , `exclude_outliers=False` , `exclude_extreme_values=False` ,
    `min_steps=None` , `max_steps=172800` , `exclude_days_of_week=None` ,
    `target_country=None`(所有国家),`start_month=None`,,
    `stop_month=None` .
*   `9f4397029d8039e43903bb5b8b7e75ec` (**日本11月-卫星网格**):
    `start_date='2015-01-01'` , `end_date='2026-01-01'` , `n_shards=100` ,
    `min_days_per_user=1`(整个学习窗口的一面),`min_age=20`,
    `max_age=104` , `exclude_outliers=False` , `exclude_extreme_values=False` ,
    `min_steps=100` , `max_steps=50000` , `exclude_days_of_week=[6]`
    (星期一至星期六，星期日除外),`target_country='Japan'`,,
    `start_month=11`, `stop_month=11` (仅11月).

*安全说明:因为Python `pickle` (`.pkl`)文件可以任意执行。
代码在去序列化时，仅装入 `.pkl` 检查点文件，您需要
本地生成或从可信来源获取。*

### 3. 与合成数据运行( `0_generate_dummy_data.ipynb`)

由于参与者级别和原始组群步骤表不能重新分配
我们公开提供`colab/analysis/0_generate_dummy_data.ipynb`。
`0_generate_dummy_data.ipynb`中的单元格填充`./data/internal/`和
`./data/public/` 配有与精确计划匹配的合成数据集，MultiIndex
`1`通过`5`。

在当地运行完整 流程 :

1.  发射 Jupyter从 仓库 根：

    ```bash
    jupyter notebook
    ```
2.  打开并运行 `colab/analysis/0_generate_dummy_data.ipynb` 生成
    `./data/` .
3.  打开和运行笔记本`1_demographic_pyramids.ipynb`通过
    `5_uk_trends.ipynb` .

## 公共参考数据来源

将`./data/public/`中的合成公开文件替换为官方文件
研究中使用的第三方参考表：

### UN 世界人口前景 WPP 2024 )

5 -来自UN的年龄组人口Excel文件 世界人口前景2024:

*   `data/public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `data/public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`

下载自：
https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Population

*   男性： 男性： 男性： 男性： 男性： 男性： 男性：
    https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx
*   女性： 男性： 女性： 女性： 女性： 女性： 女性： 女性： 女性： 女性： 女性： 女性：
    https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx

使用条件：
Copyright 2024 by United Nations, made available under a Creative Commons
许可证CC BY 3.0 IGO: http://creativecommons.org/licenses/by/3.0/igo/.

### 日本全国健康和营养调查 NHNS-J )

*   `data/public/2015_2024_japan_steps.pkl`(由年度汇编)
    `{year}JapanStepsEng.xlsx`报告2015–2019,2022,2023,2024.

下载自：
https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00450171&tstat=000001041744&cycle=7&cycle_facet=cycle&metadata=1&data=1

*(注:选择英文版e-Stat页面，不会造成数据。)
显示； 保留在日文版本上，并在浏览器中翻译页面。
每年选择表格“步数的精确度和标准偏差——
按年龄组、人数、平均、标准偏差 -- -- 总数、男女
20 年长和年长。*

使用条件：“利用该网站的内容:在互联网上提供的信息
网站(以下简称“Content”)可自由使用、复制、复制、
公开转发、翻译或作其他修改，条件是
用户遵守下文1至6的规定。
不过，请注意，简单表格、图表中的数字数据和数据，
因此，使用条件并不包括版权。
适用于此类数据，所述数据可自由使用。”

### 自由生活跟踪器校准(Nakagata等人，2022)

自由生活步数校准方程( `Y_adj = (F - 4152.0) / 0.8` )
适用于日本的NHNS-J评价(`2_japan_england_eval.ipynb`和
`4_filling_gaps_japan.ipynb`),取自：

中加塔,T.,村上,H.,川上,R.,Tripette,J.,Nakae,S.,山田,Y.
(原始内容存档于2017-09-29). Ishikawa-Takata, K., Tanaka, S.,& Miyachi, M.(2022). 13的分步计算结果。
不同活动跟踪器:实验室和自由生活的结果
实验。98 , 24–33 .
https://doi.org/10.1016/j.gaitpost.2022.08.004

### 英文老龄问题纵向研究(英文) ELSA )

研究报告预印本中的统计数据摘要：
https://www.medrxiv.org/content/10.64898/2026.03.25.26349270v1

使用条件：CC-BY 4.0国际许可证。

### Strain等人，2024——全球估计不足身体活动

*   `data/public/who_trends.csv`

不足的普遍程度身体活动,取自补充表格
用于：
https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00150-5/fulltext

使用条件：“创用公用:这是根据下列条款分发的公开查阅文章：
(c) 允许不受限制的使用CC-BY许可证，
以任何媒介进行分发和复制，但原始作品必须是：
。您不需要获得权限以重新使用此选项
条款。”

## Notebook 数据依赖

### `0_generate_dummy_data.ipynb` (合成数据集生成器).

`./data/`中生产：

*   `internal/stratify_steps/checkpoint_*_a36731bea99396ebe673f464d73ada56.pkl`
    (全球28-国家电网).
*   `internal/stratify_steps/checkpoint_*_9f4397029d8039e43903bb5b8b7e75ec.pkl`
    (日本11月电网)
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`
*   `public/2015_2024_japan_steps.pkl`

### `1_demographic_pyramids.ipynb` (人口金字塔和抽样比亚斯).

要求：

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`

### `2_japan_england_eval.ipynb` (日本和英国地面真理验证)

要求：

*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/2015_2024_japan_steps.pkl`

### `3_global_eval.ipynb` ( 28 - 国家不活动关联)

要求：

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`
*   `public/2015_2024_japan_steps.pkl`

### `4_filling_gaps_japan.ipynb` (调查漏洞-填充与截图).

要求：

*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/2015_2024_japan_steps.pkl`

### `5_uk_trends.ipynb` ( UK 按年龄和性别分列的纵向步骤趋势)

要求：

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`

## 许可证

Copyright 2026 Google LLC

许可证全文载于[`LICENSE.txt`](LICENSE.txt)。

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

这不是官方支持的 Google 产品。此项目不是
有资格享受《公约》
[Google 开源软件脆弱性奖励方案](https://bughunters.google.com/open-source-security) .
该项目仅用于示范目的，并非意在
用于生产环境。

NOTE:本研究代码的内容仓库 (一)本意不在于成为一种。
医疗器械;和 (二)非用于任何类型的临床用途，
包括但不限于诊断或预后。

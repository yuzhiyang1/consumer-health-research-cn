# 全球心脏生理模型

本目录提供论文《Global Geographic, Seasonal, Environmental and Demographic Drivers of Variations in Longitudinal Cardiac Physiology in 5 million Fitbit Users》对应的代码。研究考察了 500 万名 Fitbit 用户的纵向心脏生理指标与地理、季节、环境及人口学因素之间的关系。

仓库包含完整统计流程，包括队列统计、按贝叶斯信息准则（BIC）选择模型的六阶段线性混合效应（LME）比较，以及生成论文表格和图形的代码。模型分析静息心率（RHR）和心率变异性（HRV）。

原始 Fitbit 队列数据未随本仓库发布。编号为 00 的 Notebook 会按公开的输入格式生成一份小型合成数据，供端到端运行流程。基于这份数据的估计值仅用于演示。

## 作者和联系方式

- 主要作者 Katarina Vukosavljević
- 联系人和代码审阅 Daniel Roggen（`roggen@google.com`）

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/file_io.py` | 文件系统适配和输入输出路径 |
| `src/cohort_io.py` | 按公开 CSV 字段结构加载队列 |
| `src/generate_dummy_data.py` | 生成合成 `dummy_data*.csv` |
| `src/seasonality_lib.py` | 下采样、特征工程、模型统计与绘图 |
| `src/lme_binned.py` | 用分箱似然加速随机截距 LME 的拟合 |
| `src/stage_competition.py` | 六阶段逐步比较和 `STAGE_OVERRIDES` |
| `src/paper_tables.py` | 生成模型比较、系数和人口统计的 LaTeX 表格 |
| `src/publication_figures.py` | 生成论文图形 |
| `notebooks/` | 四个可运行的分析 Notebook |
| [`data/README.md`](data/README.md) | 输入字段结构和可选参考表 |

## 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据

`file_io.get_data_dir()` 默认返回 `./data`。可用环境变量 `WPM_DATA_DIR` 指定目录。

| 路径 | 用途 |
| --- | --- |
| `dummy_data.csv`、`dummy_data_hrv.csv` | 队列输入 |
| `extra_data/` | 地图使用的参考表 |
| `results/<run_tag>/` | Notebook 02 的单次运行产物 |
| `results/paper/` | Notebook 01、03 在启用 `SAVE_PAPER_ASSETS` 时输出的图表 |

输入按用户和自然月组织，每行包含 `id`、`month`、`age`、`sex`、`country`、`latitude`、`longitude`、`temperature`、`temperature min`、`temperature max`、`day light`，以及 `resting heart rate` 或 `heart rate variability`。 `region` 为可选列。使用自己的数据时，可按该格式写入并通过 `WPM_DATA_DIR` 指定路径，也可放入 `data/`。 `cohort_io.load_cohort` 会将表头和单位映射为程序要求的格式。

季节性观测地图和全球残差等值线图需要 `extra_data/` 下的 Natural Earth admin-0 GeoJSON 和 UNSD M49 表。文件缺失时，程序会提示并跳过相关绘图。来源及使用条件见[数据说明](data/README.md)。论文中按人均用户代表性绘制的地图未随代码发布。

## 运行分析流程

从仓库根目录执行 `jupyter lab`，依次打开并运行以下 Notebook。

0. **`00_generate_dummy_data.ipynb`** 按固定随机种子生成 `data/dummy_data.csv` 和 `data/dummy_data_hrv.csv`，已有文件时跳过。也可以改用自己的数据。
1. **`01_cohort_stats.ipynb`** 执行无需拟合模型的统计，包括通过 `seasonality_lib.apply_density_based_thinning` 对 22 岁处的 RHR 报告异常密集点下采样，生成按纬度分组的人口统计表、人口分布、季节性观测地图、用户内敏感性分析和下采样验证图。
2. **`02_modelling.ipynb`** 对每个指标各执行一次，在 `CONFIG` 中设定 `METRIC = 'rhr'` 或 `'hrv'`。特征工程包括 Gelman 标准化、年度及半年度谐波、对纬度与季节效应进行残差化处理的温度。 `stage_competition.run_competition` 比较六个阶段：基线（年龄与性别）、季节性（谐波或日照长度）、季节性与日照长度的交互（仅用于诊断）、空间（纬度及可选经度）、温度（均值或最高与最低温度）和联合比较。候选模型均为参与者随机截距 LME，由 `lme_binned.BinnedMixedLM` 拟合，估计与 `statsmodels.MixedLM` 相同，计算量随不同组大小的数量增长。每阶段取 BIC 最低的候选进入下一阶段，最终用限制最大似然（REML）重新拟合。 `STAGE_OVERRIDES` 可强制选择其他候选，分析替代路径。运行支持续跑，结果写入 `results/<run_tag>/`。
3. **`03_model_results.ipynb`** 通过 `RUN_TAGS` 选定一次运行，使用最优模型或通过 `MODEL_DIRS` 指定模型目录。输出 LME 表、每位用户的经验贝叶斯随机截距、残差诊断和 Q–Q 图。提供两个指标时，还生成按纬度带、国家和人口特征分析的图，以及全球残差图。第 2 节支持按模型比较两次运行。

Notebook 01 和 03 在页面内展示图表，LaTeX 源码折叠在表格预览下方。在 `CONFIG` 中设置 `SAVE_PAPER_ASSETS = True` 可将 `.tex` 表格、300 DPI PNG 以及 SVG 和 PDF 图形保存到 `results/paper/`。

Notebook 02 绘制联合国地区和国家趋势时，每组默认至少需要 500 名用户，可在 `CONFIG` 中调整 `MIN_USERS_PER_GROUP`。Notebook 01 的人口统计和下采样验证也会对年龄和纬度带应用最小样本数阈值。

## 输出与代码对应关系

| 输出 | Notebook | 生成函数 |
| --- | --- | --- |
| 按纬度带的人口统计（`tab:*_demographics_*`） | 01 | `paper_tables.demographics_table` |
| 人口分布和下采样验证 | 01 | `publication_figures.generate_figure_1 / _a1 / _a2 / _a3` |
| `fig:{rhr,hrv}_observed_map` | 01 | `publication_figures.plot_seasonal_maps_2x2` |
| `fig:within_user_sensitivity` | 01 | `publication_figures.make_within_user_sensitivity` |
| `tab:results_*` 等模型结果表 | 03 | `paper_tables.generate_all_tables` |
| `fig:{rhr,hrv}_residuals`、`fig:{rhr,hrv}_qq` | 03 | `publication_figures.make_model_residuals_plot / make_random_intercepts_plot` |
| `fig:lat_bin_fits`、`fig:countries_prediction_fit` | 03 | `publication_figures.make_combined_seasonal_fit_lat_bin / _country` |
| `fig:demographic_fit` | 03 | `publication_figures.make_joint_demographics_calibration` |
| `fig:global_residuals_grid` | 03 | `publication_figures.make_world_contour_residuals` |

## 第三方软件

依赖列于 `requirements.txt`，包括 NumPy、pandas、SciPy、statsmodels、scikit-learn、matplotlib、seaborn、plotly、GeoPandas 和 Jupyter，各自采用自己的许可证。 `src/lme_binned.py` 继承采用 BSD-3-Clause 的 `statsmodels.regression.mixed_linear_model.MixedLM`（© 2009–2018 statsmodels Developers），并返回 `MixedLMResults`。第三方源代码不随本仓库分发。

## 引用方式

使用该代码时，请引用文首论文。

## 许可证

Copyright 2026 Google LLC

The full license text is in [`LICENSE.txt`](LICENSE.txt).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## 免责声明

本项目不是 Google 官方提供支持的产品，也不属于 [Google 开源软件漏洞奖励计划](https://bughunters.google.com/open-source-security)的覆盖范围。它仅用于演示，不适合生产环境。此研究代码不属于医疗器械，也不适用于诊断、预后判断、治疗建议等临床用途。

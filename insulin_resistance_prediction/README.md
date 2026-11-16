# 基于可穿戴设备和常规血液生物标志物预测胰岛素抵抗

本目录包含[数据集](#数据集)和[参考代码](#参考代码结构)，对应论文[《基于可穿戴设备和常规血液生物标志物预测胰岛素抵抗》](https://arxiv.org/abs/2505.03784)。

## 概览

胰岛素抵抗是 2 型糖尿病的前兆，表现为组织对胰岛素的反应减弱。现有测量方式虽有效，但成本较高、可及性有限，影响早期干预。研究通过远程方式在美国招募 1,165 名参与者，BMI 中位数为 28 kg/m²，年龄中位数为 45 岁，HbA1c 中位数为 5.4%。研究采集了可穿戴设备时间序列和血液生物标志物，并采用胰岛素抵抗稳态模型评估值（HOMA-IR）作为参照。

研究团队训练深度神经网络预测胰岛素抵抗。联合使用可穿戴数据和常规血液生物标志物的效果优于使用任一单独来源，R² 为 0.5，AUROC 为 0.80，灵敏度为 76%，特异度为 84%。在肥胖且久坐的参与者中，灵敏度为 93%，调整后特异度为 95%。研究还检查了模型的可解释性和稳健性，并在独立的 72 人验证队列中复现预测表现。论文进一步展示了如何将预测结果接入大模型 Agent，帮助理解 HOMA-IR 值及其背景。研究探讨了借助这类信息更早识别 2 型糖尿病风险人群的可能性。

![研究设计](figures/insulin_resistance_prediction_fig_1.webp)

## 数据集

### 数据可用性

本研究中使用的解识别数据集仅供经批准的研究人员复制使用. 寻求数据集访问的研究人员必须填写胰岛素抵抗数据集访问请求表，[此处可用](https://docs.google.com/forms/d/e/1FAIpQLSebcfCZMQKBua7QS9MSvqr9n-fqmfJvmU4blOHXP1WZC0NFqA/viewform?usp=preview). 数据集访问请求表提供的电子邮件地址一旦获得批准，将添加到一个google组许可列表(`google-health-insulin-resistance-prediction-data-external@google.com`)中，该列表只允许读取一个Google云存储(GCS) Bucket存储数据集。在核准电子邮件中会给出关于如何访问水桶的说明。

从原始的解识别数据集中得出的汇总数据在`/data`文件夹中提供，所有文件都在[三角形格式](https://parquet.apache.org/)中提供，并被加载为[pandas 数据框](https://pandas.pydata.org/docs/index.html).

### ir_prediction_data .pq 数据

文件`ir_prediction_data.pq`包含一个用于胰岛素抵抗预测的综合数据集，每行代表一个单一的参与者. 列包含广泛的特性，包括：

* ID(`participant_id`):每个参与者独有的标识符。
* 数字生物标志:数据来自可穿戴设备,包括统计(平均值，中值，标准差)用于：
  * 静息心率 ( `RHR_FreeLiving_mean` , `RHR_FreeLiving_median` , `RHR_FreeLiving_std` )
  * 心率变异性 ( `HRV_FreeLiving_mean` , `HRV_FreeLiving_median` , `HRV_FreeLiving_std` )
  * 每日步骤( E) `STEPS_Daily_mean` , `STEPS_Daily_median` , `STEPS_Daily_std` )
  * 睡眠时间(`SLEEP_Duration_mean`,`SLEEP_Duration_median`,`SLEEP_Duration_std`).
* 实验室测量生物标记:各种血液测试结果，如：
  * 利皮德小组：`total cholesterol`,`hdl`,`triglycerides`,`ldl`.
  * `glucose`, `creatinine`, `sodium`, `potassium` (英语).
  * 其他关键标记如`hba1c`,`insulin`,以及`crp`.
* 人口统计:关于参与者的基本信息，包括`age`,`sex`,`bmi`,以及`ethnicity`.
* 健康条件和历史:自我报告的健康信息，如存在`hypertension`,`diabetes`,`CVD`(心血管疾病),以及吸烟状况`smoker`.
* 计算健康分数:从原始数据计算临床相关分数，包括：
  * `homa_ir`和`homa_ir_status`(HOMA-IR)是量化胰岛素抵抗的一种方法。
  * `framingham_risk` (Framingham风险心血管疾病得分)
  * `ascvd_risk` ASCVD 心血管疾病风险分数

### data.pq

文件 `data.pq` 包含一个用于胰岛素抵抗预测的综合数据集，每行代表一个单一参与者，列具有类似`ir_prediction_data.pq`的结构。

## 参考代码结构

- `IR_Analysis_Stats_Stratification_Visualization.ipynb`:本笔记本侧重于分析和可视化组群数据，它包括生成描述性统计的代码，根据BMI和身体活动对数据进行分层，并创建各种可视化，如箱形图，散块图，热图等，以探索不同特性和胰岛素抵抗之间的关系。

- `IR_MassiveAblationStudy_DeepLearningModels_InputFeatures_SiteDataTimeWindow_CrossValidationScheme.ipynb`:本笔记本使用深层学习模型，特别是马斯凯德自动编码器进行大规模减速研究，以预测HOMA-IR.它系统评价不同输入特性，时间窗口，以及交叉验证方法对模型性能的影响。

- `IR_MassiveAblationStudy_XGBoost_InputFeatures_SiteDataTimeWindow_CrossValidationScheme.ipynb`:这本笔记本与深层的学习衰竭研究类似，但侧重于XGBoost模型，它探讨了不同的输入特性，数据时间窗口，以及交叉验证方案如何影响XGBoost在预测HOMA-IR中的性能。

- `IR_VisualizingAblationResults.ipynb` :此笔记本是专用于可视化衰减研究的结果，它包括了各种图样来比较不同模型和特征集的性能，如条形图，点形图，散形图等，它还包括统计测试来比较模型性能。

## 依赖关系

用于 [提到可拉布笔记本](#参考代码结构) 标准为 [colab.research.google.com](https://colab.research.google.com)。所有蟒蛇代码都使用 [蟒蛇 3.10 (英语).](https://www.python.org/downloads/release/python-3100/) 公开发布 [Google Colab 内核](https://colab.research.google.com) .

在导入失败时，手动安装失败的软件包
通过运行 `!pip install <package>`在一个新的单元格中。

如果使用其他环境运行所提供的代码，请使用以下软件包版本：

  - 阿尔泰：5.5.0 (英语).
  - 3.7.3 (英语).
  - 马特普洛特利布：3.10.0
  - 数字：2.0.2
  - 腌菜:文件版本4.0
  - pandas : 2.2.2
  - 2025.2 (英语).
  - pytorch: (英语).
  - 重击：2.2.1
  - 剪辑： 1.16.3
  - 0.13.2 (法语).
  - 统计模型：0.14.6
  - 1.6.1 (美国英语).
  - 火炬：2.9.0 + cpu(英语：
  - 4.67.1 (英语).
  - IPython : 7.34.0
  - 隐藏( A) : 0.50.0
  - xgboost : (中文(简体) ). 3.1.2
  - 贾森： 2.0.9
  - 热量流：2.19.0
  - 0.5.9. 后2号

## 引用本文

```bib
@article{metwally2026insulin,
      title={Insulin Resistance Prediction From Wearables and Routine Blood Biomarkers}, 
      author={Ahmed A. Metwally and A. Ali Heydari and Daniel McDuff and Alexandru Solot and Zeinab Esmaeilpour and Anthony Z Faranesh and Menglian Zhou and David B. Savage and Conor Heneghan and Shwetak Patel and Cathy Speed and Javier L. Prieto},
      journal={Nature},
      volume={652},
      number={8109},
      pages={451--461},
      year={2026},
      publisher={Nature Publishing Group}
      url={https://doi.org/10.1038/s41586-026-10179-2}, 
}
```

## 参与贡献

目前，我们不接受外部对此的捐助。仓库。详情请参见
[CONTRIBUTING.md](https://github.com/Google-Health/consumer-health-research/tree/main/insulin_resistance_prediction/CONTRIBUTING.md) .

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

这不是官方支持的 Google 产品。该项目没有资格进入 [Google 开源软件脆弱性奖励方案](https://bughunters.google.com/open-source-security)。此项目仅用于演示目的，并非用于生产环境。

NOTE:本研究代码仓库的内容(一)无意成为医疗器械;和(二)无意用于任何种类的临床用途，包括但不限于诊断或预后。
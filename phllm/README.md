# 面向睡眠与健身指导的个人健康大语言模型

本目录提供发表于《Nature Medicine》的 [PH-LLM 论文](https://www.nature.com/articles/s41591-025-03888-0)所对应的[数据集](#数据集)和[参考代码](#参考代码结构)。

![PH-LLM 研究设计](docs/header.png)

> PH-LLM 从三个方面接受评估。第一，通过专业认证考试形式的单选题测试知识水平。第二，结合用户在睡眠和运动方面的目标给出个性化分析和建议。第三，根据按日汇总的数值传感器数据预测用户自报的睡眠质量。

## 概览

PH-LLM 是在 Gemini 的基础上微调的模型，用于理解按日汇总的可穿戴传感器数据，并支持睡眠与健身场景。研究团队建立了三套基准数据集，评估专业知识、个性化建议和自报睡眠质量预测。在睡眠医学考试中，PH-LLM 与参与对照的人类专家分别得分 79% 和 76%，在健身考试中分别为 88% 和 71%。在 857 个真实案例中，模型处理健身任务的表现与专家相当，给出个性化睡眠分析的表现优于原始 Gemini 模型。模型还使用多模态编码预测自报睡眠质量。项目提供数据集、评分标准和基准结果，供后续研究使用。

## 数据集

我们创建了三个基准测试数据集用于培训和/或评价PH-LLM:

1. 辅导建议
2. 专业考试
3. 病人报告的结果

在这里，我们提供了与公开来源、匿名的辅导建议以及专业检查和病人报告结果的获取程序合作的详细信息。我们还详细介绍了人类专家在评定辅导建议评级时所用的评价主题、AutoEval评分，以及用于重现主要数据数字和扩展数据数字的辅助数据文件。

### 辅导建议

我们为这两个领域提供了用于培训PH-LLM的匿名辅导建议，作为[`.jsonl`](https://jsonlines.org/)的文件：

- 睡觉：[sleep_case_studies. all.jsonl (英语).](https://github.com/Google-Health/consumer-health-research/blob/main/phllm/data/sleep_case_studies.all.jsonl)
- 适应性：[fitness_case_studies. all.jsonl (英语).](https://github.com/Google-Health/consumer-health-research/blob/main/phllm/data/fitness_case_studies.all.jsonl)

每个睡眠案例研究都以一个单 `json` 对象表示，其中包含以下密钥： 1.

- `vertical`:指案例研究域的字符串；`"sleep"`或`"fitness"`之一。
- `case_study_id`:一个独特的字符串标识符，表示具体的案例研究。
- `user_id`:一个独特的字符串标识符，表示与案例研究相关的用户。
- `input`:一个包含聚合传感器数据和相关元数据，用作模型或测算器输入的字符串。
- `split`:一串解码拆分;其中`"train"`,`"validation"`,`"test"`,`"holdout"`,或`"holdout_model"`.
- `etiology_output`:包含人类专家对病理部分的响应的弦。
- `insight_output`:包含人类专家对洞察部分的反应的弦。
- `recommendation_output`:包含人类专家对建议部分的回应的串。

每项健身案例研究均作为单项`json`标注，包含下列密钥： 1.

- `vertical`:指案例研究域的字符串；`"sleep"`或`"fitness"`之一。
- `case_study_id`:一个独特的字符串标识符，表示具体的案例研究。
- `user_id`:一个独特的字符串标识符，表示与案例研究相关的用户。
- `demographics_input`:一个包含人口统计数据的字符串，作为模型或测算器的输入。
- `health_metrics_input`:一个包含健康度量数据作为模型或测算器输入的字符串。
- `sleep_input`:包含睡眠数据的字符串，用于模型或测算器的输入。
- `training_load_input`:一个包含训练载荷数据的字符串，作为模型或测算器的输入。
- `split`:一串解码拆分;其中`"train"`,`"validation"`,`"test"`,`"holdout"`,或`"holdout_model"`.
- `demographics_output`:包含人口学部分人类专家反应的弦。
- `health_metrics_output`:包含人类专家对健康度量衡部分的反应的串。
- `muscle_soreness_input`:包含人体专家对肌肉疼痛部分的反应的弦。
- `readiness_assessment_output`:包含人类专家对准备状态评估部分的反应的弦。
- `sleep_output`:包含人类专家对睡眠部分的反应的弦。
- `subjective_readiness_input`:包含人类专家对主观准备部分的反应的弦。
- `training_load_output`:一个包含训练载荷部分的人类专家响应的字符串。

此外，我们还发布了健身案例研究的更新版本，并作了进一步的改进：

- 适应性：[fitness_case_studies. all.v2.jsonl (英语).](https://github.com/Google-Health/consumer-health-research/blob/main/phllm/data/fitness_case_studies.all.v2.jsonl)

这些案例研究已作了修订，以提高事实准确性，例如，对先前提到的数字和趋势作了具体更正。

个案研究是根据论文补充表格9中界定的标题进行的。

### 专业考试

睡眠和健身领域的多个选择问题(MCQs)应在相应的考试网站上访问。在这个仓库中，我们提供一个小演示CSV文件，其中包含我们参考代码执行所期望的格式的合成MCQs:[phllm/data/synthetic_mcq_data.csv](https://github.com/Google-Health/consumer-health-research/blob/main/phllm/data/synthetic_mcq_data.csv)。它预计有五个栏：

- `question`:包含问题后继的MCQ选项(例如`"What shoulder movements and planes of action are involved in a tennis backhand? (A) Adduction/frontal (B) Abduction/frontal (C) Adduction/transverse (D) Abduction/transverse"`)的字符串。
- `choices`: A JSON 对象字符串，包含每个字母选择的字典映射(即`"(A)"`)到相应的长式答案(如`"{'(A)': 'Adduction/frontal', '(B)': 'Abduction/frontal', '(C)': 'Adduction/transverse', '(D)': 'Abduction/transverse'}"`).
- `domain`:指标域的字符串；`"Sleep"`或`"Fitness"`之一。
- `difficulty`:指题难度的可选字符串；`"Easy"`,`"Moderate"`,`"Hard"`,或`None`.
- `answer`:正确答案(如`"(A)"`).

请注意，MCQs没有用于示范培训，所有样本都用于评价。

#### 睡眠考试

我们编译的 629 多重选择问题 MCQs 从。[BoardVitals](https://www.boardvitals.com) 我们用美国医学协会 AMA ) 物理学家表彰奖(Physician's Awards). PRA )"类别 1 - "睡眠医学"问题库，它仿照了美国内科医学委员会考试的内容( ABIM 睡眠医学认证考试，我们还使用了睡眠医学认证维护(英语:Sleep Medicine Certification Exampion). MOC 考试和纵向知识评估审查问题库，该库仿照考试内容 ABIM 睡眠医学 MOC 考试和 ABIM 纵向知识评估。MCQs 涉及与睡眠有关的广泛专题:正常睡眠和替代(N= 127 ),呼吸障碍(N= 84 ),超全息(N= 60 ),失眠症(N= 85 ),运动障碍(N= 23 ),寄生虫(N= 57 ),在其他疾病中睡觉(N= 112 ),和睡眠- wake 计时(N= 81 ).

#### 适应性考试

我们汇编了99MCQs,来源于仿照认证实力和条件专家(CSCS)考试内容的多个问题库[编制手册](https://www.nsca.com/certification/cscs/certified-strength-and-conditioning-specialist-exam-description),由[国家力量与条件协会 NSCA )](https://ncsf.org/)提供，我们使用了NSCA-CSCS教科书"实力训练和条件要素"中的考试问题。

### 病人报告的结果

评估 PH-LLM 预测病人报告的结果( PROs 从纵向被动传感器数据，我们使用了一个大型的 IRB -经核准的研究，为下列人口收集了可穿戴数据： 10,099 同意的个人，为期四周：[Google 健康数字福祉研究:数字设备使用与福祉研究议定书](https://www.researchprotocols.org/2024/1/e49189/) .

所有输入和目标预测数据，包括桶化年龄、20传感器在15天的日传感器读数和16二进制调查答复，对于PRO任务中的所有4,163样本，均以JSON格式存储。为确保参与者的隐私，在登记和证明数据使用协议后，经批准的研究人员可得到PRO任务所用的数据，用于非商业研究。请求应提交[google-digital-wellbeing-dataset@google.com](mailto:google-digital-wellbeing-dataset@google.com),并将每周评估一次。在请求批准时，将提供关于准确数据格式的其他细节。

### 辅助数据

这仓库还包含辅助数据，如人类专家以及AutoEval的评级，用于复制主数据及扩展数据数字。

- `data/autoeval_ratings/fitness_autoeval_external_ratings.tsv`:AutoEval全场收视率PH-LLM和健身领域的外部模型响应(扩展数据图7).
- `data/autoeval_ratings/fitness_autoeval_subsample_ratings.tsv`:AutoEval 健身领域各分样本的收视率PH-LLM(扩展数据图8).
- `data/autoeval_ratings/fitness_human_expert_ratings.tsv`:健身领域各回应来源的人类专家评级(主图2cd).
- `data/autoeval_ratings/sleep_autoeval_external_ratings.tsv`: AutoEval 收视率横跨PH-LLM 睡眠域的外部模型响应(扩展数据图7).
- `data/autoeval_ratings/sleep_autoeval_subsample_ratings.tsv`: AutoEval 亚采样的睡眠域中PH-LLM的响应(扩展数据图8)的收视率。
- `data/autoeval_ratings/sleep_human_expert_ratings.tsv`:睡眠域中不同响应源的人类专家评级(主图2cd).
- `data/pro_preds_bootstrap.tsv`: 启动 PROs 跨模型和目标的预测性能指标(主图3cd).
- `data/pro_prevalence.tsv`:PROs 病例流行跨目标。

## 参考代码结构

- `autoeval`:包含运行自动评价生成的案例研究的脚本和参考代码。
- `case_studies`:用于检查和直观化所创建的案例研究的哥拉布笔记本。
- `data`:此目录保存了我们的研究中使用的所有数据文件。详情请参见以上各节。
- `figures`:用于生成手稿主要数据及扩展数据数字的专用笔记本。
- `multimodal`:一个Colab笔记本，演示如何执行LLM多模态数据适配器，处理原始可穿戴感应数据。
- `professional_exams`:一个旨在运行MCQ的图书馆，专门对与睡眠和健身相关的专业考试进行评价。
- `protos`:包含用于构建我们案例研究数据的协议缓冲文件。

## 依赖关系

引用代码依赖性包括：

```
absl-py==2.3.1
immutabledict==4.2.1
numpy==2.2.2
pandas==2.2.3
tensorflow==2.18.0
# saxml==0.0.1  # https://github.com/google/saxml
```

所有蟒蛇代码都使用 [蟒蛇 3.11 (英语).](https://www.python.org/downloads/release/python-3110/)。使用 [Google Colab](https://colab.research.google.com)。所有代码都在内部运行 Google 使用自定义 Linux 内核的系统。类似 SKUs 可在下列网址查阅：[Google 云头](https://cloud.google.com)。关于每个组件运行的细节，请参见模块级文件。

## 引用 PH-LLM

如果您在研究中使用了PH-LLM数据集或代码，请引用手稿使用：

```
@article{khasentino2025phllm,
author={Khasentino, Justin and Belyaeva, Anastasiya and Liu, Xin and Yang, Zhun and Furlotte, Nicholas A. and Lee, Chace and Schenck, Erik and Patel, Yojan and Cui, Jian and Schneider, Logan Douglas and Bryant, Robby and Gomes, Ryan G. and Jiang, Allen and Lee, Roy and Liu, Yun and Perez, Javier and Rogers, Jameson K. and Speed, Cathy and Tailor, Shyam and Walker, Megan and Yu, Jeffrey and Althoff, Tim and Heneghan, Conor and Hernandez, John and Malhotra, Mark and Stern, Leor and Matias, Yossi and Corrado, Greg S. and Patel, Shwetak and Shetty, Shravya and Zhan, Jiening and Prabhakara, Shruthi and McDuff, Daniel and McLean, Cory Y.},
title={A personal health large language model for sleep and fitness coaching},
journal={Nature Medicine},
year={2025},
month={Aug},
day={14},
issn={1546-170X},
doi={10.1038/s41591-025-03888-0},
url={https://www.nature.com/articles/s41591-025-03888-0}
}
```

## 参与贡献

有关促进这方面工作的详细情况 仓库 请看 [CONTRIBUTING.md](https://github.com/Google-Health/consumer-health-research/blob/main/CONTRIBUTING.md) .
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

# 消费者健康研究（Consumer Health Research）

Google 消费者健康研究团队致力于帮助人们了解并改善自己的健康状况。

本仓库包含以下研究项目。每个目录都有自己的说明、代码或 Notebook。

| 项目 | 内容 |
| --- | --- |
| [个人健康 Agent](personal_health_agent/README.md) | 用可穿戴设备和活动数据做个性化分析。包含三个专用 Agent、编排器、两种对照架构和 Web 演示。 |
| [FHIR 检索基准](fhir-retrieval-bench/README.md) | 比较模型和不同检索策略处理 FHIR 健康记录问答的表现。 |
| [胰岛素抵抗预测](insulin_resistance_prediction/README.md) | 用可穿戴设备数据预测胰岛素抵抗，包含数据、分析和消融实验。 |
| [个人健康大语言模型（PH-LLM）](phllm/README.md) | 面向睡眠和运动等主题的个人健康问答及评估。 |
| [远程光电容积描记法（rPPG）](rppg/README.md) | 从视频中提取脉搏信号相关的研究数据。 |
| [SensorLM](sensorlm/README.md) | 将可穿戴设备传感器数据与自然语言对应起来的模型研究。 |
| [用于因果发现的合成可穿戴设备日志](synthetic_wearable_log_for_causal_discovery/README.md) | 以合成的用户活动数据做因果发现研究。 |
| [全球心脏生理模型](worldwide_model_cardiac_physiology/README.md) | 按地区和人群分析静息心率及心率变异性。 |
| [全球步数趋势](worldwide_step_trends/README.md) | 分析不同国家、人群和时期的步数趋势。 |

想了解个人健康 Agent 的实现，可以先看[框架说明](personal_health_agent/README.md)和[快速上手 Notebook](personal_health_agent/notebooks/01_quickstart.ipynb)，再看[完整编排流程](personal_health_agent/notebooks/05_full_pipeline.ipynb)。

*本仓库不是 Google 官方提供支持的产品。* [英文原仓库](https://github.com/Google-Health/consumer-health-research)

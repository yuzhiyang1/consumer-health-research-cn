# SensorLM：学习可穿戴传感器的语言

本目录提供发表于 NeurIPS 2025 的 [SensorLM 论文](https://arxiv.org/abs/2506.09108)所对应的[参考代码](#参考代码结构)。研究数据及构建方式见论文。

![SensorLM](assets/sensorlm.png)

> **面向可穿戴设备的传感器语言基础模型。** (A) 研究使用了超过 103,000 人、总计超过 5,970 万小时的多模态可穿戴设备数据。(B) 分层流程自动生成配对文本，刻画传感器数据的统计、结构和语义信息。(C) SensorLM 预训练框架及其下游任务。(D) 雷达图比较 SensorLM 与各基线在不同任务和设定下的表现。

## 概览

SensorLM 是一系列用自然语言理解可穿戴传感器数据的基础模型。真实场景的数据大量存在，但缺少与信号一一对应且信息丰富的文本描述，导致信号与语言难以对齐。研究团队设计了分层的描述生成流程，覆盖统计、结构和语义信息，构建了超过 103,000 人、总计超过 5,970 万小时的数据集。SensorLM 还将 CLIP 和 CoCa 等多模态预训练架构纳入统一框架。论文在人体活动和健康任务中检验了零样本识别、少样本学习和跨模态检索的表现，也讨论了规模效应、标注效率、传感器描述生成和对未见任务的零样本泛化。

## 参考代码结构
- `captioning.py`:包含等级代码描述生成流程.
- `constants.py`:包含用于预处理数据和生成标题的预定义模板和常数。
- `sensorlm_siglip`:包含训练SigLIP型号的SensorLM变体的代码。

## 引用 SensorLM

如果您在研究中使用了SensorLM数据集或代码，请引用手稿使用：

```bib
@inproceedings{zhang2025sensorlm,
    title={SensorLM: Learning the Language of Wearable Sensors},
    author={Yuwei Zhang and Kumar Ayush and Siyuan Qiao and A. Ali Heydari and Girish Narayanswamy and Maxwell A. Xu and Ahmed A. Metwally and Shawn Xu and Jake Garrison and Xuhai Xu and Tim Althoff and Yun Liu and Pushmeet Kohli and Jiening Zhan and Mark Malhotra and Shwetak Patel and Cecilia Mascolo and Xin Liu and Daniel McDuff and Yuzhe Yang},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2025}
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

# SensorLM - SigLIP
此文件夹包含[SigLIP](https://arxiv.org/abs/2303.15343)的[SensorLM](https://arxiv.org/abs/2506.09108)变体的训练代码。

此链接作为 [大视野](https://github.com/google-research/big_vision) 库的扩展，增加了对 SensorLM -基于模型的支持。

其中包括必要的配置文件、模型定义和预处理工具。


代码结构：

```
sensorlm_siglip/
│
├── configs/proj/image_text/
│   └── siglip_sensorlm.py         # Config for training SensorLM-SigLIP model
│
├── models/
│   └── vit.py                     # Modified Vision Transformer implementation (patch)
│
├── pp/
│   └── ops_image.py               # Custom sensor preprocessing operations
│
└── README.md
```

## 开始

对于代码的运行，克隆官方的Big Vision repo,然后将所提供的补丁文件复制到相应的目录中。

```
git clone https://github.com/google-research/big_vision.git
cd big_vision

# 将您的补丁文件复制到已有结构
cp -r ../sensorlm_siglip/*  ./big_vision
```

要训练或评价SigLIP—SensorLM型号，使用大视觉的标准发射装置与所提供的配置。


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

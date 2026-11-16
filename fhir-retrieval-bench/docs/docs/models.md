# 模型配置

## 概览

`config/model/<name>.yaml` 定义一次评估所用的回答模型（`answer`）、评判模型（`judge`）和嵌入模型（`embedding`）。回答和评判模型独立配置，可以由不同提供商提供，例如用 Claude 回答、用 Gemini 评判。

## 支持的提供商

| 提供商 | API 后端 | 认证方式 |
| --- | --- | --- |
| Google | GenAI | 环境变量 `GENAI_API_KEYS`，或者传入 `--use_vertex --gcp_project_locations project_id_1:location_1,project_id_2:location_2`。 |
| Anthropic | VertexAnthropic | 传入 `--use_vertex --gcp_project_locations project_id_1:location_1,project_id_2:location_2`。 |
| OpenAI | LiteLLM | 环境变量 `OPENAI_API_KEYS`。 |

后端按模型名前缀路由。`claude-*` 走 Vertex 的 Anthropic 后端，`openai/*` 或 `gpt-*` 走 LiteLLM，`google/*` 或 `gemini-*` 走 GenAI。添加模型时，可在 `config/model/` 中添加对应的 YAML 文件。

使用公开 API 时，可在环境变量中传入以逗号分隔的多个密钥，由程序轮换使用。

- Gemini：`export GENAI_API_KEYS="genai_key1,genai_key2,genai_key3"`
- OpenAI：`export OPENAI_API_KEYS="openai-key-1,openai-key-2"`

## 配置推理模式

`thinking_config` 用来设置模型的推理模式。设为 `null` 时，客户端会省略相关参数，使用提供商的服务端默认值。部分模型接受 `"high"`、`"medium"` 或 `"minimal"` 等档位，另一些模型需要指定推理 token 上限。具体格式请参阅模型提供商文档。也可以直接测试配置值，如果值不兼容，程序会返回格式建议。

## 在 Vertex AI 中启用模型

1. 在 [Google Cloud 控制台](https://console.cloud.google.com/)中打开项目。
2. 在 API 库里搜索并启用 **Vertex AI API**。
3. 打开 **Model Garden**。
4. 找到相应的模型（例如 `claude-opus-4-6`），点击 **Enable**。
5. 使用你的身份凭据填写启用表单。

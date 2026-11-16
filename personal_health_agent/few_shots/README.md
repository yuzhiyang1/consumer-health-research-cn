# 领域专家 Agent 少样本示例

这里的 Notebook 演示领域专家 Agent 如何使用 ReAct 工具。可先读[中文问题与答复摘要](EXAMPLES.zh-CN.md)，再打开对应的英文 Notebook 查看完整步骤。

## Notebook 格式

1. 第一个 Markdown 单元是用户问题。
2. 后续 Markdown 单元记录分析思路。代码单元执行工具或给出最终回答。
3. 搜索调用写为 `# search('query')`，加载器会提取引号中的搜索词。
4. 直接使用 Python 编写计算步骤。以 `print()` 输出最终回答。

加载器按这个顺序将单元转换成可供模型参考的 ReAct 示例。

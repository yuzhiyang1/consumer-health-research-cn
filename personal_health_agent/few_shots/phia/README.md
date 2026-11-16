# PHIA 单 Agent 少样本示例

PHIA 会读取这些 Notebook，转换成 OneTwo ReAct Agent 的示例。可先读[中文问题与答复摘要](EXAMPLES.zh-CN.md)，再打开英文 Notebook 查看代码和每一步的输出。

第一个单元是用户问题。第二个单元提供初始背景，接着交替给出分析思路、工具调用及返回结果。最后一步给出结论。加载后，各步骤依次包含 thought（分析）、action（动作）和 observation（观察），最终步骤以 is_finished 标记。

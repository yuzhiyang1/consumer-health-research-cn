# 领域专家 Agent 少样本示例

[阅读少样本示例的中文对照](EXAMPLES.zh-CN.md)

此文件夹包含Jupyter基于笔记本的 领域专家 Agent 示例。
这些实例教给模型正确的ReAct格式，用于工具使用。

## 格式

每个笔记本都遵循PHIA格式：
1. 第一个单元格( markdown): 问题/ 立即
2. 随后的单元格在：
   - Markdown : 思想/理性
   - 代码:工具使用或最后答案

### 工具使用格式

**搜索工具：**
```python
# search('your search query here')
```
搜索查询是从单引文中提取的。

**tool_code ( Python 执行：**
```python
# 实际 Python 代码到这里
some_function_call()
```

**最后答复：**
```python
# 最后答复
print("""Your comprehensive response here""")
```

## 实例包含

1. `cholesterol_interpretation.ipynb` - LDL/HDL/全胆固醇解释。
2. `hba1c_prediabetes.ipynb` - HbA1c 和糖尿病风险评估
3. `blood_pressure.ipynb` - 血压分类
4. `liver_enzymes.ipynb` - ALT / AST 肝功能
5. `kidney_function.ipynb` - Creatinine/eGFR 肾脏评估。
6. `thyroid_tsh.ipynb` - TSH和甲状腺功能
7. `vitamin_d.ipynb` - 维生素D缺乏症
8. `improve_heart_health.ipynb` - 普通心血管保健咨询
9. `iron_anemia.ipynb` - 费里廷/血红素和缺铁性贫血
10. `metabolic_summary.ipynb` - 带有参考范围的元板摘要
11. `rhr_hrv.ipynb` - 静息心率和HRV 解释。

## 添加新实例

要添加一个新的示例：
1. 按照上面的格式创建一个新的 `.ipynb`文件
2. 确保第一个单元格包含问题
3. 使用 `# search('query')` 搜索工具呼叫
4. 使用常规 Python 代码用于 tool_code 呼叫
5. 结尾为最后答案的 `print()` 语句

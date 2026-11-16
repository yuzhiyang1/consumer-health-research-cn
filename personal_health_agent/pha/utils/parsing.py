"""Utilities for parsing LLM outputs."""

# V2 代码提示以此函数签名结束，此函数签名是 LLM
# 继续。我们需要把它放在提取的代码上。
ANALYSIS_FUNCTION_SIGNATURE = """\
from typing import Any, Dict
import pandas as pd
import numpy as np
import datetime

def analysis(summary_df: pd.DataFrame | None = None,
    activities_df: pd.DataFrame | None = None,
    profile_df: pd.DataFrame | None = None,
    population_df: pd.DataFrame | None = None,) -> Dict[str, Any]:
"""


def _normalize_body_indentation(code: str, target_base_indent: int = 4) -> str:
    """Normalize indentation of a code block to use a consistent base indent.
    
    Args:
        code: The code to normalize
        target_base_indent: The base indentation level (default 4 for function body)
        
    Returns:
        Code with normalized indentation
    """
    lines = code.split('\n')
    
    # 在非空行中查找最小缩进
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # 非空线
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == float('inf') or min_indent == 0:
        min_indent = 0
    
    # 计算需要的缩进调整
    indent_adjustment = target_base_indent - min_indent
    
    # 调整每行缩进
    normalized_lines = []
    for line in lines:
        if line.strip():  # 非空线
            current_indent = len(line) - len(line.lstrip())
            new_indent = max(0, current_indent + indent_adjustment)
            normalized_lines.append(' ' * new_indent + line.lstrip())
        else:
            normalized_lines.append('')
    
    return '\n'.join(normalized_lines)


def parse_code_output(code_output: str) -> str:
    """Parses the code output to extract the final result.
    
    The V2 codegen prompt ends with a function signature that the LLM continues.
    If the extracted code doesn't start with 'def analysis', we need to prepend
    the function signature.
    """
    code_output = code_output.removeprefix("```python")
    code_output = code_output.removeprefix("```")
    code_output = code_output.removesuffix("```")
    
    # 仅条纹跟踪白空格，保存主白空格以进行缩进检测
    code_output = code_output.rstrip()
    # 仅删除主线新行(非空格)
    while code_output.startswith('\n'):
        code_output = code_output[1:]
    
    # 检查代码是否已经有函数定义
    # (万一LLM重复，或者我们使用V1 即时)
    if code_output.lstrip().startswith('def analysis') or code_output.lstrip().startswith('from typing'):
        return code_output.strip()
    
    # 检查代码是否从导入语句开始( LLM 包含完整代码)
    if code_output.lstrip().startswith('import '):
        return code_output.strip()
    
    # 否则，在函数签名之前
    # LLM 输出为具有自身缩进功能的函数体
    # 我们需要实现 4— 函数体的空间基缩进
    normalized_body = _normalize_body_indentation(code_output, target_base_indent=4)
    
    return ANALYSIS_FUNCTION_SIGNATURE + normalized_body

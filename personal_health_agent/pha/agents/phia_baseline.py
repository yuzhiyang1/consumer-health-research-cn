"""PHIA Single-Agent Baseline for PHA.

This baseline implements the original Personal Health Insights Agent (PHIA) architecture:
a single ReAct agent that handles all health queries without specialization or orchestration.

This serves as a comparison baseline to show the value of multi-agent specialization
in the PHA system.

IMPORTANT - Model Configuration:
    This baseline uses Gemini 2.5 Pro by default, matching the original PHIA implementation.
    The original PHIA was designed for Gemini 1.5 models, but those were deprecated.
    Gemini 2.5 Pro provides the best ReAct loop completion for this architecture.
    
    For fair baseline comparisons, note that:
    - PHIA baseline uses: models/gemini-2.5-pro (fixed)
    - PHA/Parallel should use the same model family when comparing architectures

Usage:
    from pha.agents import PHIABaseline, create_phia_baseline
    
    baseline = create_phia_baseline(
        gemini_api_key="your-key",
        tavily_api_key="your-tavily-key",
    )
    response = baseline.respond("How is my sleep quality?")

Research Question: "Is specialization better than one generalist?"
"""

# 调试版本 - 在每个固定器上递增
__version__ = "1.11.0"  # 固定 is_expression从复合区块中剥离打印( ) + 搜索回收

import io
import os
import re
import contextlib
import textwrap
from typing import Optional, Dict, Any, List, Sequence
from datetime import datetime

import pandas as pd
import numpy as np

# 导入共享请求数据
try:
    from ..utils.request_stats import request_stats
    _STATS_AVAILABLE = True
except ImportError:
    _STATS_AVAILABLE = False


# =============================================================================
# 工具调用日志
# =============================================================================

def _log_tool_call(tool_name: str, inputs: dict, output: Any = None, error: str = None):
    """Log a tool call to the terminal for debugging.
    
    Args:
        tool_name: Name of the tool being called.
        inputs: Dictionary of input arguments.
        output: The tool output (will be truncated for display).
        error: Error message if the tool failed.
    """
    # 记录数据
    if _STATS_AVAILABLE:
        request_stats.record_tool(tool_name, had_error=error is not None)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # 格式输入
    input_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in inputs.items())
    
    print(f"\n{'─' * 60}")
    print(f"🔧 [{timestamp}] TOOL CALL: {tool_name}")
    print(f"   Input: {input_str[:200]}{'...' if len(input_str) > 200 else ''}")
    
    if error:
        print(f"   ❌ Error: {error[:200]}")
    elif output is not None:
        # 显示的截断输出
        output_str = str(output)
        if len(output_str) > 300:
            output_str = output_str[:300] + "..."
        print(f"   ✓ Output: {output_str}")
    
    print(f"{'─' * 60}\n")


def clean_agent_response(text: str) -> str:
    """Clean internal agent labels from response text.
    
    Removes internal markers that shouldn't be shown to users:
    - [SUMMARY] tags from data science agent
    - [CODE UPDATE] tags from data science agent  
    - Coach: prefixes from health coach agent
    - Expert: prefixes from domain expert agent
    - [Finish]: prefix from ReAct agents
    - JSON-escaped newlines
    - Quoted string wrappers
    
    Args:
        text: Raw response text from agents.
        
    Returns:
        Cleaned response suitable for user display.
    """
    if not text:
        return text
    
    # 处理 JSON - 逃逸的新线(将\n字转换为实际的新线)
    if '\\n' in text:
        text = text.replace('\\n', '\n')
    
    # 如果引用了整个文字，删除周围的引文
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # 删除 SUMMARY ]和 CODEUPDATE 标记(在之后保留内容)
    text = re.sub(r'^\s*\[SUMMARY\]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\[CODE UPDATE\]\s*', '', text, flags=re.MULTILINE)
    
    # 删除教练： 响应开始前缀
    text = re.sub(r'^Coach:\s*', '', text, flags=re.MULTILINE)
    
    # 删除专家： 响应开始前缀
    text = re.sub(r'^Expert:\s*', '', text, flags=re.MULTILINE)
    
    # 删除 [完成]:前缀 (ReAct 最终答案标记)
    # 处理各种格式：[完成]:,[完成]:,[完成]:,[完成]:\n等。
    text = re.sub(r'^\s*\[Finish\]:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[Finish\]:\s*\n*', '', text)  # 还要抓住中文本的出现
    
    # 检测和拒绝系统即时泄漏
    # 这些模式显示模型正在输出指令，而不是响应
    # 将文本规范化以供检测(手性椭圆、额外空格等)
    text_normalized = text.lower().replace('…', '...').replace('  ', ' ')
    
    prompt_leak_indicators = [
        # 来自编排器/域专家提示
        'you must do so safely and transparently',
        'quantify uncertainty when appropriate',
        'avoid diagnosing or prescribing',
        'always recommend appropriate follow-up',
        'you should use the provided tools',
        'if a tool fails or required user context',
        'explicitly state the limitation',
        # 从 领域专家 Agent
        'clinically accurate, evidence-based',
        'safety-conscious guidance',
        'explicitly separate objective findings',
        'state uncertainty and limitations when data',
        'avoid diagnosing when criteria are not met',
        'recommend appropriate follow-up',
        'cite authoritative sources with links',
        'request only the minimum additional data',
        'wearables, labs, imaging, notes',
        'urgent/emergent evaluation when red flags',
        # 一般教学模式
        'you will analyze and synthesize',
        'you will avoid',
        'you will cite',
        'you will request',
        'with the goal of delivering',
        'delivering clinically accurate',
        # 工具指令模式( 不应该在最后回复中)
        'e.g., `tool_code` for math',
        'for math, `compare_blood_test',
        '`datacommons_natural_language_query`',
        'via the `search` tool',
        'use the provided tools',
        # 指令式图案
        'you are tasked with',
        'as a domain expert',
        'you are a domain expert',
        'you have access to a number of tools',
        'it is imperative that you',
    ]
    for indicator in prompt_leak_indicators:
        if indicator.lower() in text_normalized:
            print(f"[PHIA:clean] *** PROMPT LEAK DETECTED: '{indicator}' found in response, replacing with error message")
            print(f"[PHIA:clean] *** Original response was ({len(text)} chars): {repr(text[:300])}")
            return (
                "⚠️ **Response contained system prompt leakage** — the model echoed its instructions "
                "instead of answering your question.\n\n"
                f"**Matched pattern:** `{indicator}`\n\n"
                "This is a known issue with some models. Please try again, rephrase your question, "
                "or switch to a different model."
            )
    
    # 清除任何额外的空白
    text = text.strip()
    
    return text


# 尝试导入一个 2 个组件
_ONETWO_AVAILABLE = False
_ONETWO_OPENAI_AVAILABLE = False
_ONETWO_GENAI_AVAILABLE = False  # 如果有新的 GoogleGenAIAPI 后端，则为真
_ONETWO_IMPORT_ERROR = None


# =============================================================================
# OpenAI API 兼容性补丁
# =============================================================================
# 较新的OpenAI型号需要`max_completion_tokens`而不是`max_tokens`.

_OPENAI_PATCHED = False

def _patch_openai_for_compatibility():
    """Patch OpenAI client for compatibility with newer models and OneTwo.
    
    Handles:
    - max_tokens → max_completion_tokens conversion
    - Removes 'stop' parameter for GPT-5 models (not supported)
    - Transforms tool call format for OneTwo compatibility
    """
    global _OPENAI_PATCHED
    if _OPENAI_PATCHED:
        return
    
    try:
        from openai.resources.chat import completions
        import re
        
        def _extract_balanced_parens(text: str, start: int) -> tuple:
            """Extract content inside balanced parentheses starting at start position.
            
            Returns (content, end_position) or (None, -1) if not balanced.
            """
            if start >= len(text) or text[start] != '(':
                return None, -1
            
            depth = 0
            i = start
            while i < len(text):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return text[start+1:i], i
                i += 1
            return None, -1
        
        def _transform_tool_calls_for_onetwo(text: str) -> str:
            """Transform GPT-style tool calls to OneTwo-compatible format.
            
            GPT outputs: tool_code(summary_df['steps'].mean())
            OneTwo expects: tool_code('summary_df["steps"].mean()')
            
            This function wraps the code expression in quotes so OneTwo can parse it.
            """
            if not text:
                return text
            
            result = []
            i = 0
            tool_names = ['tool_code', 'search']
            
            while i < len(text):
                found_tool = False
                for tool_name in tool_names:
                    # 检查 [Act] :前缀或直接工具调用
                    if text[i:].startswith('[Act]:'):
                        # 在 [Act] 后查找工具名：
                        act_match = re.match(r'\[Act\]:\s*', text[i:])
                        if act_match:
                            prefix = act_match.group(0)
                            tool_start = i + len(prefix)
                            for tn in tool_names:
                                if text[tool_start:].startswith(tn + '('):
                                    arg_content, end_pos = _extract_balanced_parens(text, tool_start + len(tn))
                                    if arg_content is not None:
                                        arg_stripped = arg_content.strip()
                                        # 检查引用是否正确
                                        is_quoted = (
                                            (arg_stripped.startswith('"') and arg_stripped.endswith('"')) or
                                            (arg_stripped.startswith("'") and arg_stripped.endswith("'")) or
                                            arg_stripped.startswith('"""') or
                                            arg_stripped.startswith("'''")
                                        )
                                        if is_quoted:
                                            result.append(text[i:end_pos+1])
                                        else:
                                            # 以引号环绕，避开已有引号
                                            escaped = arg_content.replace("'", '"')
                                            result.append(f"{prefix}{tn}('{escaped}')")
                                        i = end_pos + 1
                                        found_tool = True
                                        break
                            if found_tool:
                                break
                    elif text[i:].startswith(tool_name + '('):
                        arg_content, end_pos = _extract_balanced_parens(text, i + len(tool_name))
                        if arg_content is not None:
                            arg_stripped = arg_content.strip()
                            is_quoted = (
                                (arg_stripped.startswith('"') and arg_stripped.endswith('"')) or
                                (arg_stripped.startswith("'") and arg_stripped.endswith("'")) or
                                arg_stripped.startswith('"""') or
                                arg_stripped.startswith("'''")
                            )
                            if is_quoted:
                                result.append(text[i:end_pos+1])
                            else:
                                escaped = arg_content.replace("'", '"')
                                result.append(f"{tool_name}('{escaped}')")
                            i = end_pos + 1
                            found_tool = True
                            break
                
                if not found_tool:
                    result.append(text[i])
                    i += 1
            
            return ''.join(result)
        
        def _truncate_at_stop_sequences(text: str, stop_sequences: list) -> str:
            """Truncate text at the first occurrence of any stop sequence.
            
            Backup for cases where stop parameter doesn't work as expected.
            """
            if not text or not stop_sequences:
                return text
            
            earliest_pos = len(text)
            for stop_seq in stop_sequences:
                pos = text.find(stop_seq)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            
            return text[:earliest_pos]
        
        _original_create = completions.Completions.create
        
        def _patched_create(self, *args, **kwargs):
            # 将 max_tokens 转换为 max_completion_tokens 用于更新 API
            if 'max_tokens' in kwargs and 'max_completion_tokens' not in kwargs:
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            
            model = kwargs.get('model', '')
            is_gpt5 = model and 'gpt-5' in model.lower()
            
            # 注：reasoning_effort 参数只与响应API配合，而不是聊天补全
            # 对于聊天完成，我们依靠明确的提示代替
            
            # 保存备份手动截断的停止序列
            saved_stop_sequences = kwargs.get('stop')

            # GPT-5 推理模型拒绝 `stop` 与 HTTP 400.
            # 在发送之前，请求是成功的，并依赖
            # `_truncate_at_stop_sequences`下面为纪念请求的停留
            # 返回内容上的语义。
            if is_gpt5 and 'stop' in kwargs:
                kwargs.pop('stop')

            # 调用原始方法
            result = _original_create(self, *args, **kwargs)
            
            # 处理后回复
            try:
                if hasattr(result, 'choices') and result.choices:
                    for choice in result.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            if choice.message.content:
                                content = choice.message.content
                                
                                # GPT-5: 备份手动截断在停止序列
                                # (以防API不尊重他们)
                                if is_gpt5 and saved_stop_sequences:
                                    content = _truncate_at_stop_sequences(content, saved_stop_sequences)
                                
                                # 变形工具调用 OneTwo 兼容性
                                content = _transform_tool_calls_for_onetwo(content)
                                
                                choice.message.content = content
            except Exception:
                pass
            
            return result
        
        completions.Completions.create = _patched_create
        _OPENAI_PATCHED = True
    except Exception:
        pass


try:
    from onetwo import ot
    from onetwo.agents import react
    from onetwo.stdlib.tool_use import llm_tool_use
    from onetwo.stdlib.tool_use import python_tool_use
    
    # 使用 GoogleGenAIAPI 后端(新google.genai SDK,无贬值警告，
    # Gemini2.5+模型的安全默认值为OFF
    try:
        from onetwo.backends import google_genai_api
        _ONETWO_GENAI_AVAILABLE = True
    except ImportError:
        pass  # 会在后端施工时间明显出错
    
    _ONETWO_AVAILABLE = True
    
    # 尝试导入 OpenAI 后端
    try:
        from onetwo.backends import openai_api
        _ONETWO_OPENAI_AVAILABLE = True
        _patch_openai_for_compatibility()
    except ImportError:
        pass
except ImportError as e:
    _ONETWO_IMPORT_ERROR = str(e)
except SyntaxError as e:
    import sys
    _ONETWO_IMPORT_ERROR = f"onetwo requires Python 3.12+, you have {sys.version}"
except Exception as e:
    _ONETWO_IMPORT_ERROR = str(e)

# 基于笔记本的示例解析
try:
    import nbformat
    _NBFORMAT_AVAILABLE = True
except ImportError:
    _NBFORMAT_AVAILABLE = False


# =============================================================================
# PHIA 序言 -- -- 及时提供综合专门知识
# =============================================================================

PHIA_PREAMBLE = textwrap.dedent("""\
{#- Preamble: Agent functionality description -#}
处理 Fitbit 数据相关问题。可以通过 `pd` 和 `np` 使用 pandas 和 numpy，无法使用 matplotlib 等其他 Python 库。
参考工具示例，按 ReAct 协议逐步调用工具，每次只调用一个。对话可能有多轮，应结合当前问题与必要的前文。

#### 可用的 DataFrame

- `summary_df`：用户每日活动和睡眠摘要。
    - `datetime`：记录日期（datetime64[ns]）。
    - `resting_heart_rate`：平均静息心率（bpm，float64）。
    - `heart_rate_variability`：心率变异性（float64）。
    - `fatburn_active_zone_minutes`：脂肪燃烧心率区间的时长（分钟，float64）。
    - `cardio_active_zone_minutes`：有氧运动心率区间的时长（分钟，float64）。
    - `peak_active_zone_minutes`：峰值心率区间的时长（分钟，float64）。
    - `active_zone_minutes`：每日活跃区间分钟数（float64）。
    - `steps`：每日步数（float64）。
    - `rem_sleep_minutes`：REM 睡眠时长（分钟，float64）。
    - `deep_sleep_minutes`：深睡时长（分钟，float64）。
    - `awake_minutes`：清醒时长（分钟，float64）。
    - `light_sleep_minutes`：浅睡时长（分钟，float64）。
    - `sleep_minutes`：总睡眠时长（分钟，float64）。
    - `bed_time`：就寝时间（datetime64[ns]）。
    - `wake_up_time`：起床时间（datetime64[ns]）。
    - `stress_management_score`：每日压力管理得分（float64）。
    - `deep_sleep_percent`：深睡百分比（float64，可选）。缺少时用 `deep_sleep_minutes / sleep_minutes * 100` 计算。
    - `rem_sleep_percent`：REM 睡眠百分比（float64，可选）。缺少时用 `rem_sleep_minutes / sleep_minutes * 100` 计算。
    - `awake_percent`：清醒百分比（float64，可选）。缺少时用 `awake_minutes / sleep_minutes * 100` 计算。
    - `light_sleep_percent`：浅睡百分比（float64，可选）。缺少时用 `light_sleep_minutes / sleep_minutes * 100` 计算。

- `activities_df`：用户运动记录。
    - `start_time`：开始时间（datetime64[ns]）。
    - `end_time`：结束时间（datetime64[ns]）。
    - `activity_name`：活动类型（str），例如 "WALKING"、"RUNNING"、"OUTDOOR_BIKE"、"STRENGTH_TRAINING"、"WORKOUT"、"ELLIPTICAL"、"YOGA"、"TREADMILL"、"HIKING"、"SWIMMING"。
    - `distance`：活动距离（英里，float64）。
    - `duration`：活动时长（分钟，float64）。
    - `calories`：消耗的热量（float64）。
    - `steps`：活动步数（float64）。
    - `active_zone_minutes`：活动获得的活跃区间分钟数（float64）。

实际字段随数据集变化。遇到 KeyError 时，可用 `print(activities_df.columns)` 查看字段。

- `profile`：用户档案。
    - `age`：年龄（int）。
    - `gender`：性别（str）。
    - `averageDailySteps`：历史日均步数（int）。
    - `elderly`：是否为老年人（字符串 "Yes" 或 "No"）。
    - `height_cm`：身高（厘米，int）。
    - `weight_kg`：体重（千克，int）。

读取年龄、性别、身高或体重时，使用 `profile['age'].values[0]`、`profile['gender'].values[0]` 等方法读取单个值。已有数据能回答时，无需再次询问用户。
""")

PHIA_REACT_PROMPT_TEXT = """\
{#- Preamble: Tools description -#}
{%- role name='system' -%}
你是一个使用工具逐步解决问题的 ReAct Agent。按以下协议实际调用工具并读取结果。

可用工具：
{% for tool in tools %}
工具名称：{{ tool.name }}
工具说明：{{ tool.description }}
{% if tool.example -%}
  工具示例：{{ tool.example_str }}
{%- endif -%}
{% endfor %}

ReAct 协议：
1. [Thought]: 用一到两句话说明下一步。
2. [Act]: 调用一个工具，并传入所需参数。
3. [Observe]: 等待并查看结果。
4. 重复第 1–3 步，直到取得回答所需的信息。
5. [Finish]: 给出完整的最终回复，使用用户最近一轮发言的语言，除非用户另有要求。

格式要求：
- [Act] 行只写一个函数调用，保持在同一行。
- 调用 tool_code 时，将所有代码放在一个带引号的字符串参数内。多行代码用 `\\n` 表示换行。
- 代码不超过 15 行。先直接计算，无需用 if/else 预判空数据。遇到错误时下一步再处理。
- 工具调用不加 Markdown 代码块或反引号。执行工具后等待 [Observe]，再依据实际结果回答。
- 在 tool_code 中使用 print() 输出带标签的结果，例如 print(f"Average RHR: {val:.2f} bpm")。不要只返回值或元组。
- 将结果、提示和建议都写入 [Finish]。不要把关键信息只放在 [Thought] 中，也不要用冒号结束回答或承诺尚未给出的内容。

正确示例：
[Thought]: I need to calculate the average steps from the data.
[Act]: tool_code('avg_steps = summary_df["steps"].mean()\nprint(f"Average daily steps: {avg_steps:.0f}")')
[Observe]: Average daily steps: 6543
[Thought]: Now I have the average. Let me provide the answer.
[Finish]: Your average daily steps is 6,543 steps.

应避免的输出：
- 只计划计算、搜索和再次计算，没有调用工具。
- 将 [Act] 的工具调用包在 Markdown 代码块里。
- 原样返回搜索结果，没有整理成回答。

{#- Preamble: ReAct few-shots #}
以下示例展示如何用工具解决不同任务。参考其中的步骤，根据当前问题独立作答。
{% for example in exemplars %}
[{{ stop_prefix }}Question]: {{ example.inputs + '\n' }}
{%- for step in example.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\n' }}
{%- endif -%}
{%- endfor -%}
{%- endfor -%}

逐步执行 ReAct 循环。依次分析、行动、观察，信息足够后用 [Finish] 给出最终回复。

{# Start of the processing of the actual inputs. -#}

下面给出当前问题和处理进度。
{#- Render the original question. -#}
{%- endrole -%}
{%- role name='user' %}
[{%- role name='system' -%}{{ stop_prefix }}{%- endrole -%}Question]: {{ state.inputs + '\n' }}
{%- endrole -%}

{# Render the current state (i.e., any steps performed up till now). -#}
{%- for step in state.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\n' }}
{%- endif -%}
{%- endfor -%}

{# If force-finishing, then prompt the LLM for the final answer. -#}
{%- if force_finish -%}
  [Finish]:{{ ' ' }}
{%- endif -%}

{#- Get a response from the LLM and return it. -#}
{%- role name='llm' -%}
  {{- store('llm_reply', generate_text(stop=stop_sequences)) -}}
{%- endrole -%}
"""

# 保留 [Finish] 和工具名称，OneTwo 会按固定格式解析。
QUESTION_PREFIX = (
    "需要计算时使用 tool_code 执行 Python，需要外部资料时使用 search。"
    "超出健康和日常保健范围的问题要如实说明无法回答。"
    "遇到拼写和语法错误时先尝试理解用户的真实意思。严格遵守 ReAct 模板，给出结构清晰的回答。"
    "用户表达健康担忧时，先用 tool_code 核对实际的睡眠、心率或活动数据，再给一般性建议。"
    "用户要求查看数据时，输出实际数值和统计量。"
    "tool_code 始终使用 print() 输出带说明的数值，不单独返回值或元组。"
    "把全部细节和建议写入 [Finish]，不要以冒号收尾。使用用户最近一轮发言的语言，除非用户另有要求。"
    "Question: "
)


# =============================================================================
# 工具名称
# =============================================================================

_PYTHON_TOOL_NAME = "tool_code"
_SEARCH_TOOL_NAME = "search"
_FINISH_TOOL_NAME = "finish"


# =============================================================================
# 帮助函数
# =============================================================================

def _autofix_truncated_code(code: str) -> str:
    """Fix code truncated during extraction from tool_code('...').
    
    When models generate long code inside tool_code('...'), the string
    can get truncated (by token limits or argument parsing), leaving
    compound statements like ``else:`` with no body — a SyntaxError
    that the model then loops on 10x without fixing.
    
    Strategy: detect truncation (empty compound-statement body) and
    insert ``pass`` so the code can execute. The truncated branch is
    almost always a non-essential guard (e.g., ``else: print("No data")``),
    while the main analysis lives in the ``if`` block.
    """
    try:
        compile(code, '<string>', 'exec')
        return code  # 已经有效
    except SyntaxError as e:
        error_msg = str(e)
        if 'expected an indented block' not in error_msg:
            return code  # 不同的错误，不要碰
        error_lineno = e.lineno
    
    lines = code.split('\n')
    
    # 错误消息包含实际的复合语句行：
    # "期待在"else"的台词后出现缩进区块3".
    # e. lineno 可能指向别处(到下一个代码行),所以
    # 解析消息的真实源行。
    msg_match = re.search(r'on line (\d+)', error_msg)
    stmt_lineno = int(msg_match.group(1)) if msg_match else error_lineno
    
    if not stmt_lineno or stmt_lineno < 1 or stmt_lineno > len(lines):
        return code
    
    # 插入 `pass`作为空复合语句的正文。
    offending_line = lines[stmt_lineno - 1]
    indent = len(offending_line) - len(offending_line.lstrip())
    lines.insert(stmt_lineno, ' ' * (indent + 4) + 'pass')
    fixed = '\n'.join(lines)
    
    try:
        compile(fixed, '<string>', 'exec')
        print(f"[PHIA:autofix] Fixed truncated block on line {stmt_lineno}: "
              f"'{offending_line.strip()}' → added pass")
        return fixed
    except SyntaxError:
        return code  # 修补没用的，还原吧


def simple_python_executor(code: str, sandbox_globals: dict) -> str:
    """Executes python code and captures the output.
    
    Args:
        code: Python code to execute.
        sandbox_globals: Global variables available to the code.
        
    Returns:
        The output of the code execution.
    """
    _log_tool_call("tool_code", {"code": code[:100] + "..." if len(code) > 100 else code})
    
    # 自动修补短路码块(如： 其它:没有来自
    # tool_code 字符串切换。没有这个，模型循环打开
    # 同样的SyntaxError,用于max_steps迭代。
    code = _autofix_truncated_code(code)
    
    local_scope = {}
    string_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(string_buffer):
            code_lines = code.strip().split('\n')
            last_line = code_lines[-1].strip() if code_lines else ""
            
            # 确定最后一行是否为表达式
            is_expression = (
                last_line and 
                not last_line.startswith(('import ', 'from ', 'def ', 'class ', 
                                          'if ', 'for ', 'while ', 'with ', 
                                          'try:', 'except', 'finally', 'elif ', 'else:',
                                          'pass', 'break', 'continue', 'return',
                                          'raise', 'del ', 'assert ',
                                          'print(', 'print (')) and
                '=' not in last_line.split('#')[0] and
                not last_line.startswith('#')
            )
            
            if is_expression and len(code_lines) > 1:
                exec('\n'.join(code_lines[:-1]), sandbox_globals, local_scope)
                result = eval(last_line, {**sandbox_globals, **local_scope}, local_scope)
                printed_output = string_buffer.getvalue()
                output = printed_output if printed_output else str(result)
            else:
                exec(code, sandbox_globals, local_scope)
                printed_output = string_buffer.getvalue()
                
                if not printed_output and is_expression:
                    try:
                        result = eval(last_line, {**sandbox_globals, **local_scope}, local_scope)
                        output = str(result)
                    except:
                        output = printed_output or "Code executed successfully."
                else:
                    output = printed_output or "Code executed successfully."
        
        _log_tool_call("tool_code", {"code": "..."}, output=output[:200] if len(output) > 200 else output)
        return output
        
    except Exception as e:
        error_str = str(e)
        
        # 内线重试:固定截断的复合语句(如:其他:无躯体).
        # 这是第二道防线——上面的_autofix_truncated_code
        # 抓住这个，但如果它没有 任何理由，在这里处理它。
        if 'expected an indented block' in error_str:
            try:
                # 从错误消息中解析语句行
                msg_match = re.search(r'on line (\d+)', error_str)
                if msg_match:
                    fix_lineno = int(msg_match.group(1))
                    fix_lines = code.split('\n')
                    if 1 <= fix_lineno <= len(fix_lines):
                        stmt = fix_lines[fix_lineno - 1]
                        indent = len(stmt) - len(stmt.lstrip())
                        fix_lines.insert(fix_lineno, ' ' * (indent + 4) + 'pass')
                        fixed_code = '\n'.join(fix_lines)
                        
                        # 尝试执行固定代码
                        fix_buffer = io.StringIO()
                        fix_scope = {}
                        with contextlib.redirect_stdout(fix_buffer):
                            exec(fixed_code, sandbox_globals, fix_scope)
                        fix_output = fix_buffer.getvalue() or "Code executed successfully."
                        print(f"[PHIA:autofix-retry] Fixed '{stmt.strip()}' on line {fix_lineno} "
                              f"and executed successfully")
                        _log_tool_call("tool_code", {"code": "..."}, 
                                       output=fix_output[:200] if len(fix_output) > 200 else fix_output)
                        return fix_output
            except Exception as retry_err:
                print(f"[PHIA:autofix-retry] Retry also failed: {retry_err}")
        
        _log_tool_call("tool_code", {"code": "..."}, error=error_str)
        return f"Error executing code: {e}"


def tavily_search_func(query: str, api_key: Optional[str] = None) -> str:
    """Performs web search using Tavily API.
    
    Args:
        query: The search query string.
        api_key: Tavily API key.
        
    Returns:
        Formatted search results as a string.
    """
    _log_tool_call("search", {"query": query})
    
    try:
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                _log_tool_call("search", {"query": query}, error="Tavily API key not found")
                return "Error: Tavily API key not found."

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        search_results = client.search(
            query=query,
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced"
        )

        formatted_results = []
        
        if search_results.get('answer'):
            formatted_results.append(f"Summary: {search_results['answer']}\n")
        
        result_count = 0
        if search_results.get('results'):
            result_count = len(search_results['results'])
            formatted_results.append("Search Results:")
            for i, result in enumerate(search_results['results'], 1):
                formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
                formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
                formatted_results.append(f"   Content: {result.get('content', 'No content')[:200]}...")
                if result.get('score'):
                    formatted_results.append(f"   Relevance Score: {result['score']:.2f}")
        
        result_text = "\n".join(formatted_results) if formatted_results else "No search results found."
        _log_tool_call("search", {"query": query}, output=f"Found {result_count} results")
        return result_text
        
    except Exception as e:
        _log_tool_call("search", {"query": query}, error=str(e))
        return f"Error performing search: {str(e)}"


def build_exemplars_from_notebooks(
    notebook_paths: Sequence[str],
    verbose: bool = False,
) -> List["react.ReActState"]:
    """Construct ReAct exemplars from notebook files.
    
    Args:
        notebook_paths: Paths to notebook files containing exemplars.
        verbose: If True, print processing information.
        
    Returns:
        List of ReActState objects.
    """
    if not _NBFORMAT_AVAILABLE:
        if verbose:
            print("nbformat not available, returning empty exemplars")
        return []
    
    if not _ONETWO_AVAILABLE:
        if verbose:
            print("onetwo not available, returning empty exemplars")
        return []
    
    exemplars = []
    
    for notebook_path in notebook_paths:
        if verbose:
            print(f"Processing: {notebook_path}")
            
        if not os.path.exists(notebook_path):
            if verbose:
                print(f"  File not found: {notebook_path}")
            continue
            
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.reads(f.read(), as_version=4)
        except Exception as e:
            if verbose:
                print(f"  Error reading notebook: {e}")
            continue
        
        cells = nb.get("cells", [])
        if not cells:
            continue
        
        # 第一个单元格是问题描述
        description_cell = cells.pop(0)
        problem_description = "".join(description_cell.get("source", [])).strip()
        problem_description = problem_description.lstrip("#").strip()
        
        planner_updates = []
        
        # 进程单元格成对( markdownthought + 代码动作)
        i = 0
        while i < len(cells):
            if (i < len(cells) - 1 and 
                cells[i].get("cell_type") == "markdown" and 
                cells[i + 1].get("cell_type") == "code"):
                
                thought = "".join(cells[i].get("source", [])).strip()
                code = "".join(cells[i + 1].get("source", [])).strip()
                code = code.replace('# @test {"skip": true}\n', "")
                
                # 确定要调用的职能
                function_name = _SEARCH_TOOL_NAME if "search" in code.lower() else _PYTHON_TOOL_NAME
                
                if function_name == _SEARCH_TOOL_NAME:
                    # 从代码中提取搜索查询，如： 搜索 ('query')
                    try:
                        args = (code.split("'")[1],)
                    except IndexError:
                        args = (code,)
                else:
                    args = (code,)
                
                # 从单元格输出中获取观察
                observation = ""
                try:
                    outputs = cells[i + 1].get("outputs", [])
                    if outputs:
                        output = outputs[0]
                        if "text" in output:
                            observation = "".join(output["text"]).strip()
                        elif "text/plain" in output.get("data", {}):
                            observation = "".join(output["data"]["text/plain"]).strip()
                except (KeyError, IndexError):
                    pass
                
                fmt = (llm_tool_use.ArgumentFormat.MARKDOWN 
                       if function_name == _PYTHON_TOOL_NAME 
                       else llm_tool_use.ArgumentFormat.PYTHON)
                
                step = react.ReActStep(
                    thought=thought,
                    is_finished=False,
                    action=llm_tool_use.FunctionCall(
                        function_name=function_name,
                        args=args,
                        kwargs={},
                    ),
                    observation=observation,
                    fmt=fmt,
                )
                planner_updates.append(step)
                i += 2
            else:
                i += 1
        
        # 以完成工具处理最后一步
        if planner_updates and not planner_updates[-1].is_finished:
            last_step = planner_updates[-1]
            if (last_step.action and 
                last_step.action.function_name == _PYTHON_TOOL_NAME):
                
                action_code = last_step.action.args[0]
                final_answer = None
                
                if "print(" in action_code:
                    for pattern in ['print("""', "print('", 'print("']:
                        if pattern in action_code:
                            try:
                                end_pattern = '""")' if pattern == 'print("""' else ("')" if pattern == "print('" else '")')
                                final_answer = action_code.split(pattern)[1].rsplit(end_pattern, 1)[0]
                                break
                            except IndexError:
                                continue
                    
                    if final_answer is None:
                        final_answer = action_code
                
                if final_answer is not None:
                    final_step = react.ReActStep(
                        thought=last_step.thought,
                        is_finished=True,
                        action=llm_tool_use.FunctionCall(
                            function_name=_FINISH_TOOL_NAME,
                            args=(final_answer,),
                            kwargs={},
                        ),
                        observation=final_answer,
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    )
                    planner_updates[-1] = final_step
        
        exemplars.append(react.ReActState(
            inputs=problem_description,
            updates=planner_updates,
        ))
    
    return exemplars


def _create_default_exemplars() -> List["react.ReActState"]:
    """Create default exemplars when notebooks are not available.
    
    Returns:
        List of hardcoded ReActState exemplars.
    """
    if not _ONETWO_AVAILABLE:
        return []
    
    exemplars = []
    
    # 实例1:睡眠分析
    exemplars.append(react.ReActState(
        inputs="How has my sleep been over the past week?",
        updates=[
            react.ReActStep(
                thought="I need to analyze the user's sleep data from the summary_df. I'll look at sleep_minutes, deep_sleep_minutes, and rem_sleep_minutes for the last 7 days.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Get last 7 days of sleep data
last_week = summary_df.tail(7)
avg_sleep = last_week['sleep_minutes'].mean()
avg_deep = last_week['deep_sleep_minutes'].mean()
avg_rem = last_week['rem_sleep_minutes'].mean()
deep_pct = avg_deep / avg_sleep * 100 if avg_sleep > 0 else 0
rem_pct = avg_rem / avg_sleep * 100 if avg_sleep > 0 else 0
print(f"Average sleep: {avg_sleep:.0f} minutes ({avg_sleep/60:.1f} hours)")
print(f"Average deep sleep: {avg_deep:.0f} min ({deep_pct:.1f}%)")
print(f"Average REM sleep: {avg_rem:.0f} min ({rem_pct:.1f}%)")""",),
                    kwargs={},
                ),
                observation="Average sleep: 420 minutes (7.0 hours)\nAverage deep sleep: 78 min (18.5%)\nAverage REM sleep: 94 min (22.3%)",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The user is getting about 7 hours of sleep with good deep and REM percentages. Let me provide a summary with health context.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("Based on your data from the past week, you're averaging 7 hours of sleep per night, which meets the recommended 7-9 hours for adults. Your deep sleep (78 min, 18.5%) and REM sleep (94 min, 22.3%) are within healthy ranges. Deep sleep is important for physical recovery, while REM sleep supports cognitive function and memory consolidation.",),
                    kwargs={},
                ),
                observation="Based on your data from the past week, you're averaging 7 hours of sleep per night, which meets the recommended 7-9 hours for adults. Your deep sleep (78 min, 18.5%) and REM sleep (94 min, 22.3%) are within healthy ranges. Deep sleep is important for physical recovery, while REM sleep supports cognitive function and memory consolidation.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    # 实例2:活动分析与搜索
    exemplars.append(react.ReActState(
        inputs="What's a good resting heart rate for someone my age?",
        updates=[
            react.ReActStep(
                thought="I should first check the user's age from their profile, then search for information about healthy resting heart rate ranges.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Get user's age and recent resting heart rate
age = profile['age'].values[0]
recent_rhr = summary_df['resting_heart_rate'].tail(7).mean()
print(f"User age: {age}")
print(f"Recent average RHR: {recent_rhr:.0f} bpm")""",),
                    kwargs={},
                ),
                observation="User age: 34\nRecent average RHR: 68 bpm",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The user is 34 years old with an RHR of 68 bpm. Let me search for healthy RHR ranges for this age group.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_SEARCH_TOOL_NAME,
                    args=("healthy resting heart rate range for adults age 30-40",),
                    kwargs={},
                ),
                observation="Search Results:\n1. Normal resting heart rate for adults is 60-100 bpm. Athletes may have RHR as low as 40 bpm. A lower RHR generally indicates better cardiovascular fitness.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                thought="I have all the information needed to provide a comprehensive answer about the user's resting heart rate.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("Your resting heart rate of 68 bpm is within the healthy range of 60-100 bpm for adults. At 34 years old, this indicates good cardiovascular health. A lower RHR generally suggests better fitness - athletes often have RHR in the 40-60 bpm range. To potentially lower your RHR, consider regular aerobic exercise, stress management, and adequate sleep.",),
                    kwargs={},
                ),
                observation="Your resting heart rate of 68 bpm is within the healthy range of 60-100 bpm for adults. At 34 years old, this indicates good cardiovascular health. A lower RHR generally suggests better fitness - athletes often have RHR in the 40-60 bpm range. To potentially lower your RHR, consider regular aerobic exercise, stress management, and adequate sleep.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    # 例 3:情感/痛苦查询——ALWAYS先检查数据
    exemplars.append(react.ReActState(
        inputs="I'm really worried about my health lately, been feeling tired and my heart races sometimes",
        updates=[
            react.ReActStep(
                thought="The user is expressing health concerns. I should FIRST check their actual data to provide personalized reassurance, then supplement with general advice.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Check sleep, heart rate, and activity data
avg_sleep = summary_df['sleep_minutes'].mean()
avg_rhr = summary_df['resting_heart_rate'].mean()
avg_hrv = summary_df['heart_rate_variability'].mean()
avg_steps = summary_df['steps'].mean()
avg_deep = summary_df['deep_sleep_minutes'].mean()
deep_pct = avg_deep / avg_sleep * 100 if avg_sleep > 0 else 0
print(f"Avg sleep: {avg_sleep:.0f} min ({avg_sleep/60:.1f} hrs)")
print(f"Avg deep sleep: {avg_deep:.0f} min ({deep_pct:.1f}%)")
print(f"Avg resting HR: {avg_rhr:.1f} bpm")
print(f"Avg HRV: {avg_hrv:.1f}")
print(f"Avg daily steps: {avg_steps:.0f}")""",),
                    kwargs={},
                ),
                observation="Avg sleep: 418 min (7.0 hrs)\nAvg deep sleep: 72 min (17.2%)\nAvg resting HR: 62.2 bpm\nAvg HRV: 45.0\nAvg daily steps: 8234",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The data shows healthy metrics. Let me search for information about fatigue and heart racing to give informed advice.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_SEARCH_TOOL_NAME,
                    args=("causes of fatigue and heart racing in adults",),
                    kwargs={},
                ),
                observation="Search Results:\n1. Common causes include stress, caffeine, poor sleep quality, dehydration, and anxiety. Resting heart rate under 100 bpm is normal.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                thought="I have both the user's data and general health context. I can now provide a reassuring, data-grounded response.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("I understand you're feeling worried, and I want to help put things in perspective with your actual data. Your resting heart rate averages 62.2 bpm, which is excellent — well within the healthy 60-100 bpm range. Your HRV of 45 also looks healthy. You're averaging about 7 hours of sleep with 17% deep sleep, which is reasonable but could be improved.\n\nThe tiredness and occasional racing heart could be related to several factors: stress, caffeine intake, or inconsistent sleep timing. Here are some steps that could help:\n\n1. Try to maintain a consistent sleep schedule — going to bed and waking up at the same time daily.\n2. If you drink coffee, try reducing intake or cutting it off by early afternoon.\n3. Stay hydrated throughout the day.\n4. Your step count of 8,234 shows good activity — keep it up!\n\nThat said, if the heart racing episodes are frequent or severe, please consult a doctor to rule out any underlying conditions. Your overall data looks healthy, which is reassuring.",),
                    kwargs={},
                ),
                observation="I understand you're feeling worried, and I want to help put things in perspective with your actual data. Your resting heart rate averages 62.2 bpm, which is excellent — well within the healthy 60-100 bpm range. Your HRV of 45 also looks healthy. You're averaging about 7 hours of sleep with 17% deep sleep, which is reasonable but could be improved.\n\nThe tiredness and occasional racing heart could be related to several factors: stress, caffeine intake, or inconsistent sleep timing. Here are some steps that could help:\n\n1. Try to maintain a consistent sleep schedule — going to bed and waking up at the same time daily.\n2. If you drink coffee, try reducing intake or cutting it off by early afternoon.\n3. Stay hydrated throughout the day.\n4. Your step count of 8,234 shows good activity — keep it up!\n\nThat said, if the heart racing episodes are frequent or severe, please consult a doctor to rule out any underlying conditions. Your overall data looks healthy, which is reassuring.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    return exemplars


# =============================================================================
# 请检查access-date=中的日期值 (帮助)
# =============================================================================

def is_onetwo_available() -> bool:
    """Check if onetwo is available for PHIA baseline."""
    return _ONETWO_AVAILABLE


def get_onetwo_import_error() -> Optional[str]:
    """Get the onetwo import error message if any."""
    return _ONETWO_IMPORT_ERROR


# =============================================================================
# PHIA 基线 类
# =============================================================================

if _ONETWO_AVAILABLE:
    
    class PHIABaseline:
        """PHIA Single-Agent Baseline.
        
        Implements the original PHIA architecture: a single ReAct agent that
        handles all health queries using Python code execution and web search.
        
        Key differences from PHA:
        - Single generalist agent instead of 3 specialists
        - No orchestration or routing logic
        - Combined prompt covering all health domains
        """
        
        def __init__(
            self,
            tavily_api_key: Optional[str] = None,
            few_shot_dir: Optional[str] = None,
            debug_verbose: bool = False,
        ):
            """Initialize the PHIA baseline.
            
            Args:
                tavily_api_key: Tavily API key for web search.
                few_shot_dir: Directory containing few-shot notebook examples.
                debug_verbose: If True, print debug information.
            """
            self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
            self.few_shot_dir = few_shot_dir
            self.debug_verbose = debug_verbose
            
            if debug_verbose:
                print(f"[PHIA Baseline] Version {__version__}")
            
            # 代理状态
            self._agent = None
            self._gemini_api_key = None
            self._sandbox_globals = None
            
            # 数据
            self.summary_df: Optional[pd.DataFrame] = None
            self.activities_df: Optional[pd.DataFrame] = None
            self.profile_df: Optional[pd.DataFrame] = None
            
            # 对话历史
            self.conversation_history: list = []
            self.last_response: str = ""
        
        def configure(
            self,
            api_key: Optional[str] = None,
            provider: str = "gemini",
            model_name: Optional[str] = None,
            # 遗留参数
            gemini_api_key: Optional[str] = None,
        ) -> "PHIABaseline":
            """Configure the PHIA baseline with API credentials.
            
            Args:
                api_key: API key for the selected provider.
                provider: "gemini" or "openai".
                model_name: Model to use. Defaults vary by provider:
                    - Gemini: "models/gemini-2.5-pro"
                    - OpenAI: "gpt-4o"
                gemini_api_key: DEPRECATED - use api_key instead.
                
            Returns:
                Self for method chaining.
            """
            # 处理遗留参数
            if gemini_api_key and not api_key:
                api_key = gemini_api_key
                provider = "gemini"
            
            if not api_key:
                raise ValueError("API key is required")
            
            self._api_key = api_key
            self._provider = provider.lower()
            
            # 基于提供者设置默认模式
            if model_name is None:
                if self._provider == "gemini":
                    model_name = "models/gemini-3.1-flash-lite-preview"
                elif self._provider == "openai":
                    model_name = "gpt-4o"
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            
            self._model_name = model_name
            
            # 使用寄存器( ) 模式配置一个 2 个后端
            if self._provider == "gemini":
                # 需要时添加模型/前缀
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                if not _ONETWO_GENAI_AVAILABLE:
                    raise ImportError(
                        "onetwo's GoogleGenAIAPI backend not available. "
                        "Update onetwo: pip install -U git+https://github.com/google-deepmind/onetwo"
                    )
                
                print(f"[PHIA:configure] Registering GoogleGenAIAPI backend: model={model_name}, temp=0.0")
                llm_engine = google_genai_api.GoogleGenAIAPI(
                    api_key=api_key,
                    generate_model_name=model_name,
                    # Pin 聊天模式也——OneTwo 验证两者;默认为
                    # `models/gemini-2.5-flash` 并不是所有的键有。
                    chat_model_name=model_name,
                    temperature=0.0,
                    max_tokens=8192,
                    threadpool_size=1,
                )
                llm_engine.register()
            elif self._provider == "openai":
                if not _ONETWO_OPENAI_AVAILABLE:
                    raise ValueError(
                        "OneTwo OpenAI backend not available. "
                        "Please use Gemini or update onetwo."
                    )
                # 原因/ GPT-5 模型只接受温度= 1.0 (其
                # API 默认值。我们用在其他地方的0.0最接近的模拟值。
                openai_temp = 1.0 if model_name.startswith(("o1", "o3", "o4", "gpt-5")) else 0.0
                print(f"[PHIA:configure] Registering OpenAI backend: model={model_name}, temp={openai_temp}")
                llm_engine = openai_api.OpenAIAPI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=openai_temp,
                    batch_size=1,
                )
                llm_engine.register()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            return self
        
        def load_data(
            self,
            summary_df: pd.DataFrame,
            activities_df: pd.DataFrame,
            profile_df: pd.DataFrame,
        ) -> "PHIABaseline":
            """Load user health data.
            
            Args:
                summary_df: Daily summary DataFrame.
                activities_df: Activities DataFrame.
                profile_df: User profile DataFrame.
                
            Returns:
                Self for method chaining.
            """
            self.summary_df = summary_df
            self.activities_df = activities_df
            self.profile_df = profile_df
            
            # 设置用于代码执行的沙盒全球
            self._sandbox_globals = {
                "pd": pd,
                "np": np,
                "summary_df": summary_df,
                "activities_df": activities_df,
                "profile": profile_df,
            }
            
            return self
        
        def _build_agent(self) -> None:
            """Build the ReAct agent with tools and exemplars."""
            if self._sandbox_globals is None:
                raise RuntimeError("Data not loaded. Call load_data() first.")
            
            # 创建工具
            python_tool = llm_tool_use.Tool(
                name=_PYTHON_TOOL_NAME,
                function=lambda code: simple_python_executor(code, self._sandbox_globals),
                description='Python interpreter. Executes code on pandas DataFrames (summary_df, activities_df, profile).',
            )
            
            search_tool = llm_tool_use.Tool(
                name=_SEARCH_TOOL_NAME,
                function=lambda query: tavily_search_func(query, api_key=self.tavily_api_key),
                description='Web search engine for finding health, wellness, and fitness information.',
            )
            
            finish_tool = llm_tool_use.Tool(
                name=_FINISH_TOOL_NAME,
                function=lambda x: x,
                description='Returns the final answer.',
            )
            
            agent_tools = [python_tool, search_tool, finish_tool]
            
            # 装入示例
            exemplars = []
            if self.few_shot_dir and os.path.isdir(self.few_shot_dir):
                notebook_files = [
                    os.path.join(self.few_shot_dir, f)
                    for f in os.listdir(self.few_shot_dir)
                    if f.endswith('.ipynb')
                ]
                exemplars = build_exemplars_from_notebooks(
                    notebook_files,
                    verbose=self.debug_verbose,
                )
            
            # 如果未装入，则返回到默认示例
            if not exemplars:
                if self.debug_verbose:
                    print("Using default hardcoded exemplars")
                exemplars = _create_default_exemplars()
            
            # 创建代理
            self._agent = react.ReActAgent(
                exemplars=exemplars,
                environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
                    tools=agent_tools,
                ),
                max_steps=10,  # 限制以预防环路失控
                stop_prefix="",
            )
            
            # 设置提示
            self._agent.prompt = react.ReActPromptJ2(
                text=PHIA_PREAMBLE + PHIA_REACT_PROMPT_TEXT
            )
        
        def respond(self, user_message: str) -> str:
            """Process a user message and return a response.
            
            Args:
                user_message: The user's query.
                
            Returns:
                The agent's response (cleaned of internal labels).
            """
            # 重置此请求的数据
            if _STATS_AVAILABLE:
                request_stats.reset()
            
            # 总是记录用于诊断的密钥信息
            from pha.streaming import emit_step
            print(f"\n{'─' * 60}")
            print(f"[PHIA] Model: {getattr(self, '_model_name', 'unknown')}")
            print(f"[PHIA] Provider: {getattr(self, '_provider', 'unknown')}")
            print(f"[PHIA] Query: {user_message[:80]}...")
            print(f"{'─' * 60}")
            
            # 为每个查询重建代理以避免状态积累
            # (遵循代理人无国籍的原PHIA模式).
            self._build_agent()
            
            if self.debug_verbose:
                print(f"[PHIA Baseline] Processing: {user_message[:50]}...")
            
            # 添加问题前缀以更好地回答
            full_query = QUESTION_PREFIX + user_message
            
            # 使用 ot.run () 运行代理( 遵循原始 PHIA 模式)
            try:
                emit_step("PHIA", "正在运行 ReAct Agent…")
                final_answer, final_state = ot.run(
                    self._agent(inputs=full_query, return_final_state=True)
                )
                
                # 总是记录返回的 ot.run ()
                print(f"\n[PHIA] ot.run() completed")
                print(f"[PHIA] final_answer type: {type(final_answer).__name__}")
                print(f"[PHIA] final_answer: {repr(str(final_answer))[:300]}")
                if hasattr(final_state, 'updates') and final_state.updates:
                    n_steps = len(final_state.updates)
                    print(f"[PHIA] final_state has {n_steps} steps")
                    emit_step("PHIA", f"ReAct 已完成 {n_steps} 个步骤")
                    for i, step in enumerate(final_state.updates):
                        thought_preview = repr(str(step.thought))[:120] if step.thought else 'None'
                        action_preview = repr(str(step.action))[:120] if step.action else 'None'
                        obs_preview = repr(str(step.observation))[:120] if step.observation else 'None'
                        print(f"[PHIA]   Step {i}: finished={step.is_finished}")
                        print(f"[PHIA]     thought: {thought_preview}")
                        print(f"[PHIA]     action:  {action_preview}")
                        print(f"[PHIA]     observe: {obs_preview}")
                else:
                    print(f"[PHIA] final_state: {repr(final_state)[:200]}")
                
                # 提取最后的答案
                emit_step("PHIA", "正在提取并整理回复…")
                response = self._extract_answer(final_answer, final_state)
                
                # 如果提取失败，请尝试从 OneTwo 解析错误
                _FALLBACK_INDICATORS = [
                    "Could not extract a valid answer",
                    "Response contained system prompt leakage",
                ]
                if any(indicator in response for indicator in _FALLBACK_INDICATORS):
                    emit_step("PHIA", "未能提取回复，正在尝试恢复…")
                    recovered = self._try_recover_from_parse_error(
                        final_answer, final_state, user_message
                    )
                    if not recovered:
                        # 尝试循环恢复( 模型重复了相同的工具调用)
                        recovered = self._try_recover_from_loop(
                            final_state, user_message
                        )
                    if recovered:
                        response = recovered
                
                # 检查截断的响应( 常见于 2.5 Pro)
                # 模型产生长期的想法 吃到象征性的预算，
                # 离开[结 文本断绝中刑
                if self._is_truncated_response(response):
                    emit_step("PHIA", "回复可能不完整，正在重新生成…")
                    print(f"[PHIA] Truncated response detected ({len(response)} chars): {repr(response[:100])}")
                    direct = self._generate_direct_answer(user_message)
                    if direct:
                        response = direct
                
                # 检查低质量回复:模糊的序言或原始数据堆放
                # 通过了综合检查但实际上没有帮助
                if not self._is_substantive_response(response, final_state):
                    emit_step("PHIA", "回复信息不足，正在生成更完整的答案…")
                    print(f"[PHIA] Non-substantive response detected ({len(response)} chars): {repr(response[:100])}")
                    # 从 ReAct 步骤中收集 ALL 语境:工具输出，
                    # 搜索结果，以及模型本身对想法的分析
                    full_context = self._collect_full_step_context(final_state)
                    enriched = self._generate_enriched_answer(
                        user_message, full_context
                    )
                    if enriched:
                        response = enriched
                    else:
                        direct = self._generate_direct_answer(user_message)
                        if direct:
                            response = direct
                    
            except Exception as e:
                print(f"\n[PHIA] *** EXCEPTION in ot.run(): {e}")
                import traceback
                traceback.print_exc()
                emit_step("PHIA", f"处理失败： {type(e).__name__}: {str(e)[:100]}")
                response = f"I encountered an error processing your question: {str(e)}"
            
            print(f"\n[PHIA] Final response ({len(response)} chars): {response[:200]}...")
            
            self.last_response = response
            
            # 更新对话历史
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
            })
            
            # 打印请求摘要
            if _STATS_AVAILABLE:
                request_stats.print_summary()
            
            # 返回用户前的清洁内部代理标签
            return clean_agent_response(response)
        
        def _extract_answer(self, final_answer: Any, final_state: Any) -> str:
            """Extract a clean answer from the agent's response.
            
            Args:
                final_answer: The final answer returned by ot.run().
                final_state: The final state of the agent.
                
            Returns:
                A clean response string.
            """
            # 帮助检查文本是否为原始搜索结果
            def is_raw_search_output(text: str) -> bool:
                text_lower = text.lower()
                url_count = text_lower.count('url:')
                relevance_count = text_lower.count('relevance score:')
                has_search_header = 'search results:' in text_lower
                return url_count >= 2 or relevance_count >= 2 or (has_search_header and url_count >= 1)
            
            print(f"\n[PHIA:extract] === Starting _extract_answer ===")
            
            # * 早期发现 API 错误( 速率限制、 认证等)***
            # 这些显示为# ERROR# 字符串在 final_answer 或观察中
            error_sources = []
            if final_answer:
                error_sources.append(str(final_answer))
            if hasattr(final_state, 'updates') and final_state.updates:
                for step in final_state.updates:
                    if step.observation:
                        error_sources.append(str(step.observation))
            
            for src in error_sources:
                if '#ERROR#' in src:
                    # 提取实际错误消息
                    api_error_patterns = {
                        '429': 'Rate limit exceeded',
                        'Resource exhausted': 'Rate limit exceeded', 
                        'quota': 'API quota exceeded',
                        '401': 'Authentication error',
                        '403': 'Permission denied',
                        '500': 'API server error',
                        '503': 'API service unavailable',
                    }
                    for pattern, label in api_error_patterns.items():
                        if pattern.lower() in src.lower():
                            print(f"[PHIA:extract] *** API ERROR detected: {label} ***")
                            return (
                                f"⚠️ **{label}** — the model API returned an error.\n\n"
                                f"```\n{src[len('#ERROR#:'):].strip()[:300]}\n```\n\n"
                                f"Please wait a moment and try again, or switch to a different model."
                            )
            
            # 如果有直接的最后答案请检查是否完整
            if final_answer:
                answer = str(final_answer)
                print(f"[PHIA:extract] Step 1 - Checking final_answer ({len(answer)} chars): {repr(answer[:200])}")
                cleaned = self._clean_react_format(answer)
                print(f"[PHIA:extract]   After _clean_react_format ({len(cleaned)} chars): {repr(cleaned[:200])}")
                # 如果为原始搜索输出，请跳过
                if is_raw_search_output(cleaned):
                    print(f"[PHIA:extract]   REJECTED: raw search output")
                elif cleaned and self._is_complete_response(cleaned):
                    print(f"[PHIA:extract]   ACCEPTED as complete response")
                    return cleaned
                else:
                    print(f"[PHIA:extract]   REJECTED: _is_complete_response returned False (empty={not cleaned}, len={len(cleaned)})")
                    if cleaned:
                        self._debug_completeness_check(cleaned)
                    # 如果文本是长的(> 500 字符),模型可能写
                    # 完整答案，但前缀它与规划句。
                    # 尝试剥除规划序言并重新检查。
                    if cleaned and len(cleaned) > 500:
                        salvaged = self._strip_planning_preamble(cleaned)
                        if salvaged and self._is_complete_response(salvaged):
                            print(f"[PHIA:extract]   SALVAGED after stripping planning preamble ({len(salvaged)} chars)")
                            return salvaged
            else:
                print(f"[PHIA:extract] Step 1 - final_answer is falsy: {repr(final_answer)}")
            
            # 尝试从最后状态提取
            if hasattr(final_state, 'updates') and final_state.updates:
                # 查找完成步骤(这些步骤有实际最后的答案)
                print(f"[PHIA:extract] Step 2 - Checking finished steps...")
                found_finished = False
                for step in reversed(final_state.updates):
                    if step.is_finished and step.observation:
                        found_finished = True
                        obs = str(step.observation)
                        print(f"[PHIA:extract]   Finished step obs ({len(obs)} chars): {repr(obs[:200])}")
                        if is_raw_search_output(obs):
                            print(f"[PHIA:extract]   REJECTED: raw search output")
                            continue
                        cleaned = self._clean_react_format(obs)
                        print(f"[PHIA:extract]   After clean ({len(cleaned)} chars): {repr(cleaned[:200])}")
                        if cleaned and self._is_complete_response(cleaned):
                            print(f"[PHIA:extract]   ACCEPTED finished step")
                            return cleaned
                        else:
                            print(f"[PHIA:extract]   REJECTED: _is_complete_response False")
                            if cleaned:
                                self._debug_completeness_check(cleaned)
                            # 与 Step 1 相同的打捞逻辑
                            if cleaned and len(cleaned) > 500:
                                salvaged = self._strip_planning_preamble(cleaned)
                                if salvaged and self._is_complete_response(salvaged):
                                    print(f"[PHIA:extract]   SALVAGED finished step after stripping preamble ({len(salvaged)} chars)")
                                    return salvaged
                if not found_finished:
                    print(f"[PHIA:extract]   No finished steps found")
                
                # 如果没有完成步骤，请寻找最后的实质性观察
                print(f"[PHIA:extract] Step 3 - Checking all observations...")
                for i, step in enumerate(reversed(final_state.updates)):
                    if step.observation:
                        obs = str(step.observation)
                        obs_stripped = obs.strip()
                        
                        print(f"[PHIA:extract]   Obs[rev {i}] ({len(obs)} chars): {repr(obs[:150])}")
                        
                        if is_raw_search_output(obs):
                            print(f"[PHIA:extract]     SKIP: raw search output")
                            continue
                        
                        has_react_markers = any(m in obs for m in ['[Act]:', '[Thought]:', '[Observe]:'])
                        obs_no_backticks = obs_stripped.replace('`', '')
                        is_raw_tool_call = obs_no_backticks.startswith('tool_code(') or obs_no_backticks.startswith('search(')
                        is_onetwo_error = '#ERROR#' in obs or 'Invalid syntax for positional arguments' in obs
                        is_raw_pandas = 'dtype:' in obs and 'Name:' in obs  # 检查是否原样返回 pandas Series。
                        is_raw_code_output = obs_stripped.startswith('(np.') or obs_stripped.startswith('np.')  # 原始数字输出
                        # 检测代码内容( 模式倾销工具调用为文本)
                        code_indicators = ['summary_df[', 'activities_df[', 'profile[',
                                           '.mean()', '.sum()', '.max()', '.min()',
                                           'print(f"', "print(f'", '\\n']
                        is_code_content = sum(1 for ci in code_indicators if ci in obs) >= 3
                        
                        if len(obs) <= 100:
                            print(f"[PHIA:extract]     SKIP: too short ({len(obs)} <= 100)")
                        elif 'Error' in obs:
                            print(f"[PHIA:extract]     SKIP: contains 'Error'")
                        elif has_react_markers:
                            print(f"[PHIA:extract]     SKIP: has ReAct markers")
                        elif is_raw_tool_call:
                            print(f"[PHIA:extract]     SKIP: raw tool call")
                        elif is_onetwo_error:
                            print(f"[PHIA:extract]     SKIP: OneTwo error")
                        elif is_raw_pandas:
                            print(f"[PHIA:extract]     SKIP: raw pandas output")
                        elif is_raw_code_output:
                            print(f"[PHIA:extract]     SKIP: raw code output")
                        elif is_code_content:
                            print(f"[PHIA:extract]     SKIP: code content (not natural language)")
                        elif self._is_complete_response(obs):
                            print(f"[PHIA:extract]     ACCEPTED as substantial observation")
                            return f"Based on my analysis of your data:\n\n{obs}"
                        else:
                            print(f"[PHIA:extract]     SKIP: _is_complete_response False")
                            self._debug_completeness_check(obs)
                
                # 尝试使用最后一个想法，如果它有有用的内容
                print(f"[PHIA:extract] Step 4 - Checking thoughts...")
                for step in reversed(final_state.updates):
                    if step.thought:
                        thought = str(step.thought)
                        thought_lower = thought.lower()
                        planning_indicators = ['i need to', 'i need the', 'then use search', 'then use tool_code', 'from summary_df']
                        if any(p in thought_lower for p in planning_indicators):
                            print(f"[PHIA:extract]   Thought SKIP (planning): {repr(thought[:100])}")
                            continue
                        if any(word in thought_lower for word in ['average', 'found', 'shows', 'indicates', 'data']):
                            cleaned = self._clean_react_format(thought)
                            if cleaned and len(cleaned) > 50 and self._is_complete_response(cleaned):
                                print(f"[PHIA:extract]   ACCEPTED thought: {repr(cleaned[:100])}")
                                return cleaned
                            else:
                                print(f"[PHIA:extract]   Thought REJECTED: {repr(thought[:100])}")
                        else:
                            print(f"[PHIA:extract]   Thought SKIP (no analytical words): {repr(thought[:100])}")
            else:
                print(f"[PHIA:extract] Step 2 - No updates in final_state")
            
            # 退缩——表面什么出了问题
            print(f"[PHIA:extract] *** ALL EXTRACTION ATTEMPTS FAILED - returning diagnostic error ***")
            
            # 收集诊断信息
            n_steps = 0
            step_details = []
            if final_state and hasattr(final_state, 'updates') and final_state.updates:
                n_steps = len(final_state.updates)
                for i, step in enumerate(final_state.updates):
                    finished = getattr(step, 'is_finished', '?')
                    obs_preview = str(getattr(step, 'observation', ''))[:100]
                    step_details.append(f"  Step {i}: finished={finished}, obs=`{obs_preview}`")
            
            answer_preview = str(final_answer)[:300] if final_answer else "None"
            steps_str = "\n".join(step_details) if step_details else "  (no steps)"
            
            return (
                f"⚠️ **Could not extract a valid answer** from the agent's output.\n\n"
                f"**ReAct steps executed:** {n_steps}\n"
                f"```\n{steps_str}\n```\n\n"
                f"**Raw final_answer preview:**\n```\n{answer_preview}\n```\n\n"
                f"This usually means the model's output format didn't match expectations. "
                f"Try rephrasing your question or switching models."
            )
        
        def _try_recover_from_parse_error(
            self, final_answer: Any, final_state: Any, user_message: str
        ) -> Optional[str]:
            """Recover when OneTwo can't parse the model's tool calls.
            
            Gemini Flash (and sometimes other models) outputs tool calls that
            OneTwo's parser can't handle. This commonly happens when:
            - tool_code() calls span multiple lines
            - search() calls have formatting OneTwo doesn't expect
            
            Recovery strategy:
            1. Detect the parse error in step observations
            2. Collect successful intermediate results from earlier steps
            3. Extract code/search from ALL sources and execute
            4. Generate enriched answer using all context (intermediate + new)
            5. Fall back to enriched answer with just intermediate results
            """
            # 步骤 1 : 检查失败是否是一个工具调用解析错误
            has_parse_error = False
            if hasattr(final_state, 'updates'):
                for step in final_state.updates:
                    obs = str(step.observation) if step.observation else ''
                    # OneTwo 发出不同的错误模式：
                    # -##ERROR:无效的语法 呼叫：tool_code(").
                    # - "# ERROR#:没有找到\ [Act\]:?或\[结束]:?"
                    if '#ERROR#' in obs and ('Invalid syntax' in obs or 
                                              'find' in obs.lower()):
                        has_parse_error = True
                        break
            
            if not has_parse_error:
                return None
            
            print(f"\n[PHIA:recover] Detected OneTwo parse error — attempting manual recovery")
            
            # 步骤2:从先前的步骤中收集成功的中间结果
            intermediate_results = []
            if hasattr(final_state, 'updates'):
                for i, step in enumerate(final_state.updates):
                    if step.observation:
                        obs = str(step.observation).strip()
                        if '#ERROR#' not in obs and 'Error executing code' not in obs and len(obs) > 10:
                            intermediate_results.append(obs)
                            print(f"[PHIA:recover] Collected intermediate result from step {i}: {obs[:100]}")
            
            # 步骤 3:搜索 ALL 代码查询或搜索查询的来源
            sources_to_search = []
            # final_answer
            if final_answer:
                sources_to_search.append(("final_answer", str(final_answer)))
            # 步骤和想法(通常使用代码的地方)
            if hasattr(final_state, 'updates'):
                for i, step in enumerate(final_state.updates):
                    if step.action:
                        sources_to_search.append((f"step[{i}].action", str(step.action)))
                    if step.thought:
                        sources_to_search.append((f"step[{i}].thought", str(step.thought)))
                    if step.observation:
                        sources_to_search.append((f"step[{i}].observation", str(step.observation)))
            
            # 3a: 尝试提取和执行代码
            code = None
            for source_name, source_text in sources_to_search:
                code = self._extract_code_from_raw_output(source_text)
                if code:
                    print(f"[PHIA:recover] Found code in {source_name} ({len(code)} chars)")
                    break
            
            if code:
                print(f"[PHIA:recover] Extracted code:\n{code[:300]}")
                
                if _STATS_AVAILABLE:
                    request_stats.record_tool("tool_code (recovered)")
                result = simple_python_executor(code, self._sandbox_globals)
                if 'Error executing code' not in result:
                    print(f"[PHIA:recover] Code executed successfully: {result[:300]}")
                    summary = self._generate_summary(user_message, result, code_context=code)
                    if summary:
                        return summary
                    return f"Based on my analysis of your data:\n\n{result}"
                else:
                    print(f"[PHIA:recover] Code execution failed: {result[:200]}")
            
            # 3b: 尝试提取并执行搜索查询
            search_result = None
            search_query = self._extract_search_from_raw_output(sources_to_search)
            if search_query:
                print(f"[PHIA:recover] Found search query: {search_query}")
                try:
                    search_result = tavily_search_func(search_query, api_key=self.tavily_api_key)
                    if search_result and 'Error' not in search_result:
                        intermediate_results.append(f"Search results for '{search_query}':\n{search_result}")
                        print(f"[PHIA:recover] Search executed successfully ({len(search_result)} chars)")
                    else:
                        print(f"[PHIA:recover] Search returned error: {str(search_result)[:100]}")
                except Exception as search_err:
                    print(f"[PHIA:recover] Search execution failed: {search_err}")
            
            # 步骤 4: 生成所有收集的上下文的丰富答案
            if intermediate_results:
                context = "\n\n".join(intermediate_results)
                print(f"[PHIA:recover] Generating enriched answer with {len(intermediate_results)} intermediate results")
                enriched = self._generate_enriched_answer(user_message, context)
                if enriched:
                    return enriched
            
            # 步骤 5:倒回直接 LLM 回答
            print(f"[PHIA:recover] Falling back to direct LLM answer")
            return self._generate_direct_answer(user_message)
        
        def _try_recover_from_loop(
            self, final_state: Any, user_message: str
        ) -> Optional[str]:
            """Recover when the model gets stuck in a tool call loop.
            
            Some models (especially GPT-4o-mini) repeat the same tool_code call
            many times without progressing to [Finish]. If we detect this pattern,
            we can use the first successful observation to generate an answer.
            """
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return None
            
            # 收集非error 观察及其相关代码
            observations = []
            obs_to_code = {}  # 地图观察 – 生成它的代码
            for step in final_state.updates:
                if step.observation:
                    obs = str(step.observation)
                    if '#ERROR#' not in obs and 'Error executing code' not in obs:
                        observations.append(obs)
                        # 尝试捕捉生成此观察的代码
                        if step.action and obs not in obs_to_code:
                            action_str = str(step.action)
                            if 'tool_code' in action_str:
                                import re
                                code_match = re.search(r"args=\(['\"](.+?)['\"],?\)", action_str, re.DOTALL)
                                if code_match:
                                    obs_to_code[obs] = code_match.group(1)
            
            if len(observations) < 2:
                return None
            
            # 检查同一观测是否重复 3+次( 探测到的跳动)
            from collections import Counter
            obs_counts = Counter(observations)
            most_common_obs, count = obs_counts.most_common(1)[0]
            
            if count < 3:
                return None
            
            print(f"\n[PHIA:recover] Detected tool call loop: same observation repeated {count}x")
            print(f"[PHIA:recover] Using repeated observation: {most_common_obs[:200]}")
            
            # 收集 ALL 代码上下文，从步骤中进行更好的解释
            code_context, _ = self._collect_step_context(final_state)
            
            # 使用此观察来生成直接的答案
            summary = self._generate_summary(
                user_message, most_common_obs, code_context=code_context
            )
            if summary:
                return summary
            
            # 返回到直接 LLM 回答
            return self._generate_direct_answer(user_message)
        
        def _generate_summary(self, user_message: str, code_result: str, code_context: str = "") -> Optional[str]:
            """Generate a user-friendly summary of code execution results."""
            # 构建一个包含代码上下文的提示，因此是 LLM
            # 知道每个数字意味着什么(预防幻觉标签)
            code_section = ""
            if code_context:
                code_section = (
                    f'\nThe code that produced these results was:\n'
                    f'```python\n{code_context[:1500]}\n```\n\n'
                    f'Use the variable names and print labels from the code above '
                    f'to correctly identify what each number represents.\n'
                )
            
            prompt = (
                f'The user asked: "{user_message}"\n\n'
                f'I ran an analysis on their Fitbit health data and got these results:\n'
                f'{code_result}\n'
                f'{code_section}\n'
                f'Please provide a helpful, well-formatted response to the user\'s '
                f'question based on these results. Include context about what the '
                f'numbers mean for their health. Be conversational and supportive. '
                f'无需提及工具、代码、DataFrame 或其他实现细节。使用用户提问的语言作答。'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] LLM summary generated ({len(result)} chars)")
            return result
        
        def _generate_direct_answer(self, user_message: str) -> Optional[str]:
            """Generate a direct answer using the LLM when code recovery fails.
            
            Uses the data summary (already available) to answer the question
            without going through ReAct code execution.
            """
            # 从沙盒 Globals 中获取数据背景
            data_context = ""
            if hasattr(self, 'summary_df') and self.summary_df is not None:
                data_context = self.summary_df.tail(14).to_string()
            elif self._sandbox_globals and 'summary_df' in self._sandbox_globals:
                data_context = self._sandbox_globals['summary_df'].tail(14).to_string()
            
            if not data_context:
                return None
            
            # 如果有配置信息，则添加此信息
            profile_context = ""
            if hasattr(self, 'profile_df') and self.profile_df is not None:
                profile_context = f"\n\nUser profile:\n{self.profile_df.to_string()}"
            
            prompt = (
                f'You are a health data analyst. The user asked: "{user_message}"\n\n'
                f'Here is a summary of their recent Fitbit health data:\n'
                f'{data_context[:3000]}{profile_context}\n\n'
                f'Please provide a helpful, well-formatted response based on this data. '
                f'Reference specific numbers from the data. Be conversational and supportive. '
                f'如果数据没有提供相关信息，请如实说明。使用用户提问的语言作答。'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] Direct LLM answer generated ({len(result)} chars)")
            return result
        
        def _call_llm_direct(self, prompt: str) -> Optional[str]:
            """Make a direct LLM call for recovery, routing to correct provider."""
            try:
                if self._provider == "openai":
                    return self._call_openai_direct(prompt)
                else:
                    return self._call_gemini_direct(prompt)
            except Exception as e:
                print(f"[PHIA:recover] Direct LLM call failed ({self._provider}): {e}")
                return None
        
        def _call_gemini_direct(self, prompt: str) -> Optional[str]:
            """Direct Gemini API call for recovery."""
            from google import genai
            from google.genai import types as genai_types
            
            client = genai.Client(api_key=self._api_key)
            model = getattr(self, '_model_name', 'models/gemini-3.1-flash-lite-preview')
            if not model.startswith('models/'):
                model = f'models/{model}'
            
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
            )
            if response and response.text:
                return response.text.strip()
            return None
        
        def _call_openai_direct(self, prompt: str) -> Optional[str]:
            """Direct OpenAI API call for recovery."""
            try:
                from openai import OpenAI
            except ImportError:
                print("[PHIA:recover] openai package not available for recovery")
                return None
            
            client = OpenAI(api_key=self._api_key)
            model = getattr(self, '_model_name', 'gpt-4o')
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return None
        
        def _extract_code_from_raw_output(self, raw_text: str) -> Optional[str]:
            """Extract Python code from raw LLM output that OneTwo couldn't parse.
            
            Handles the common Flash format:
                [Act]: tool_code(
                  code_line_1
                  code_line_2
                )
            
            Uses balanced parenthesis tracking to find the full code block.
            """
            import re
            
            # 查找 tool_code (文本中)
            idx = raw_text.find('tool_code(')
            if idx == -1:
                return None
            
            # 音轨平衡括号以找到匹配的关闭
            start = idx + len('tool_code')
            depth = 0
            i = start
            while i < len(raw_text):
                ch = raw_text[i]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        code = raw_text[start + 1:i].strip()
                        # 如果存在的话，删除下加标记的反键包装
                        code = re.sub(r'^```(?:python)?\s*', '', code)
                        code = re.sub(r'\s*```$', '', code)
                        # 如果引用了整件事，删除周围的引号
                        for q in ['"""', "'''", '"', "'"]:
                            if code.startswith(q) and code.endswith(q):
                                code = code[len(q):-len(q)]
                                break
                        return code if code else None
                i += 1
            
            # 平衡剖析失败— 尝试修炼： 抓住所有之间
            # tool_code (和下 ReAct 部分标记)
            code = raw_text[start + 1:]
            for marker in ['[Observe]', '[Thought]', '[Finish]', '[Act]']:
                pos = code.find(marker)
                if pos != -1:
                    code = code[:pos]
                    break
            
            # 剥离线索(如果存在)
            code = code.rstrip().rstrip(')')
            code = code.strip()
            
            # 删除向下标记的反键包装
            code = re.sub(r'^```(?:python)?\s*', '', code)
            code = re.sub(r'\s*```$', '', code)
            
            return code if code else None
        
        def _extract_search_from_raw_output(
            self, sources: list
        ) -> Optional[str]:
            """Extract a search query from raw LLM output that OneTwo couldn't parse.
            
            Looks for search('query') patterns in all sources, and also infers
            search intent from model thoughts (e.g., "I need to search for X").
            
            Args:
                sources: List of (source_name, source_text) tuples.
                
            Returns:
                Extracted search query string, or None.
            """
            # 模式 1:明确搜索 () 呼叫
            for source_name, source_text in sources:
                for pattern in [
                    r"""search\(['"](.*?)['"]\)""",
                    r"""search\(["'](.*?)["']\)""",
                ]:
                    match = re.search(pattern, source_text)
                    if match:
                        query = match.group(1).strip()
                        if query and len(query) > 3:
                            print(f"[PHIA:recover] Extracted search query from {source_name}: '{query}'")
                            return query
            
            # 模式 2:从思想中搜索意图
            # "我要在旧金山寻寻寻一般的台阶"
            # "现在我应该去寻找典型的旧金山"
            for source_name, source_text in sources:
                text_lower = source_text.lower()
                for trigger in ['search for ', 'look up ', 'find out about ',
                                'search about ', 'look for ']:
                    idx = text_lower.find(trigger)
                    if idx != -1:
                        # 将句子的其余部分提取为搜索查询
                        query_start = idx + len(trigger)
                        remainder = source_text[query_start:]
                        # 句子边界停止
                        for end_marker in ['.', '!', '\n', ',', ' and ', ' to ']:
                            end_idx = remainder.find(end_marker)
                            if end_idx != -1 and end_idx > 3:
                                remainder = remainder[:end_idx]
                                break
                        query = remainder.strip().rstrip('.')
                        if query and 5 < len(query) < 100:
                            print(f"[PHIA:recover] Inferred search query from {source_name} thought: '{query}'")
                            return query
            
            return None
        
        def _debug_completeness_check(self, text: str):
            """Print why _is_complete_response rejects a text. For diagnostics only."""
            if not text:
                print(f"[PHIA:complete?]   → REJECTED: empty text")
                return
            text_normalized = text.lower().replace('`', '').replace('  ', ' ')
            incomplete_markers = [
                '[act]:', '[thought]:', '```tool_code', '```python',
                'i need to', "i'll",
                'i need the user', 'i need your',
                'then use search', 'then use tool_code', 'then use web',
                'from summary_df', 'from activities_df',
                'and compute the', 'and calculate the',
                ', then use', ', then search', ', then calculate',
            ]
            planning_let_me = [
                'let me check', 'let me analyze', 'let me look', 'let me calculate',
                'let me search', 'let me get', 'let me find', 'let me pull',
                'let me run', 'let me compute', 'let me first',
            ]
            planning_i_will = [
                'i will check', 'i will analyze', 'i will now', 'i will look',
                'i will search', 'i will calculate', 'i will use', 'i will first',
                'i will run', 'i will compute', 'i will get',
            ]
            all_markers = incomplete_markers + planning_let_me + planning_i_will
            for marker in all_markers:
                if marker in text_normalized:
                    idx = text_normalized.find(marker)
                    ctx_start = max(0, idx - 20)
                    ctx_end = min(len(text_normalized), idx + len(marker) + 20)
                    context = text_normalized[ctx_start:ctx_end]
                    print(f"[PHIA:complete?]   → REJECTED: found marker '{marker}' in: ...{context}...")
                    return
            if text_normalized.strip().startswith('i need'):
                print(f"[PHIA:complete?]   → REJECTED: starts with 'i need'")
                return
            text_norm_stripped = text_normalized.strip()
            if text_norm_stripped.startswith('tool_code(') or text_norm_stripped.startswith('search('):
                print(f"[PHIA:complete?]   → REJECTED: raw tool call")
                return
            code_indicators = ['summary_df[', 'activities_df[', 'profile[', 
                               '.mean()', '.sum()', '.max()', '.min()',
                               'print(f"', 'print(f\'', '\\n']
            code_indicator_count = sum(1 for ci in code_indicators if ci in text)
            if code_indicator_count >= 3:
                print(f"[PHIA:complete?]   → REJECTED: code content ({code_indicator_count} indicators)")
                return
            if 'dtype:' in text and 'Name:' in text:
                print(f"[PHIA:complete?]   → REJECTED: raw pandas output")
                return
            url_count = text_normalized.count('url:')
            relevance_count = text_normalized.count('relevance score:')
            if url_count >= 2 or relevance_count >= 2:
                print(f"[PHIA:complete?]   → REJECTED: raw search results (urls={url_count}, relevance={relevance_count})")
                return
            if '#ERROR#' in text or 'Invalid syntax for positional arguments' in text:
                print(f"[PHIA:complete?]   → REJECTED: OneTwo error")
                return
            if len(text) <= 30:
                print(f"[PHIA:complete?]   → REJECTED: too short ({len(text)} <= 30)")
                return
            print(f"[PHIA:complete?]   → ACCEPTED (no markers found, len={len(text)})")
        
        def _clean_react_format(self, text: str) -> str:
            """Remove ReAct formatting artifacts from text.
            
            Args:
                text: Text that may contain ReAct formatting.
                
            Returns:
                Cleaned text.
            """
            if not text:
                return ""
            
            text = str(text)
            
            # 删除开始时常见的 ReAct前缀
            prefixes_to_remove = [
                '[Thought]:', '[Act]:', '[Observe]:', '[Finish]:',
                'Thought:', 'Act:', 'Observe:', 'Finish:',
            ]
            for prefix in prefixes_to_remove:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            
            # 删除文本中的代码块
            import re
            text = re.sub(r'```[\w]*\n.*?```', '', text, flags=re.DOTALL)
            
            # 删除内线 [行为] : 以及之后的所有内容( 不完整的响应)
            for marker in ['[Act]:', '[Thought]:', '[Observe]:']:
                if marker in text:
                    text = text.split(marker)[0].strip()
            
            # 删除 JSON 格式化文物
            if text.startswith('```json') or text.strip().startswith('{'):
                return ""
            
            # 如果只留下代码块，返回空
            if text.startswith('```') and text.endswith('```'):
                return ""
            
            return text.strip()
        
        def _strip_planning_preamble(self, text: str) -> str:
            """Strip planning/meta sentences from the start of a long response.
            
            Some models (especially 2.5 Flash) write their [Finish] with a
            planning sentence prefix like:
              "I have gathered the user's key health metrics. I will now
               synthesize this information... [actual good content follows]"
            
            This method removes those leading sentences so the actual content
            can pass _is_complete_response.
            
            Args:
                text: Long text (>500 chars) that was rejected by _is_complete_response.
                
            Returns:
                The text with planning preamble stripped, or empty string if
                we can't salvage anything substantial.
            """
            import re
            
            # 计划标记以检查单个句子
            planning_markers = [
                'i will now', 'i will synthesize', 'i will analyze', 'i will check',
                'i will use', 'i will search', 'i will calculate', 'i will first',
                'i will provide', 'i will combine', 'i will compile',
                'i have gathered', 'i have analyzed', 'i have retrieved',
                'i have already', 'i have now', 'i have collected',
                'i need to', "i'll now", "i'll synthesize", "i'll analyze",
                'let me synthesize', 'let me analyze', 'let me compile',
                'let me now', 'let me provide', 'let me combine',
                'now i can', 'now i will', 'now that i have',
            ]
            
            # 分为句子(处理“.”和“.\n”)
            sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=5)
            
            if len(sentences) <= 1:
                return ""
            
            # 查找有多少主要句子在规划/元数据
            skip_count = 0
            for sent in sentences[:3]:  # 最多检查一下3句子
                sent_lower = sent.lower()
                if any(m in sent_lower for m in planning_markers):
                    skip_count += 1
                else:
                    break  # 一开始停止非规划句
            
            if skip_count == 0:
                return ""  # 未找到规划序言
            
            # 没有规划句子的文本重建
            remainder = ' '.join(sentences[skip_count:]).strip()
            
            # 只有我们还有实质性的内容，才能返回
            if len(remainder) < 200:
                return ""
            
            print(f"[PHIA:salvage] Stripped {skip_count} planning sentence(s), "
                  f"{len(remainder)} chars remaining")
            return remainder
        
        def _is_complete_response(self, text: str) -> bool:
            """Check if a response looks complete (not mid-ReAct-loop).
            
            Args:
                text: Response text to check.
                
            Returns:
                True if the response looks complete.
            """
            if not text:
                return False
            
            # 去除反引号和多余空格，统一后再检测特定模式。
            text_normalized = text.lower().replace('`', '').replace('  ', ' ')
            
            # 反应不完整的迹象 - 模型大声思考
            incomplete_markers = [
                '[act]:', '[thought]:', '```tool_code', '```python',
                'i need to', "i'll",
                'i need the user',  # "我需要用户的平均值..."
                'i need your',  # "我需要你每天平均的台阶..."
                'then use search', 'then use tool_code', 'then use web',  # 规划报表
                'from summary_df', 'from activities_df',  # 计划中提及的数据框架
                'and compute the', 'and calculate the',  # 规划的继续
                ', then use', ', then search', ', then calculate',  # 多步骤计划
            ]
            
            # "let me" is a planning phrase ("let me check", "let me analyze"),
            # 但NOT,当用 对话("只要让我知道，"请让我知道")
            # "I will"是计划("I will nowcheck"),但NOT当最终("I will be happy to").
            # 以更多背景处理这些内容：
            planning_let_me = [
                'let me check', 'let me analyze', 'let me look', 'let me calculate',
                'let me search', 'let me get', 'let me find', 'let me pull',
                'let me run', 'let me compute', 'let me first',
            ]
            planning_i_will = [
                'i will check', 'i will analyze', 'i will now', 'i will look',
                'i will search', 'i will calculate', 'i will use', 'i will first',
                'i will run', 'i will compute', 'i will get',
            ]
            
            for marker in incomplete_markers:
                if marker in text_normalized:
                    return False
            
            for marker in planning_let_me:
                if marker in text_normalized:
                    return False
            
            for marker in planning_i_will:
                if marker in text_normalized:
                    return False
            
            # 检查从“我需要”开始的规划报表
            if text_normalized.strip().startswith('i need'):
                return False
            
            # 检测原始工具呼叫(模型有时直接输出这些，
            # 可能包裹在背影中)
            # 使用已脱落的反光片 text_normalized
            text_norm_stripped = text_normalized.strip()
            if text_norm_stripped.startswith('tool_code(') or text_norm_stripped.startswith('search('):
                return False
            
            # 检测主要为代码而不是自然语言的响应
            # 这捕捉到模型将预定工具调用为文本的情况
            code_indicators = ['summary_df[', 'activities_df[', 'profile[', 
                               '.mean()', '.sum()', '.max()', '.min()',
                               'print(f"', 'print(f\'', '\\n']
            code_indicator_count = sum(1 for ci in code_indicators if ci in text)
            if code_indicator_count >= 3:
                return False
            
            # 检测原始 pandas 输出( 序列重排， dtypeinfo)
            # 例如"年龄：0 34\ nName:年龄，型号:int64".
            if 'dtype:' in text and 'Name:' in text:
                return False
            
            # 检测原始搜索结果被返回而不是合成答案
            # GPT-5. 2有时会输出原始的观察作为最后的答案。
            url_count = text_normalized.count('url:')
            relevance_count = text_normalized.count('relevance score:')
            if url_count >= 2 or relevance_count >= 2:
                # 多重 URLs 或关联分数 = 原始搜索输出，而非合成
                return False
            
            # 检测 OneTwo 解析错误(发生于 OpenAI + PHIA ).
            if '#ERROR#' in text or 'Invalid syntax for positional arguments' in text:
                return False
            
            # 完整的回复应具有一些实质性内容
            return len(text) > 30
        
        def _is_truncated_response(self, text: str) -> bool:
            """Detect responses that got cut off mid-generation.
            
            Common with verbose models (2.5 Pro) where long thoughts eat
            into the token budget, leaving the [Finish] text truncated.
            
            Examples of truncated responses:
            - "Here's a breakdown of what your numbers mean and some dietary tips to help you maintain and even improve it."
            - "Of course! Here are some dietary tips to help you maintain a healthy lifestyle:"
            """
            if not text or len(text) > 500:
                # 长期答复不太可能被截断
                return False
            
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            
            # 冒号结尾可能表示后续内容未生成。
            if text_stripped.endswith((':', '：')):
                print(f"[PHIA:truncation] Detected: ends with ':'")
                return True
            
            # 简短的反应，承诺列表/细目，但不能提供
            promise_phrases = [
                "here's a breakdown", "here are some", "here is a breakdown",
                "the following", "consider the following", "these tips",
                "some tips", "some suggestions", "some recommendations",
            ]
            has_promise = any(p in text_lower for p in promise_phrases)
            
            # 检查承诺是否实际实现( 列出项目或实质性内容)
            has_list = any(marker in text for marker in ['- ', '* ', '1.', '•'])
            has_paragraph = text.count('\n\n') >= 1 and len(text) > 300
            
            if has_promise and not has_list and not has_paragraph:
                print(f"[PHIA:truncation] Detected: promises content ({len(text)} chars) but no list/detail")
                return True
            
            return False
        
        def _is_substantive_response(self, text: str, final_state: Any) -> bool:
            """Check if a response has actual substance — not just a vague preamble
            or raw data dump.
            
            Catches two failure modes:
            A) Model does all analysis in [Thought] but writes a short, vague [Finish]
               like "Let's look at your data together..." with no actual numbers.
            B) Extraction falls back to raw tool_code output like 
               "Average Sleep Minutes: 417.80\\nAverage Deep Sleep: 62.38..."
               which is a data dump with no interpretation.
            
            Returns:
                True if the response is substantive enough to send to the user.
            """
            if not text:
                return True  # 让其他检查处理为空
            
            # 跳过此检查错误消息(它们有自己的格式)
            if text.startswith('⚠️') or 'Could not extract' in text:
                return True
            
            # 检查是否实际使用了tool_code(如果没有使用工具，
            # 只有对话方能做出正确反应)
            tool_code_used = False
            if hasattr(final_state, 'updates') and final_state.updates:
                for step in final_state.updates:
                    if step.action and 'tool_code' in str(step.action):
                        tool_code_used = True
                        break
            
            if not tool_code_used:
                return True  # 没有计算数据，对话反应很好
            
            # 修正A:检测模糊的序言——没有数据值的短答。
            # 尽管tool_code被用于计算数据
            if len(text) < 500:
                import re
                # 查找实际健康数值(例如 " 62.17 bpm], "417 mine, "7.0 小时")
                health_numbers = re.findall(r'\d+\.?\d*\s*(?:bpm|min|hrs?|hours?|[%％]|steps|ms|小时|分钟|毫秒|步|次/分钟)', text.lower())
                # 同时检查独立的有意义的数字( 不仅仅是 " 1]" 列表标记)
                data_numbers = re.findall(r'\b\d{2,}\b', text)  # 带有 2+ 数字
                
                if not health_numbers and not data_numbers:
                    print(f"[PHIA:quality] REJECTED: short response ({len(text)} chars) with "
                          f"no health data values despite tool_code usage")
                    return False
            
            # Fix B:检测原始数据堆放——多为"标签:数字"行。
            # 没有谈话的句子或解释
            if self._is_raw_data_dump(text):
                print(f"[PHIA:quality] REJECTED: raw data dump with no interpretation")
                return False
            
            return True
        
        def _is_raw_data_dump(self, text: str) -> bool:
            """Detect responses that are just raw tool output with no interpretation.
            
            Pattern: Multiple lines of "Label: number" with no sentence structure.
            Example: "Average Sleep Minutes: 417.80\\nAverage Deep Sleep: 62.38..."
            """
            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
            if len(lines) < 3:
                return False
            
            import re
            # 计算“ 标签： 数字” 格式的行数
            data_lines = 0
            sentence_lines = 0
            for line in lines:
                # 跳过“根据我的分析”前缀
                if line.startswith('Based on'):
                    continue
                # “ 标签： 数字” 模式
                if re.match(r'^[\w\s]+:\s*[\d.]+', line):
                    data_lines += 1
                # 实际句子(有期、问号或感叹，以及> 40字符)
                elif len(line) > 40 and any(p in line for p in ['. ', '? ', '! ']):
                    sentence_lines += 1
            
            # 如果大多数数据线没有解释句子，那就是一个垃圾
            if data_lines >= 3 and sentence_lines == 0:
                return True
            
            return False
        
        def _collect_step_context(self, final_state: Any) -> tuple:
            """Collect code and data from ReAct steps for recovery context.
            
            Returns:
                (code_context, data_context): 
                    code_context: The code that was executed (with variable names)
                    data_context: The output/observations from tool calls
            """
            code_snippets = []
            data_snippets = []
            
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return "", ""
            
            for step in final_state.updates:
                # 从动作中提取代码
                if step.action:
                    action_str = str(step.action)
                    if 'tool_code' in action_str:
                        # 尝试提取代码参数
                        import re
                        # 匹配 args= (“ code_here” 或 args= (“code_here”),
                        code_match = re.search(r"args=\(['\"](.+?)['\"],?\)", action_str, re.DOTALL)
                        if code_match:
                            code_snippets.append(code_match.group(1))
                
                # 提取观察(非异常工具输出)
                if step.observation:
                    obs = str(step.observation)
                    if ('#ERROR#' not in obs and 'Search Results' not in obs 
                            and len(obs) > 10):
                        data_snippets.append(obs)
            
            # 重复数据片段(破获案件)
            seen = set()
            unique_data = []
            for d in data_snippets:
                if d not in seen:
                    seen.add(d)
                    unique_data.append(d)
            
            code_context = '\n'.join(code_snippets[:3])  # 最大 3 代码块
            data_context = '\n'.join(unique_data[:3])  # 最大 3 独特输出
            
            return code_context, data_context
        
        def _collect_full_step_context(self, final_state: Any) -> str:
            """Collect ALL useful context from ReAct steps for rich recovery.
            
            Unlike _collect_step_context (which returns just code + data),
            this collects thoughts, search results, and tool output — everything
            the model already reasoned about — so the recovery LLM can write
            a comprehensive answer without re-doing the work.
            
            Returns:
                A formatted string with all step context.
            """
            sections = []
            
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return ""
            
            for i, step in enumerate(final_state.updates):
                step_parts = []
                
                # 思考:模型的分析(往往包含真实的洞察力).
                if step.thought:
                    thought = str(step.thought).strip()
                    # 跳过纯规划思想
                    thought_lower = thought.lower()
                    if not any(p in thought_lower for p in [
                        'i need to', 'i should first', 'let me first'
                    ]):
                        step_parts.append(f"Analysis: {thought}")
                
                # 观察:工具输出或搜索结果
                if step.observation:
                    obs = str(step.observation).strip()
                    if '#ERROR#' not in obs and len(obs) > 10:
                        if 'Search Results' in obs:
                            step_parts.append(f"Search findings: {obs[:800]}")
                        else:
                            step_parts.append(f"Data: {obs[:500]}")
                
                if step_parts:
                    sections.append(f"Step {i+1}:\n" + "\n".join(step_parts))
            
            return "\n\n".join(sections)
        
        def _generate_enriched_answer(
            self, user_message: str, full_context: str
        ) -> Optional[str]:
            """Generate a comprehensive answer using all available step context.
            
            Unlike _generate_summary (which only gets raw numbers), this gets
            the model's own analysis, search results, and data — everything
            needed to write a rich response without losing context.
            """
            if not full_context:
                return None
            
            prompt = (
                f'You are a friendly health data analyst. The user asked: "{user_message}"\n\n'
                f'Here is what was found by analyzing their Fitbit health data:\n\n'
                f'{full_context[:4000]}\n\n'
                f'Please provide a helpful, comprehensive response to the user\'s question '
                f'based on ALL of the above findings. Include:\n'
                f'- Specific numbers from the data analysis\n'
                f'- Context about what those numbers mean for their health\n'
                f'- Relevant information from the search results\n'
                f'- Actionable suggestions\n'
                f'Be conversational, supportive, and thorough. '
                f'无需提及工具、代码、DataFrame 或其他实现细节。使用用户提问的语言作答。'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] Enriched answer generated ({len(result)} chars)")
            return result
        
        def reset_conversation(self) -> None:
            """Reset the conversation state."""
            self.conversation_history = []
            self.last_response = ""
            # 重建代理以清除任何状态
            self._agent = None


else:
    # 无法使用一个 2 个时的后退类
    class PHIABaseline:
        """PHIA Baseline placeholder when onetwo is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PHIABaseline requires onetwo which is not available. "
                f"Error: {_ONETWO_IMPORT_ERROR}"
            )


# =============================================================================
# 工厂函数
# =============================================================================

def create_phia_baseline(
    api_key: Optional[str] = None,
    provider: str = "gemini",
    tavily_api_key: Optional[str] = None,
    settings: Optional[Any] = None,
    few_shot_dir: Optional[str] = None,
    debug_verbose: bool = False,
    model_name: Optional[str] = None,
    # 遗留参数
    gemini_api_key: Optional[str] = None,
) -> PHIABaseline:
    """Create a fully configured PHIABaseline.
    
    Args:
        api_key: API key for the selected provider.
        provider: "gemini" or "openai". Default is "gemini".
        tavily_api_key: Optional Tavily API key for web search.
        settings: Optional Settings object for data paths.
        few_shot_dir: Directory containing few-shot notebook examples.
        debug_verbose: If True, print debug information.
        model_name: Optional model name. Defaults vary by provider:
            - Gemini: "models/gemini-2.5-pro" (recommended)
            - OpenAI: "gpt-4o"
        gemini_api_key: DEPRECATED - use api_key instead.
        
    Returns:
        Configured PHIABaseline instance.
        
    Example:
        ```python
        # With Gemini
        baseline = create_phia_baseline(
            api_key="your-key",
            provider="gemini",
        )
        
        # With OpenAI
        baseline = create_phia_baseline(
            api_key="your-key",
            provider="openai",
        )
        
        response = baseline.respond("How is my sleep?")
        ```
    """
    if not _ONETWO_AVAILABLE:
        raise ImportError(
            f"PHIABaseline requires onetwo. Error: {_ONETWO_IMPORT_ERROR}"
        )
    
    from config.settings import Settings
    from ..utils import load_persona
    
    # 使用已提供的设置或创建默认值
    if settings is None:
        settings = Settings()
    
    # 装入数据
    summary_df, activities_df, profile_df, _ = load_persona(settings=settings)
    
    # 创建和配置 基线
    baseline = PHIABaseline(
        tavily_api_key=tavily_api_key,
        few_shot_dir=few_shot_dir,
        debug_verbose=debug_verbose,
    )
    
    # 处理新旧两个问题API
    # 确定来自 api_key 或 遗留 gemini_api_key 的提供者
    effective_api_key = api_key if api_key else gemini_api_key
    effective_provider = provider if api_key else "gemini"
    
    # 配置模型(如果未指定使用默认)
    baseline.configure(
        api_key=effective_api_key,
        provider=effective_provider,
        model_name=model_name,
    )
    
    baseline.load_data(
        summary_df=summary_df,
        activities_df=activities_df,
        profile_df=profile_df,
    )
    
    return baseline


def is_onetwo_openai_available() -> bool:
    """Check if OneTwo OpenAI backend is available."""
    return _ONETWO_OPENAI_AVAILABLE
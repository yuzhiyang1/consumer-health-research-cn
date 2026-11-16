"""Domain Expert Agent for health contextualization and interpretation.

This agent provides medical domain expertise for interpreting health data,
contextualizing metrics against population norms, and providing evidence-based
health information using the ReAct framework.

Usage:
    from pha.agents import DomainExpertAgent
    
    agent = DomainExpertAgent()
    agent.get_agent(gemini_api_key="your-key")
    response = agent.call_agent("What does my HbA1c of 5.8% mean?")
"""

import json
import os
import textwrap
from typing import Optional, List, Literal, Sequence, Any
import urllib.parse
import pandas as pd
import requests
from datetime import datetime

# 用于基于笔记本的示例解析(遵循 PHIA 模式)
try:
    import nbformat
    _NBFORMAT_AVAILABLE = True
except ImportError:
    _NBFORMAT_AVAILABLE = False

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


# =============================================================================
# OpenAI API 兼容性补丁
# =============================================================================
# 较新的OpenAI型号需要`max_completion_tokens`而不是`max_tokens`.
# 此补丁确保了像一二这样的库的兼容性。

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
            """Extract content inside balanced parentheses."""
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
            """Transform GPT-style tool calls to OneTwo-compatible format."""
            if not text:
                return text
            result = []
            i = 0
            tool_names = ['tool_code', 'search']
            while i < len(text):
                found_tool = False
                for tool_name in tool_names:
                    if text[i:].startswith('[Act]:'):
                        act_match = re.match(r'\[Act\]:\s*', text[i:])
                        if act_match:
                            prefix = act_match.group(0)
                            tool_start = i + len(prefix)
                            for tn in tool_names:
                                if text[tool_start:].startswith(tn + '('):
                                    arg_content, end_pos = _extract_balanced_parens(text, tool_start + len(tn))
                                    if arg_content is not None:
                                        arg_stripped = arg_content.strip()
                                        is_quoted = (
                                            (arg_stripped.startswith('"') and arg_stripped.endswith('"')) or
                                            (arg_stripped.startswith("'") and arg_stripped.endswith("'")) or
                                            arg_stripped.startswith('"""') or arg_stripped.startswith("'''")
                                        )
                                        if is_quoted:
                                            result.append(text[i:end_pos+1])
                                        else:
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
                                arg_stripped.startswith('"""') or arg_stripped.startswith("'''")
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
            # 手动 `_truncate_at_stop_sequences` 后处理
            # 尊重返回内容上要求的停止语义。
            if is_gpt5 and 'stop' in kwargs:
                kwargs.pop('stop')

            result = _original_create(self, *args, **kwargs)
            
            try:
                if hasattr(result, 'choices') and result.choices:
                    for choice in result.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            if choice.message.content:
                                content = choice.message.content
                                
                                # GPT-5: 备份手动截断在停止序列
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


# 尝试导入一个 2 个组件
# 注:由于使用新类型的参数语法，需要1个Python 3.12+
_ONETWO_AVAILABLE = False
_ONETWO_OPENAI_AVAILABLE = False
_ONETWO_GENAI_AVAILABLE = False  # 如果有新的 GoogleGenAIAPI 后端，则为真
_ONETWO_IMPORT_ERROR = None
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
        # 当OpenAI 后端可用时应用补丁
        _patch_openai_for_compatibility()
    except ImportError:
        pass  # OpenAI 后端不可用
except ImportError as e:
    _ONETWO_IMPORT_ERROR = str(e)
except SyntaxError as e:
    # Python版本太老了(一二需要3.12+).
    import sys
    _ONETWO_IMPORT_ERROR = f"onetwo requires Python 3.12+, you have {sys.version}"
except Exception as e:
    _ONETWO_IMPORT_ERROR = str(e)


from ..prompts.domain_expert_prompts import (
    check_reference_ranges,
)


__version__ = "0.9.0"


# =============================================================================
# 工具示例
# =============================================================================

ORGANIC_SEARCH_TOOL_EXAMPLE = textwrap.dedent("""\
      search('what is the normal range for HDL cholesterol?')
      # returns (['https://my.clevelandclinic.org/health/articles/11920-cholesterol-numbers-what-do-they-mean'],
      ['Your HDL ("good" cholesterol) is the one number you want to be high (ideally above 60). Your LDL ("bad" cholesterol) should be below 100.])
    """)

DC_EXAMPLE_RESPONSE = textwrap.dedent("""
    datacommons_natural_language_query(query="What is the most common reason for death in the US?"),

    #returns A JSON with the necessary information to augment generation
    """)


# =============================================================================
# 序言
# =============================================================================

CONTEXTUAL_REASONING_PREAMBLE_V1 = """\
{#- 系统指令和可用工具 -#}
{%- role name='system' -%}
你具备内科、健康和运动医学知识，能够解释个人健康数据、化验结果和可穿戴设备记录。使用搜索工具查证资料，并引用实际查到的链接。

仔细核对已有数据。必要时调用工具，只在确实缺少关键资料时向用户补问。回复应给足背景，结合用户的年龄、性别、BMI 和生活习惯，使用可信证据并解释医学术语及缩写。只有确知缺少某种数据且该数据对当前问题有用时，才说明缺少什么、请用户提供什么。使用用户最新一轮发言的语言作答，除非用户另有要求。

当前可用工具：
{% for tool in tools %}
工具名称：{{ tool.name }}
工具说明：{{ tool.description }}
{% if tool.example -%}
  工具示例：{{ tool.example_str }}
{%- endif -%}
{% endfor %}

严格执行 ReAct 循环：
1. [Thought]: 用一到两句话说明下一步。
2. [Act]: 单独输出一个工具调用，例如 [Act]: search('normal HDL cholesterol range')。
3. [Observe]: 等待并阅读工具结果。
4. 反复执行前三步，直到信息足够。
5. [Finish]: 整理证据并向用户给出最终回答。

格式要求：
- 搜索词用引号包裹，例如 search('your search query')。
- 工具调用不得放在 Markdown 代码块或反引号中。
- 执行实际工具调用，无需只写计划。阅读并归纳搜索结果，避免原样输出。
- 最终回答始终使用 [Finish]:。

工具分工：
- search：查找人群统计、城市或国家对比、临床指南、医学事实和健康建议。
- datacommons_natural_language_query：查询 CDC、WHO 等来源的美国人群健康数据。
- compare_blood_test_values_with_reference_ranges：判断化验结果是否处于参考范围。

数据访问限制：
- 无法直接访问 summary_df、activities_df 或用户原始健康数据。
- 个人指标由数据科学 Agent 计算，见 [SUPPORTING_AGENT_INSIGHTS]。使用已有发现，无需导入 pandas 或读取 DataFrame。
- 比较不同城市或国家的人群时，必须通过 search 查找统计数据，不能使用 tool_code 在本地计算。

示例 1：使用数据科学 Agent 的结果
[Thought]: 数据科学 Agent 已算出用户平均走 8156 步，接下来查找用于对比的人群统计。
[Act]: search('average daily steps San Francisco adults')
[Observe]: （旧金山成人步数的来源链接和摘要）
[Finish]: 按你的记录，日均步数为 8156 步，高于查到的旧金山成人参考值（约 6500 步）。

示例 2：比较多个地区
[Thought]: 需要查找旧金山成人的平均步数。
[Act]: search('average daily steps San Francisco adults')
[Observe]: （旧金山成人步数的来源链接和摘要）
[Thought]: 接下来查找香港成人数据。
[Act]: search('average daily steps Hong Kong adults')
[Observe]: （香港成人步数的来源链接和摘要）
[Finish]: 你的日均步数为 8156 步，高于示例中的旧金山参考范围（约 6000–7000 步），与香港的参考范围（约 7500–8500 步）相近。

{#- Preamble: ReAct few-shots #}
以下示例展示如何借助工具分析健康数据。参考示例的步骤，根据当前问题独立作答。
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
{%- if step.is_finished and step.observation and not step.action -%}
  [Finish]: {{ step.observation + '\n' }}
{%- endif -%}
{%- endfor -%}
{%- endfor %}

{# Start of the processing of the actual inputs. -#}

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
{%- if step.is_finished and step.observation and not step.action -%}
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


# =============================================================================
# 领域专家 Agent 类
# =============================================================================

if _ONETWO_AVAILABLE:
    
    class DomainExpertAgent:
        """Domain Expert Agent using onetwo ReActAgent.
        
        Uses onetwo's ReActAgent for iterative reasoning with tools.
        
        Example:
            >>> agent = DomainExpertAgent()
            >>> agent.get_agent(gemini_api_key="your-key")
            >>> response = agent.call_agent("Is my LDL of 145 concerning?")
        """
        
        def __init__(
            self,
            preamble: Optional["react.ReActPromptJ2"] = None,
            context_memory: str = "",
            search_backend: str = "tavily",
            tavily_api_key: Optional[str] = None,
        ):
            """Initialize the Domain Expert Agent.
            
            Args:
                preamble: Custom ReAct prompt (uses default if not provided).
                context_memory: Additional context to include.
                search_backend: "tavily" or "duckduckgo" for web search.
                tavily_api_key: API key for Tavily (optional).
            """
            self.system_prompt = preamble
            self.context_memory = context_memory
            self.user_phr_data: Optional[str] = None
            self.domain_expert_agent: Optional[react.ReActAgent] = None
            self.search_backend = search_backend
            self.tavily_api_key = tavily_api_key
        
        def setup_agent(
            self, 
            exemplar_files: Optional[Sequence[str]] = None
        ) -> "react.ReActAgent":
            """Sets up the domain expert agent as a ReAct agent.
            
            Args:
                exemplar_files: Optional list of notebook file paths for few-shot examples.
                    If not provided, uses default few_shots directory or hardcoded exemplars.
            
            Raises:
                ValueError: If search_backend is "tavily" but no API key is provided.
            """
            # 验证 Tavily API 密钥，如果选中后端
            if self.search_backend == "tavily":
                api_key = self.tavily_api_key or os.environ.get("TAVILY_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Tavily API key required for search. Either:\n"
                        "  1. Set TAVILY_API_KEY environment variable\n"
                        "  2. Pass tavily_api_key parameter to DomainExpertAgent()\n"
                        "  3. Use search_backend='duckduckgo' for free (lower quality) search\n"
                        "Get a Tavily API key at: https://tavily.com/"
                    )
            
            # 尝试装入基于笔记本的示例( PHIA 模式)
            health_exemplars = []
            
            if exemplar_files:
                # 使用已提供的示例文件
                health_exemplars = self._build_exemplars_from_notebooks(exemplar_files)
            else:
                # 尝试查找默认 few_shots目录
                import glob
                # 查找此文件的 few_shots 或常见位置
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'few_shots', '*.ipynb'),
                    os.path.join(os.getcwd(), 'few_shots', '*.ipynb'),
                    'few_shots/*.ipynb',
                ]
                for pattern in possible_paths:
                    found_files = glob.glob(pattern)
                    if found_files:
                        health_exemplars = self._build_exemplars_from_notebooks(found_files)
                        break
            
            # 如果找不到笔记本，则返回到硬编码实例
            if not health_exemplars:
                health_exemplars = self._create_health_domain_exemplars()
            
            # 获取默认 ReAct 实例用于搜索/完成格式加固。
            # python_tool_name 需要但未使用——通过占位符
            # 因为tool_code 不再可供这名特工使用。
            # 代码执行由 data_science_agent 处理;域名
            # 专家应依靠其计算出的见解。
            default_fewshots = react.default_react_exemplars(
                python_tool_name="_unused",
                search_tool_name="search",
                finish_tool_name="finish",
            )
            health_exemplars.extend(default_fewshots)

            # 创建 ReAct 代理
            domain_expert_agent = react.ReActAgent(
                prompt=self.get_jinja_preamble_from_string(),
                exemplars=health_exemplars,
                environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
                    tools=self.get_llm_toolbox(),
                ),
                max_steps=5,
                stop_prefix="",
            )
            self.domain_expert_agent = domain_expert_agent
            return domain_expert_agent
        
        def _build_exemplars_from_notebooks(
            self, 
            example_files: Sequence[str]
        ) -> List["react.ReActState"]:
            """Build ReAct exemplars from notebook files (PHIA pattern).
            
            This follows the exact same format as PHIA's build_exemplars function.
            Notebooks should have:
            - First cell: markdown with the question
            - Subsequent cells: alternating markdown (thought) and code (action)
            - Search calls formatted as: # search('query')
            - Final answer as: print("...")
            
            Args:
                example_files: Paths to notebook files.
                
            Returns:
                List of ReActState exemplars.
            """
            if not _NBFORMAT_AVAILABLE:
                print("Warning: nbformat not available, cannot parse notebook exemplars")
                return []
            
            _PYTHON_TOOL_NAME = "tool_code"
            _SEARCH_TOOL_NAME = "search"
            _FINISH_TOOL_NAME = "finish"
            
            exemplars = []
            for example_file in example_files:
                if not os.path.exists(example_file):
                    continue
                    
                try:
                    with open(example_file, 'r', encoding='utf-8') as f:
                        raw_nb = f.read()
                    nb = nbformat.reads(raw_nb, as_version=4)
                except Exception as e:
                    print(f"Failed to parse notebook: {example_file}: {e}")
                    continue
                
                cells = nb.get("cells", [])
                if not cells:
                    continue
                
                # 问题描述在第一个单元格
                description_cell = cells.pop(0)
                problem_description = "".join(description_cell.get("source", [])).strip()
                problem_description = problem_description.lstrip("#").strip()
                
                planner_updates = []
                
                # 进程单元格成对( markdownthought + 代码动作)
                i = 0
                while i < len(cells):
                    if (i < len(cells) - 1 
                        and cells[i].get("cell_type") == "markdown"
                        and cells[i + 1].get("cell_type") == "code"):
                        
                        thought = "".join(cells[i].get("source", [])).strip()
                        code = "".join(cells[i + 1].get("source", [])).strip()
                        
                        # 删除测试标记
                        code = code.replace('# @test {"skip": true}\n', "")
                        
                        # 根据代码内容确定调用函数
                        code_lower = code.lower()
                        if "search" in code_lower and code.strip().startswith("#"):
                            function_name = _SEARCH_TOOL_NAME
                            # 从# 搜索('query') 中提取查询
                            try:
                                args = (code.split("'")[1],)
                            except IndexError:
                                args = (code,)
                        elif "compare_blood_test" in code_lower or "check_interval" in code_lower:
                            function_name = "compare_blood_test_values_with_reference_ranges"
                            args = (code,)
                        else:
                            function_name = _PYTHON_TOOL_NAME
                            args = (code,)
                        
                        # 如果可用，可以从输出中获取观察
                        observation = ""
                        try:
                            outputs = cells[i + 1].get("outputs", [])
                            if outputs:
                                output = outputs[0]
                                if "text" in output:
                                    observation = "".join(output["text"]).strip()
                                elif "data" in output and "text/plain" in output["data"]:
                                    observation = "".join(output["data"]["text/plain"]).strip()
                        except (KeyError, IndexError):
                            pass
                        
                        # 格式：MARKDOWN,用于搜索 tool_code, PYTHON
                        # 这符合 PHIA 的图案
                        fmt = (
                            llm_tool_use.ArgumentFormat.MARKDOWN
                            if function_name == _PYTHON_TOOL_NAME
                            else llm_tool_use.ArgumentFormat.PYTHON
                        )
                        
                        step = react.ReActStep(
                            thought=thought,
                            is_finished=False,
                            action=llm_tool_use.FunctionCall(
                                function_name=function_name, 
                                args=args, 
                                kwargs={}
                            ),
                            observation=observation,
                            fmt=fmt,
                        )
                        planner_updates.append(step)
                        i += 2
                    else:
                        i += 1
                
                # 检查最后一个步骤是否应该标记为已完成
                if planner_updates and not planner_updates[-1].is_finished:
                    last_step = planner_updates[-1]
                    if (last_step.action 
                        and last_step.action.function_name == _PYTHON_TOOL_NAME):
                        action_code = last_step.action.args[0] if last_step.action.args else ""
                        
                        # 检查显示最后答案的打印( )
                        if "print(" in action_code:
                            final_answer = None
                            if 'print("""' in action_code and '""")' in action_code:
                                final_answer = action_code.split('print("""')[1].rsplit('""")', 1)[0]
                            elif "print('" in action_code and "')" in action_code:
                                final_answer = action_code.split("print('")[1].rsplit("')", 1)[0]
                            elif 'print("' in action_code and '")' in action_code:
                                final_answer = action_code.split('print("')[1].rsplit('")', 1)[0]
                            
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
                
                if planner_updates:
                    exemplars.append(
                        react.ReActState(inputs=problem_description, updates=planner_updates)
                    )
            
            print(f"Loaded {len(exemplars)} exemplars from notebooks")
            return exemplars
        
        def _create_health_domain_exemplars(self) -> List["react.ReActState"]:
            """Create health-domain exemplars simulating notebook-derived examples.
            
            This provides fallback exemplars when notebook-based examples are not
            available. It creates examples with health analysis patterns.
            
            IMPORTANT: Use ArgumentFormat.PYTHON for all steps so that actions render as:
                `function_name('arg1', 'arg2')`
            This matches the [Act]: format that the ReAct parser expects.
            
            Args should be simple tuples like ('arg1', 'arg2'), NOT nested tuples.
            """
            from onetwo.stdlib.tool_use import llm_tool_use
            
            exemplars = []
            
            # 例1:通过搜索实现元素健康摘要
            exemplars.append(react.ReActState(
                inputs="Please summarize my metabolic health based on my blood test data.",
                updates=[
                    react.ReActStep(
                        thought="To provide a comprehensive summary of metabolic health, I need to focus on glucose metabolism markers (fasting glucose, HbA1c, insulin) and also consider liver and kidney health. I'll search for the key biomarkers.",
                        action=llm_tool_use.FunctionCall(
                            function_name="search",
                            args=("blood biomarkers for metabolic health liver kidney",),
                        ),
                        observation="(['https://www.ncbi.nlm.nih.gov/...'], ['Key metabolic biomarkers include fasting glucose, HbA1c, insulin for glucose metabolism; ALT, AST, GGT for liver health; and creatinine, eGFR, BUN for kidney function.'])",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                    react.ReActStep(
                        thought="Now I have the key biomarkers. I should check if the user's values are within normal ranges using the reference range tool.",
                        action=llm_tool_use.FunctionCall(
                            function_name="compare_blood_test_values_with_reference_ranges",
                            args=([[95, 70, 100], [5.0, None, 5.7], [24, 7, 56]],),
                        ),
                        observation="[('Within Clinical Reference Range', 'Within interval'), ('Within Clinical Reference Range', 'Within interval'), ('Within Clinical Reference Range', 'Within interval')]",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                    react.ReActStep(
                        thought="The user's glucose, HbA1c, and ALT are all within normal ranges. I can now provide a comprehensive summary.",
                        is_finished=True,
                        action=llm_tool_use.FunctionCall(
                            function_name="finish",
                            args=("Your metabolic health markers are within normal ranges. Your fasting glucose (95 mg/dL) and HbA1c (5.0%) indicate good glucose control. Your liver enzyme ALT (24 U/L) is also normal, suggesting healthy liver function.",),
                        ),
                        observation="Your metabolic health markers are within normal ranges. Your fasting glucose (95 mg/dL) and HbA1c (5.0%) indicate good glucose control. Your liver enzyme ALT (24 U/L) is also normal, suggesting healthy liver function.",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                ],
            ))
            
            # 示例2:胆固醇解释与搜索
            exemplars.append(react.ReActState(
                inputs="What do my cholesterol levels mean? LDL is 142, HDL is 52, total is 210.",
                updates=[
                    react.ReActStep(
                        thought="I need to search for the recommended cholesterol ranges to interpret these values accurately.",
                        action=llm_tool_use.FunctionCall(
                            function_name="search",
                            args=("cholesterol levels LDL HDL normal range guidelines",),
                        ),
                        observation="(['https://www.heart.org/...'], ['Optimal LDL is below 100 mg/dL. LDL 100-129 is near optimal, 130-159 is borderline high, 160-189 is high. HDL should be 60+ mg/dL for protection. Total cholesterol should be below 200 mg/dL.'])",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                    react.ReActStep(
                        thought="Now I can compare the user's values against these guidelines using the reference range tool.",
                        action=llm_tool_use.FunctionCall(
                            function_name="compare_blood_test_values_with_reference_ranges",
                            args=([[142, None, 100], [52, 60, None], [210, None, 200]],),
                        ),
                        observation="[('Outside of Clinical Reference Range', 'Above upper bound'), ('Outside of Clinical Reference Range', 'Below lower bound'), ('Outside of Clinical Reference Range', 'Above upper bound')]",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                    react.ReActStep(
                        thought="The user's LDL is elevated (borderline high at 142), HDL is below optimal (52 vs 60+), and total cholesterol is slightly elevated. I should provide a clear interpretation.",
                        is_finished=True,
                        action=llm_tool_use.FunctionCall(
                            function_name="finish",
                            args=("Your cholesterol profile shows some areas for attention. Your LDL (142 mg/dL) is in the borderline high range (130-159), your HDL (52 mg/dL) is below the protective level of 60, and your total cholesterol (210 mg/dL) is slightly above the recommended 200. Consider discussing lifestyle modifications with your healthcare provider.",),
                        ),
                        observation="Your cholesterol profile shows some areas for attention. Your LDL (142 mg/dL) is in the borderline high range (130-159), your HDL (52 mg/dL) is below the protective level of 60, and your total cholesterol (210 mg/dL) is slightly above the recommended 200. Consider discussing lifestyle modifications with your healthcare provider.",
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    ),
                ],
            ))
            
            return exemplars
        
        def get_agent(
            self,
            api_key: Optional[str] = None,
            provider: str = "gemini",
            model_name: Optional[str] = None,
            temperature: float = 0.6,
            exemplar_files: Optional[Sequence[str]] = None,
            # 向后兼容的遗留参数
            gemini_api_key: Optional[str] = None,
        ) -> "react.ReActAgent":
            """Configure and return the domain expert agent.
            
            Args:
                api_key: API key for the selected provider.
                provider: "gemini", "openai", or "anthropic" (requires OneTwo).
                model_name: Model to use (uses provider defaults if not specified).
                temperature: Sampling temperature.
                exemplar_files: Optional list of notebook paths for few-shot examples.
                gemini_api_key: DEPRECATED - use api_key instead.
            
            Returns:
                Configured ReActAgent.
            """
            # 处理遗留参数
            if gemini_api_key and not api_key:
                api_key = gemini_api_key
                provider = "gemini"
                
            self.configure_agent_backend(
                api_key=api_key,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
            )
            return self.setup_agent(exemplar_files=exemplar_files)
        
        def call_agent(
            self, 
            prompt: str, 
            verbose: bool = False,
            return_state: bool = False
        ) -> str:
            """Call the domain expert agent with a user prompt.
            
            Args:
                prompt: The user's health question.
                verbose: If True, print debug information.
                return_state: If True, return (answer, state) tuple.
            
            Returns:
                The agent's response (or tuple of response and state if return_state=True).
            """
            if self.domain_expert_agent is None:
                return "Error: Agent not initialized. Call get_agent() first."
            
            # 如果有用户健康数据，请予以补充
            if self.user_phr_data and self.user_phr_data != "\n<USER DATA NOT FOUND>\n":
                full_prompt = (
                    prompt
                    + "\n[User's Personal Health Record]\n"
                    + self.user_phr_data
                )
            else:
                full_prompt = prompt
            
            try:
                if verbose:
                    print(f"[DEBUG] Calling ReAct agent with prompt length: {len(full_prompt)}")
                    print(f"[DEBUG] Max steps: {self.domain_expert_agent.max_steps}")
                
                # 使用 return_final_state = 需要同时获得答案和状态 (PHIA 模式)
                result = ot.run(
                    self.domain_expert_agent(
                        inputs=full_prompt, 
                        return_final_state=True
                    )
                )
                
                # 结果是return_final_state=True时的(答，状态)Tuple
                if isinstance(result, tuple) and len(result) == 2:
                    answer, state = result
                    
                    if verbose:
                        print(f"[DEBUG] Answer type: {type(answer)}")
                        print(f"[DEBUG] State type: {type(state)}")
                        if hasattr(state, 'updates'):
                            print(f"[DEBUG] Number of ReAct steps: {len(state.updates)}")
                            for i, step in enumerate(state.updates):
                                print(f"[DEBUG] Step {i}:")
                                print(f"[DEBUG]   thought={step.thought[:100] if step.thought else 'None'}...")
                                if hasattr(step, 'action') and step.action:
                                    print(f"[DEBUG]   action={step.action.function_name}")
                                if hasattr(step, 'observation') and step.observation:
                                    obs = step.observation[:200] if isinstance(step.observation, str) else str(step.observation)[:200]
                                    print(f"[DEBUG]   observation={obs}...")
                    
                    # 验证并提取正确答案
                    clean_answer = self._extract_clean_answer(answer, state, verbose)
                    
                    if return_state:
                        return clean_answer, state
                    return clean_answer
                else:
                    # 意外结果格式的倒置
                    if verbose:
                        print(f"[DEBUG] Unexpected result format: {type(result)}")
                    return self._validate_response(str(result))
                    
            except Exception as e:
                error_msg = f"Agent execution error: {type(e).__name__}: {e}"
                if verbose:
                    import traceback
                    traceback.print_exc()
                return error_msg
        
        def _extract_clean_answer(self, answer: Any, state: Any, verbose: bool = False) -> str:
            """Extract and validate a clean answer from ReAct agent output.
            
            Args:
                answer: The raw answer from the agent.
                state: The agent state with ReAct steps.
                verbose: If True, print debug info.
                
            Returns:
                A validated, cleaned response.
            """
            # 先试试直接回答
            if answer:
                answer_str = str(answer)
                validated = self._validate_response(answer_str)
                if validated and self._is_valid_response(validated):
                    return validated
            
            # 尝试从完成的步数中提取
            if hasattr(state, 'updates') and state.updates:
                for step in reversed(state.updates):
                    if step.is_finished and step.observation:
                        obs_str = str(step.observation)
                        validated = self._validate_response(obs_str)
                        if validated and self._is_valid_response(validated):
                            return validated
                
                # 寻找实质性意见(工具产出)
                for step in reversed(state.updates):
                    if step.observation:
                        obs_str = str(step.observation)
                        if len(obs_str) > 100 and 'Error' not in obs_str:
                            validated = self._validate_response(obs_str)
                            if validated and self._is_valid_response(validated):
                                return validated
            
            # 退缩——表面什么出了问题
            n_steps = len(state.updates) if hasattr(state, 'updates') and state.updates else 0
            step_summary = ""
            if hasattr(state, 'updates') and state.updates:
                parts = []
                for i, step in enumerate(state.updates):
                    thought = (step.thought[:80] + "...") if step.thought and len(step.thought) > 80 else (step.thought or "None")
                    obs = str(step.observation)[:80] if step.observation else "None"
                    parts.append(f"  Step {i}: thought=`{thought}` | obs=`{obs}`")
                step_summary = "\n".join(parts)
            
            # 打印完整原始答案(不切换)因此后端错误
            # OpenAI/Anthropic/Gemini可诊断. 用户信息解析。
            # 继续修剪。
            print(f"[DE:extract] All extraction attempts failed. Steps={n_steps}, answer={repr(str(answer))}")
            return (
                f"⚠️ **Domain Expert could not extract a valid answer** from the ReAct agent output.\n\n"
                f"**Steps executed:** {n_steps}\n"
                f"**Raw answer:** `{str(answer)[:200] if answer else 'None'}`\n\n"
                f"This usually means the model produced incomplete or malformed ReAct output. "
                f"Try rephrasing your question or switching models."
            )
        
        def _validate_response(self, text: str) -> str:
            """Validate and clean a response string.
            
            Args:
                text: Raw response text.
                
            Returns:
                Cleaned text or empty string if invalid.
            """
            if not text:
                return ""
            
            # 删除 ReAct 标记
            import re
            for marker in ['[Thought]:', '[Act]:', '[Observe]:', '[Finish]:']:
                if text.startswith(marker):
                    text = text[len(marker):].strip()
            
            # 删除代码块
            text = re.sub(r'```[\w]*\n.*?```', '', text, flags=re.DOTALL)
            
            # 删除内线 ReAct 标记
            for marker in ['[Act]:', '[Thought]:', '[Observe]:']:
                if marker in text:
                    text = text.split(marker)[0].strip()
            
            return text.strip()
        
        def _is_valid_response(self, text: str) -> bool:
            """Check if a response is valid (not prompt leakage, not incomplete).
            
            Args:
                text: Response text to validate.
                
            Returns:
                True if the response is valid.
            """
            if not text or len(text) < 30:
                return False
            
            text_lower = text.lower()
            
            # 快速渗漏指标
            leakage_indicators = [
                'you are tasked with',
                'you must do so safely',
                'you should use the provided tools',
                'as a domain expert',
                'you are a domain expert',
                '你具备内科、健康和运动医学知识，能够解释个人健康数据',
                'clinically accurate, evidence-based',
                'explicitly separate objective findings',
                'wearables, labs, imaging, notes',
            ]
            for indicator in leakage_indicators:
                if indicator in text_lower:
                    return False
            
            # 答复指标不完整
            # 去除反引号和多余空格，再匹配常见的无效输出。
            text_normalized = text_lower.replace('`', '').replace('  ', ' ')
            
            incomplete_indicators = [
                '[act]:', '[thought]:', '```tool_code', '```python',
                'i need to', 'i will', 'let me', "i'll",
                'i need your', 'i need the user',  # 规划报表
                'then use search', 'then use web', 'then use tool',
                'from summary_df', 'from activities_df',
                'and compute the', 'and calculate the',
                'tool_code(', 'search(',
                '#error#',
            ]
            for indicator in incomplete_indicators:
                if indicator in text_normalized:
                    return False
            
            # 检查从“我需要”开始的规划报表
            if text_normalized.strip().startswith('i need'):
                return False
            
            # 检测原始搜索结果被返回而不是合成答案
            url_count = text_normalized.count('url:')
            relevance_count = text_normalized.count('relevance score:')
            if url_count >= 2 or relevance_count >= 2:
                return False
            
            return True
            
            return True
        
        def configure_agent_backend(
            self,
            api_key: Optional[str] = None,
            provider: str = "gemini",
            model_name: Optional[str] = None,
            temperature: float = 0.6,
            # 向后兼容的遗留参数
            gemini_api_key: Optional[str] = None,
        ) -> None:
            """Configure the LLM backend.
            
            Args:
                api_key: API key for the selected provider.
                provider: "gemini", "openai", or "anthropic" (requires OneTwo).
                model_name: Model name (uses provider defaults if not specified).
                temperature: Sampling temperature.
                gemini_api_key: DEPRECATED - use api_key instead.
            """
            # 处理遗留参数
            if gemini_api_key and not api_key:
                api_key = gemini_api_key
                provider = "gemini"
            
            # 更新/从全球配置读取
            from ..llm.config import configure_global, get_config
            
            # 如果提供了明确的参数，请更新全局配置
            if api_key or provider != "gemini":  # 粗略检查用户是否尝试配置
                # 注意：DE 的有效供应商有限，但我们还是更新全局
                # 默认提供商到基因组时，如果不通过则由参数默认处理，
                # 但是，如果我们在这里，我们可能想要尊重全球配置，如果参数是空的？
                pass

            # 如果我们有新值，更新全局配置
            configure_global(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
            )
            
            # 现在读回已完全解析的配置
            # 但是，configure_global如果我们通过Nones,不覆盖？
            # 实际上configure_global将过值视为覆数。
            # 如果我们通过了本地参数，将更新全球配置。
            
            config = get_config()
            # 如果参数为无，请使用配置值
            # (注：configure_global通过尝试从env加载，在内部处理无逻辑，
            # 但如果我们没有通过 API 密钥和内存是空的，详细解析发生在这里).
            
            provider = config.provider
            api_key = config.api_key
            model_name = config.model_name
            
            # Anthropic现在通过我们的OneTwo - 兼容
            # AnthropicAPI包装机(pha/llm/onetwo_anthropic.py).
            # 建筑分公司低于每个供应商的发送量。

            if not api_key:
                # 在尝试即时处理前先检查一次
                env_vars = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
                raise ValueError(
                    f"{provider.title()} API key required. Set {env_vars.get(provider, 'API_KEY')} "
                    f"environment variable."
                )
            
            # 基于提供者配置后端
            if provider == "gemini":
                # 必要时为 Gemini 添加模型/前缀
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                if not _ONETWO_GENAI_AVAILABLE:
                    raise ImportError(
                        "onetwo's GoogleGenAIAPI backend not available. "
                        "Update onetwo: pip install -U git+https://github.com/google-deepmind/onetwo"
                    )
                
                print(f"[DomainExpert:configure] Using GoogleGenAIAPI backend: {model_name}")
                backend = google_genai_api.GoogleGenAIAPI(
                    api_key=api_key,
                    generate_model_name=model_name,
                    # Pin 聊天模式也——OneTwo 验证两者;默认为
                    # `models/gemini-2.5-flash` 并不是所有的键有。
                    chat_model_name=model_name,
                    temperature=temperature,
                    threadpool_size=1,
                )
                backend.register()
            elif provider == "openai":
                if not _ONETWO_OPENAI_AVAILABLE:
                    raise ValueError(
                        "OneTwo OpenAI backend not available. "
                        "Please update onetwo or use Gemini models for ReAct functionality."
                    )
                # 原因/GPT-5模型只接受温度=1.0(
                # API 默认值。使用这个——它是最接近于
                # 0.7,我们用于Gemini/Anthropic.
                if model_name.startswith(("o1", "o3", "o4", "gpt-5")):
                    temperature = 1.0
                backend = openai_api.OpenAIAPI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    batch_size=1,
                )
                backend.register()
            elif provider == "anthropic":
                from pha.llm.onetwo_anthropic import AnthropicAPI, is_available
                if not is_available():
                    raise ImportError(
                        "Anthropic ReAct backend requires both the `anthropic` "
                        "SDK and onetwo. Install: pip install anthropic"
                    )
                print(f"[DomainExpert:configure] Using AnthropicAPI backend: {model_name}")
                backend = AnthropicAPI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                )
                backend.register()
            else:
                raise ValueError(
                    f"Provider '{provider}' not supported for ReAct agent. "
                    f"Use 'gemini', 'openai', or 'anthropic'."
                )
        
        def get_jinja_preamble_from_string(
            self,
            preamble_string: str = CONTEXTUAL_REASONING_PREAMBLE_V1,
        ) -> "react.ReActPromptProtocol":
            """Returns the preamble for the agent."""
            return react.ReActPromptJ2(text=preamble_string)
        
        def set_user_health_data(self, data: str) -> None:
            """Set user health data as markdown string.
            
            Args:
                data: Health data in markdown format.
            """
            self.user_phr_data = data
        
        def set_user_health_data_from_df(self, df: pd.DataFrame) -> None:
            """Set user health data from DataFrame.
            
            Args:
                df: DataFrame with health records.
            """
            self.user_phr_data = df.to_markdown()
        
        def get_llm_toolbox(self) -> List[llm_tool_use.Tool]:
            """Returns the list of tools available to the agent."""
            return [
                self._setup_search_tool(),
                self._setup_datacommons_tool(),
                self._setup_reference_range_tool(),
                self._setup_finish_tool(),
            ]
        
        def _setup_search_tool(self) -> llm_tool_use.Tool:
            """Set up the web search tool."""
            
            def search_tool(
                query: str,
                max_results: int = 3,
            ) -> tuple:
                """Search the web for health information.
                
                Args:
                    query: Search query.
                    max_results: Maximum results to return.
                
                Returns:
                    Tuple of (list of URLs, list of snippets).
                """
                _log_tool_call("search", {"query": query, "max_results": max_results})
                
                urls = []
                snippets = []
                error_msg = None
                
                if self.search_backend == "tavily":
                    # 使用 Tavily (API 键已在 setup_agent 中验证).
                    try:
                        from tavily import TavilyClient
                        api_key = self.tavily_api_key or os.environ.get("TAVILY_API_KEY")
                        client = TavilyClient(api_key=api_key)
                        response = client.search(
                            query=query,
                            max_results=max_results,
                            include_domains=[
                                "mayoclinic.org", "cdc.gov", "who.int",
                                "nih.gov", "clevelandclinic.org",
                            ],
                        )
                        for item in response.get("results", []):
                            urls.append(item.get("url", ""))
                            snippets.append(item.get("content", ""))
                    except ImportError:
                        error_msg = "tavily package not installed"
                        snippets.append("Error: tavily package not installed. Run: pip install tavily-python")
                    except Exception as e:
                        error_msg = str(e)
                        snippets.append(f"Tavily search error: {e}")
                else:
                    # 使用 DuckDuckGo (美国英语).
                    try:
                        try:
                            from ddgs import DDGS
                        except ImportError:
                            from duckduckgo_search import DDGS
                        
                        ddgs = DDGS()
                        results = ddgs.text(query, max_results=max_results)
                        for r in results:
                            urls.append(r.get("href", ""))
                            snippets.append(r.get("body", ""))
                    except ImportError:
                        error_msg = "duckduckgo-search package not installed"
                        snippets.append("Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search")
                    except Exception as e:
                        error_msg = str(e)
                        snippets.append(f"DuckDuckGo search error: {e}")
                
                # 日志结果
                if error_msg:
                    _log_tool_call("search", {"query": query}, error=error_msg)
                else:
                    _log_tool_call("search", {"query": query}, output=f"Found {len(urls)} results: {urls[:2]}...")
                
                return urls, snippets
            
            return llm_tool_use.Tool(
                name="search",
                function=search_tool,
                description=(
                    "API for performing a web search, which provides a list of URLs "
                    "and relevant snippets for the given query. Given a search query, "
                    "this tool finds authoritative health information with URLs for citation."
                ),
                example=ORGANIC_SEARCH_TOOL_EXAMPLE,
                color="blue",
            )
        
        def _setup_datacommons_tool(self) -> llm_tool_use.Tool:
            """Set up the Data Commons query tool.
            
            Uses DataCommonsClient from pha.tools.web_search for querying
            public health statistics from authoritative sources.
            """
            from ..tools.web_search import DataCommonsClient
            
            # 创建客户端实例
            dc_client = DataCommonsClient()
            
            def datacommons_natural_language_query(query: str) -> dict:
                """Query Data Commons for public health statistics.
                
                Data Commons provides access to public datasets from authoritative
                sources including CDC, WHO, Census Bureau, and more.
                
                Args:
                    query: Natural language query about statistics.
                        Examples:
                        - "What is the most common cause of death in the US?"
                        - "What is the average life expectancy in California?"
                        - "How many people have diabetes in the US?"
                
                Returns:
                    JSON response with statistical data including:
                    - charts: Visualizations with data
                    - dataCsv: Raw data in CSV format
                    - srcs: Source citations (CDC, WHO, etc.)
                    - title: Description of the data
                """
                _log_tool_call("datacommons_natural_language_query", {"query": query})
                
                try:
                    result = dc_client.query(query)
                    # 汇总记录结果
                    if isinstance(result, dict):
                        title = result.get("title", "No title")
                        _log_tool_call("datacommons_natural_language_query", {"query": query}, 
                                      output=f"Title: {title}")
                    else:
                        _log_tool_call("datacommons_natural_language_query", {"query": query}, 
                                      output=str(result)[:200])
                    return result
                except Exception as e:
                    _log_tool_call("datacommons_natural_language_query", {"query": query}, error=str(e))
                    return {"error": str(e)}
            
            return llm_tool_use.Tool(
                name="datacommons_natural_language_query",
                function=datacommons_natural_language_query,
                description=(
                    "API for querying DataCommons. Given a natural language query, "
                    "this tool queries datacommons to find accurate statistics "
                    "from authoritative sources like CDC, WHO, and Census Bureau. "
                    "Use this for population-level health statistics, mortality data, "
                    "disease prevalence, and demographic health information."
                ),
                example=DC_EXAMPLE_RESPONSE,
                color="blue",
            )
        
        def _setup_python_sandbox_tool(self) -> llm_tool_use.Tool:
            """Set up the Python code execution tool."""
            
            # 使用 Python 调用格式来匹配 ReAct [法案]:期望
            # NOT 标记下代码块，混淆解析器
            python_sandbox_example = textwrap.dedent("""\
              tool_code("1 + 1") returns "2".
              tool_code("70 * 2.205") returns "154.35" (kg to lbs).
              tool_code("round(8156/10000*100, 1)") returns "81.6" (percentage).
              NOTE: No imports allowed. No access to summary_df or pandas.""")
            
            async def run_python(code: str) -> str:
                """Execute Python code.
                
                Args:
                    code: Python code to execute.
                
                Returns:
                    Execution result as string.
                """
                _log_tool_call("tool_code", {"code": code[:100] + "..." if len(code) > 100 else code})
                
                result = None
                error_msg = None
                
                # 先试两瓶沙盒
                try:
                    from onetwo.stdlib.code_execution import python_sandbox
                    sandbox = python_sandbox.PythonSandboxSafeSubset()
                    async with sandbox.start() as sb:
                        result = await sb.run(code)
                        result = str(result)
                        _log_tool_call("tool_code", {"code": "..."}, output=result[:200] if len(result) > 200 else result)
                        return result
                except Exception:
                    pass
                
                # 回落到限制执行
                try:
                    import io
                    import contextlib
                    import math
                    
                    # 为基本计算提供安全的内建结构
                    safe_builtins = {
                        'abs': abs, 'all': all, 'any': any, 'bool': bool,
                        'dict': dict, 'enumerate': enumerate, 'filter': filter,
                        'float': float, 'int': int, 'len': len, 'list': list,
                        'map': map, 'max': max, 'min': min, 'print': print,
                        'range': range, 'round': round, 'set': set, 'sorted': sorted,
                        'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
                        'True': True, 'False': False, 'None': None,
                    }
                    
                    output = io.StringIO()
                    local_vars = {}
                    
                    with contextlib.redirect_stdout(output):
                        exec(code, {"__builtins__": safe_builtins, "math": math}, local_vars)
                    
                    result = output.getvalue()
                    if not result and local_vars:
                        # 返回上次指定的变量
                        result = str(list(local_vars.values())[-1])
                    result = result or "Code executed successfully"
                    _log_tool_call("tool_code", {"code": "..."}, output=result[:200] if len(str(result)) > 200 else result)
                    return result
                except Exception as e:
                    error_msg = str(e)
                    _log_tool_call("tool_code", {"code": "..."}, error=error_msg)
                    return f"Error: {e}"
            
            return llm_tool_use.Tool(
                name="tool_code",
                function=run_python,
                description=(
                    "Simple Python calculator for unit conversions, percentages, and basic math. "
                    "NOTE: Does NOT have access to summary_df, pandas, or user health data. "
                    "Use only for simple arithmetic like '70 * 2.205' or 'round(8156/10000*100, 1)'."
                ),
                example=python_sandbox_example,
                color="blue",
            )
        
        def _setup_reference_range_tool(self) -> llm_tool_use.Tool:
            """Set up the clinical reference range comparison tool."""
            
            check_interval_example = textwrap.dedent("""
                compare_blood_test_values_with_reference_ranges([[24, 4, 36]])
                # returns [('Within Clinical Reference Range', 'Within interval')]
            """)
            
            async def compare_blood_test_values_with_reference_ranges(
                values_and_intervals: List[tuple],
            ) -> List[tuple]:
                """Check if values are within clinical reference ranges.
                
                Args:
                    values_and_intervals: List of [value, lower_bound, upper_bound].
                        Use None for one-sided intervals.
                
                Returns:
                    List of (status, proximity) tuples.
                """
                _log_tool_call("compare_blood_test_values_with_reference_ranges", 
                              {"values_and_intervals": str(values_and_intervals)[:100]})
                
                try:
                    result = check_reference_ranges(values_and_intervals)
                    _log_tool_call("compare_blood_test_values_with_reference_ranges", 
                                  {"values": "..."}, output=str(result))
                    return result
                except Exception as e:
                    _log_tool_call("compare_blood_test_values_with_reference_ranges", 
                                  {"values": "..."}, error=str(e))
                    raise
            
            return llm_tool_use.Tool(
                name="compare_blood_test_values_with_reference_ranges",
                function=compare_blood_test_values_with_reference_ranges,
                description=(
                    "Tool for checking if lab values are within clinical reference "
                    "ranges. Input: [[value, lower_bound, upper_bound], ...]. "
                    "Use None for one-sided intervals. Returns status and proximity."
                ),
                example=check_interval_example,
                color="blue",
            )
        
        def _setup_finish_tool(self) -> llm_tool_use.Tool:
            """Set up the finish tool for returning final answers."""
            return llm_tool_use.Tool(
                name="finish",
                function=lambda x: x,
                description="Function for returning the final answer.",
            )

else:
    # 在无法提供1至2个时退后执行
    
    class DomainExpertAgent:
        """Domain Expert Agent (fallback without onetwo).
        
        This is a simplified implementation for when onetwo is not installed.
        For full ReAct functionality, install onetwo:
            pip install git+https://github.com/google-deepmind/onetwo
        """
        
        def __init__(
            self,
            preamble: Optional[str] = None,
            context_memory: str = "",
            search_backend: str = "tavily",
            tavily_api_key: Optional[str] = None,
        ):
            """Initialize the Domain Expert Agent (fallback mode)."""
            self.system_prompt = preamble
            self.context_memory = context_memory
            self.user_phr_data: Optional[str] = None
            self.search_backend = search_backend
            self.tavily_api_key = tavily_api_key
            self._backend = None
            self._is_configured = False
        
        def get_agent(
            self,
            api_key: Optional[str] = None,
            provider: Optional[str] = None,
            model_name: Optional[str] = None,
            temperature: float = 0.6,
            # 向后兼容的遗留参数
            gemini_api_key: Optional[str] = None,
        ) -> "DomainExpertAgent":
            """Configure the agent backend.
            
            Note: This is a fallback without full ReAct support.
            Install onetwo for full functionality.
            
            Args:
                api_key: API key for the selected provider.
                provider: "gemini", "openai", or "anthropic".
                model_name: Model to use (uses provider defaults if not specified).
                temperature: Sampling temperature.
                gemini_api_key: DEPRECATED - use api_key instead.
            """
            from ..llm import get_llm_backend
            
            # 处理遗留参数
            if gemini_api_key and not api_key:
                api_key = gemini_api_key
                if not provider:
                    provider = "gemini"
            
            # 如果未指定提供者，请从全局配置中获取
            from ..llm.config import get_config
            config = get_config()
            if not provider:
                provider = config.provider
            
            self._backend = get_llm_backend(
                backend_type=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
            )
            self._is_configured = True
            return self
        
        def call_agent(self, prompt: str) -> str:
            """Call the agent with a prompt (simplified, no ReAct)."""
            from datetime import datetime
            from pha.streaming import emit_step
            ts = datetime.now().strftime("%H:%M:%S")
            emit_step("DE", "领域专家 Agent 正在解读问题…")
            print(f"\n[DE:{ts}] call_agent: {prompt[:100]}...")
            
            if not self._is_configured:
                print(f"[DE:{ts}] *** ERROR: Agent not configured")
                return "Error: Agent not configured. Call get_agent() first."
            
            # 无 ReAct 支持时，将可用的健康背景与问题直接交给模型。
            full_prompt = f"""你是内科和健康领域专家。给出全面、个性化的健康信息，解释医学术语并说明临床参考范围。
使用用户最新一轮发言的语言，除非用户另有要求。

"""
            if self.user_phr_data:
                full_prompt += f"User's Health Data:\n{self.user_phr_data}\n\n"
            
            full_prompt += f"Question: {prompt}"
            
            print(f"[DE:{ts}] Full prompt: {len(full_prompt)} chars, has_health_data={bool(self.user_phr_data)}")
            result = self._backend.generate(full_prompt)
            print(f"[DE:{ts}] Response ({len(result)} chars): {result[:200]}...")
            return result
        
        def set_user_health_data(self, data: str) -> None:
            """Set user health data."""
            self.user_phr_data = data
        
        def set_user_health_data_from_df(self, df: pd.DataFrame) -> None:
            """Set user health data from DataFrame."""
            self.user_phr_data = df.to_markdown()


def is_react_available() -> bool:
    """Check if onetwo ReAct is available."""
    return _ONETWO_AVAILABLE

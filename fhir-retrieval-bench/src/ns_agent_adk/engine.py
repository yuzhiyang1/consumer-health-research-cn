"""Graph-Aligned Clinical Reasoning: Budget-Constrained Agent Implementation."""

import datetime
import os
from typing import Any, Literal

from absl import logging
from google.adk import runners
# from google.adk.planners import built_in_planner
from google.adk.agents import llm_agent
from google.adk.agents.context import Context
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import in_memory_session_service
from google.adk.tools import function_tool
from google.genai import types
import jinja2

from ns_agent_adk import tools as tools_module
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.core import linearizer as linearizer_module


def parse_events(events: list[Any]) -> dict[str, Any]:
    """
    Parses a list of Agent Event objects and returns step-by-step interaction statistics.
    Includes a mathematical validation check to ensure cumulative token context adds up.
    """
    stats = {
        "aggregate_summary": {
            "total_events": len(events),
            "total_tools_invoked": 0,
            "peak_cumulative_tokens": 0,
            "total_duration_seconds": 0.0,
            "context_validation": {
                "initial_prompt_tokens": 0,
                "total_tokens_added_mid_conversation": 0,
                "final_prompt_tokens": 0,
                "is_math_valid": False
            }
        },
        "steps": []
    }

    if not events:
        return stats

    # 计算总期限
    start_time = events[0].timestamp
    end_time = events[-1].timestamp
    stats["aggregate_summary"]["total_duration_seconds"] = round(end_time - start_time, 2)

    # 跟踪环境增长的变量
    previous_prompt_count = 0
    initial_prompt_count = 0
    total_growth = 0

    for i, event in enumerate(events):
        content = getattr(event, 'content', None)
        role = getattr(content, 'role', None) if content else None
        
        step_data = {
            "step_number": i + 1,
            "role": role,
            "prompt_tokens_this_step": 0,
            "context_growth_from_last_turn": 0,  # 三角洲
            "tokens_generated_this_step": 0,
            "cumulative_context_size": 0,
            "tools_called": [],
            "tool_outputs": [],
            "is_text_response": False
        }

        # 1. 提取 Token Usage(通常只在"模版"转弯时出现)
        usage = getattr(event, 'usage_metadata', None)
        if usage:
            current_prompt_count = getattr(usage, 'prompt_token_count', 0)
            total_tokens = getattr(usage, 'total_token_count', 0)
            
            step_data["prompt_tokens_this_step"] = current_prompt_count
            step_data["cumulative_context_size"] = total_tokens
            
            # --- CONTEXT VALIDATION LOGIC ---
            if initial_prompt_count == 0:
                # 这是第一个模式转弯
                initial_prompt_count = current_prompt_count
            else:
                # 计算自上次模式转向以来新增了多少个符号
                growth = current_prompt_count - previous_prompt_count
                step_data["context_growth_from_last_turn"] = growth
                total_growth += growth
                
            previous_prompt_count = current_prompt_count

            # 音轨峰值符
            if total_tokens > stats["aggregate_summary"]["peak_cumulative_tokens"]:
                stats["aggregate_summary"]["peak_cumulative_tokens"] = total_tokens

            # 此步骤中土生土长的托肯
            candidates = getattr(usage, 'candidates_token_count', 0)
            thoughts = getattr(usage, 'thoughts_token_count', 0)
            step_data["tokens_generated_this_step"] = candidates + (thoughts if thoughts else 0)

        # 2. 提取工具调用，响应，和文本
        parts = getattr(content, 'parts', []) if content else []
        for part in parts:
            func_call = getattr(part, 'function_call', None)
            if func_call and getattr(func_call, 'name', None):
                step_data["tools_called"].append(func_call.name)
            
            func_response = getattr(part, 'function_response', None)
            if func_response:
                step_data["tool_outputs"].append({
                    "tool_name": getattr(func_response, 'name', 'unknown'),
                    "output": getattr(func_response, 'response', None)
                })

            if getattr(part, 'text', None):
                step_data["is_text_response"] = True

        stats["aggregate_summary"]["total_tools_invoked"] += len(step_data["tools_called"])
        stats["steps"].append(step_data)

    # --- FINAL VALIDATION CHECK ---
    # 初始 基线 + 随时间推移添加的所有块上下文 MUST 等于最终的快速大小。
    final_prompt_count = previous_prompt_count
    is_valid = (initial_prompt_count + total_growth == final_prompt_count)
    
    stats["aggregate_summary"]["context_validation"] = {
        "initial_prompt_tokens": initial_prompt_count,
        "total_tokens_added_mid_conversation": total_growth,
        "final_prompt_tokens": final_prompt_count,
        "is_math_valid": is_valid
    }

    return stats

def generate_budgeted_linearization(
    graph: graph_module.ChronologicalHypergraph,
    query: str,
    config: Any,
    precomputed_embeddings=None,
    anchor_date: datetime.date | datetime.datetime | None = None,
) -> str:
  """Constructs a context view that GUARANTEES fitting in the window."""
  linearizer = linearizer_module.CausalSaliencyLinearizer(
      graph,
      config,
      node_embeddings=precomputed_embeddings,
      anchor_date=anchor_date,
  )
  return linearizer.linearize(query)


def _load_preamble(
    user_context: list[str],
    current_time: datetime.date | datetime.datetime,
    max_llm_calls: int,
    enabled_tools: list[str] | None = None,
) -> str:
  """Loads and renders the Jinja prompt template."""
  templates_dir = os.path.join(os.path.dirname(__file__), "templates")
  env = jinja2.Environment(
      loader=jinja2.FileSystemLoader(searchpath=templates_dir)
  )
  template = env.get_template("system_preamble.jinja2")

  if user_context:
    context_str = "\n".join(user_context)
  else:
    context_str = ""

  if current_time:
    if current_time > (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=3 * 365)
    ) or current_time < (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=3 * 365)
    ):
      context_str += f"\nThe current time is {current_time}."
    else:
      context_str += (
          f"\nThe current time is {current_time}. Note that the dates are"
          " shifted for privacy."
      )
  logging.info("user context: %s", context_str)
  return template.render(
      context_str=context_str,
      max_llm_calls=max_llm_calls,
      enabled_tools=enabled_tools,
  )


def _build_graph_agent(
    graph: graph_module.ChronologicalHypergraph,
    config: Any,
    embedder=None,
    node_embeddings=None,
    anchor_date: datetime.date | datetime.datetime | None = None,
    user_context: list[str] | None = None,
    max_llm_calls: int = 15,
) -> llm_agent.Agent:
  """Constructs the Budget-Constrained ADK Agent."""

  tools_handler = tools_module.ClinicalGraphTools(
      graph, embedder=embedder, node_embeddings=node_embeddings
  )
  all_tools_map = {
      "inspect_node": function_tool.FunctionTool(tools_handler.inspect_node),
      "search_graph": function_tool.FunctionTool(tools_handler.search_graph),
      "follow_links": function_tool.FunctionTool(tools_handler.follow_links),
      "filter_graph_by_time": function_tool.FunctionTool(
          tools_handler.filter_graph_by_time
      ),
  }
  # 以配置为基础动态地编写 ADK FunctionTools
  tools = [all_tools_map[tool_name] for tool_name in config.enabled_tools]
  if isinstance(anchor_date, datetime.datetime):
    current_time = anchor_date
  elif isinstance(anchor_date, datetime.date):
    current_time = datetime.datetime.combine(
        anchor_date, datetime.time.min
    ).replace(tzinfo=datetime.timezone.utc)
  else:
    current_time = None

  rendered_preamble = _load_preamble(
      user_context=user_context,
      current_time=current_time,
      max_llm_calls=max_llm_calls,
      enabled_tools=config.enabled_tools,
  )

  invocation_call_counts = {}

  def graceful_termination_callback(callback_context: Context, llm_request: LlmRequest):
    if max_llm_calls is None:
      return None
    count = invocation_call_counts.get(callback_context.invocation_id, 0)
    logging.info(f"DEBUG: LLM call count: {count}")

    if count == max_llm_calls - 1:
      warning_text = (
          "\n\nSYSTEM_WARNING: You have reached your maximum reasoning budget ("
          f"{max_llm_calls} steps). You can no longer call tools. "
          "You MUST output your best final answer or summary based on the "
          "information gathered so far."
      )
      last_content = llm_request.contents[-1] if llm_request.contents else None
      if last_content and last_content.role == "user":
        last_content.parts.append(types.Part.from_text(text=warning_text))
      else:
        llm_request.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=warning_text)]
            )
        )

      if llm_request.config.tool_config is None:
        llm_request.config.tool_config = types.ToolConfig()
      llm_request.config.tool_config.function_calling_config = types.FunctionCallingConfig(
          mode="NONE"
      )
    elif count >= max_llm_calls:
      return LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text="I'm sorry, I've reached my maximum reasoning steps and cannot explore further. Please feel free to ask a more specific question."
                  )
              ]
          )
      )

    invocation_call_counts[callback_context.invocation_id] = count + 1
    return None

  # thinking_config = types.ThinkingConfig(
  #     include_thoughts=True, thinking_budget=2000
  # )

  raw_thinking_cfg = getattr(config, "llm_thinking_config", None)
  thinking_config = None
  if raw_thinking_cfg is not None:
    thinking_budget_val = None
    thinking_level = None
    if isinstance(raw_thinking_cfg, int):
      thinking_budget_val = raw_thinking_cfg
    elif isinstance(raw_thinking_cfg, str):
      thinking_level = types.ThinkingLevel(raw_thinking_cfg.upper())

    thinking_config = types.ThinkingConfig(
        thinking_budget=thinking_budget_val,
        thinking_level=thinking_level,
    )

  cfg_kwargs = {}
  if config.llm_temperature is not None:
    cfg_kwargs["temperature"] = config.llm_temperature
  if thinking_config is not None:
    cfg_kwargs["thinking_config"] = thinking_config

  generate_content_config = types.GenerateContentConfig(**cfg_kwargs)

  return llm_agent.Agent(
      name="NeuroSymbolicAgent",
      model=config.get_llm_model(),
      instruction=rendered_preamble,
      tools=tools,
      before_model_callback=graceful_termination_callback,
      generate_content_config=generate_content_config,
      # 以防我们想让设计师思考：
      # planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
  )


class NeuroSymbolicAgent:
  """Wrapper class for running the ADK agent."""

  def __init__(
      self,
      config: Any,
      graph: graph_module.ChronologicalHypergraph,
      precomputed_node_embeddings=None,
      anchor_date: (
          datetime.date
          | datetime.datetime
          | Literal["latest_resource", "real_today"]
          | None
      ) = "latest_resource",
      user_context: list[str] | None = None,
  ):
    self.config = config
    self.graph = graph
    self.precomputed_node_embeddings = precomputed_node_embeddings
    self.user_context = user_context

    # 决断 anchor_date
    if anchor_date == "latest_resource":
      max_ts_dt = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
      for node in graph.get_all_nodes():
        if node.timestamp and node.timestamp > max_ts_dt:
          max_ts_dt = node.timestamp
      self.anchor_date = max_ts_dt
    elif anchor_date == "real_today":
      self.anchor_date = datetime.datetime.now(datetime.timezone.utc)
    else:
      self.anchor_date = anchor_date

    # 证明代理商
    self.agent = _build_graph_agent(
        graph,
        config,
        embedder=config.get_embedder()
        if hasattr(config, "get_embedder")
        else None,
        node_embeddings=self.precomputed_node_embeddings,
        anchor_date=self.anchor_date,
        user_context=self.user_context,
        max_llm_calls=self.config.max_agent_steps,
    )

  def execute(self, query: str, verbose: bool = False):
    """Executes the agent and returns final response and the ADK event trace."""

    skeleton_view = generate_budgeted_linearization(
        self.graph,
        query,
        self.config,
        self.precomputed_node_embeddings,
        anchor_date=self.anchor_date,
    )

    full_prompt = (
        "=== BUDGETED SKELETON (TOP HITS) ===\n"
        f"{skeleton_view}\n\n"
        "=== USER QUERY ===\n"
        f"{query}\n\n"
        "Begin reasoning:"
    )

    if verbose:
      print(f"DEBUG: Prompt Length: {len(full_prompt)} chars")

    # 创建 ADK 运行器
    runner = runners.Runner(
        agent=self.agent,
        app_name="fhir_neuro_symbolic_app",
        session_service=in_memory_session_service.InMemorySessionService(),
        auto_create_session=True,
    )

    # 在同步模拟器循环中运行代理并收集事件。
    # 我们用隐含的跑步方法 来避免Ayncio Colab循环冲突
    events = []

    # 我们使用静态的ID来进行简单和演示。
    # 在全网应用中，后端会提供独特的会话IDs.
    run_config = RunConfig(max_llm_calls=self.config.max_agent_steps) if self.config.max_agent_steps is not None else None
    for event in runner.run(
        user_id="fhir_agent_user",
        session_id="fhir_agent_session",
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=full_prompt)]
        ),
        run_config=run_config,
    ):
      events.append(event)

    # 来自事件流的总结性最后回应。最后一个事件可能是
    # 空动作事件。
    final_response = ""
    for event in reversed(events):
      if (
          getattr(event, "author", None) == self.agent.name
          and event.content
          and event.content.parts
      ):
        text_parts = [
            p.text
            for p in event.content.parts
            if p.text and not getattr(p, "thought", False)
        ]
        if text_parts:
          final_response = "".join(text_parts)
          break
    print("final_response", final_response)
    # 返回反应和全部互动追踪
    # 可视化器 UI
    return final_response, events

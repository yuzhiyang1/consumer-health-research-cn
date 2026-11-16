"""Multi-Agent Orchestrator for PHA.

This orchestrator coordinates between Data Science, Domain Expert, and Health Coach
agents to provide comprehensive health insights.

Usage:
    from pha.agents import MultiAgentOrchestrator
    
    orchestrator = MultiAgentOrchestrator()
    orchestrator.configure(gemini_api_key="your-key")
    orchestrator.set_agents(
        data_science_agent=ds_agent,
        domain_expert_agent=de_agent,
        health_coach_agent=hc_agent,
    )
    
    response = orchestrator.respond("How is my sleep quality?")
"""

# 调试版本
__version__ = "1.9.0"  # 添加了请求结束摘要

import copy
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# 导入共享请求数据
from ..utils.request_stats import request_stats


# =============================================================================
# 代理呼叫日志
# =============================================================================

def _log_agent_call(agent_name: str, query: str = None, response: str = None, error: str = None, flag_full_content: bool = False):
    """Log an agent call to the terminal for debugging.
    
    Args:
        agent_name: Name of the agent being called.
        query: The query being sent to the agent.
        response: The agent's response (will be truncated).
        error: Error message if the call failed.
        flag_full_content: If True, log the full content.
    """
    # 记录数据
    request_stats.record_agent(agent_name)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n{'═' * 60}")
    print(f"🤖 [{timestamp}] AGENT CALL: {agent_name}")
    
    if query:
        if flag_full_content:
            print(f"   Query: {query}")
        else:
            query_preview = query[:150] + "..." if len(query) > 150 else query
            print(f"   Query: {query_preview}")
    
    if error:
        if flag_full_content:
            print(f"   Error: {error}")
        else:
            error_preview = error[:200] + "..." if len(error) > 200 else error
            print(f"   Error: {error_preview}")
    elif response is not None:
        if flag_full_content:
            print(f"   ✓ Response: {response}")
        else:
            response_preview = response[:200] + "..." if len(response) > 200 else response
            print(f"   ✓ Response: {response_preview}")
    
    print(f"{'═' * 60}\n")


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
    - Agent name references (e.g., "The Data Science agent found...")
    - Internal insight markers (e.g., [DATA_SCIENCE_AGENT_INSIGHTS])
    
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
    
    # 删除内部代理名称引用( 用户不应该看到执行细节)
    # 图案："数据科学 Agent...","根据domain_expert_agent......"等。
    agent_ref_patterns = [
        # 裂缝特工洞察标记(必须精确匹配，大小写不敏感)
        r'\[DATA_SCIENCE_AGENT_INSIGHTS\]\s*\n?',
        r'\[DOMAIN_EXPERT_AGENT_INSIGHTS\]\s*\n?',
        r'\[HEALTH_COACH_AGENT_INSIGHTS\]\s*\n?',
        r'\[SUPPORTING_AGENT_INSIGHTS\]\s*\n?',
        # 带有各种结构的完整短语 - 贪婪匹配以捕捉续作
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Data Science [Aa]gent(?:\'s)?(?:\s+(?:analysis|findings|calculations?|results?|data|insights?))?(?:\s+(?:found|reported|calculated|determined|analyzed|shows?|indicates?|suggests?|reveals?))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Domain Expert [Aa]gent(?:\'s)?(?:\s+(?:analysis|findings|assessment|interpretation|insights?))?(?:\s+(?:found|reported|determined|analyzed|shows?|indicates?|suggests?|reveals?))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Health Coach [Aa]gent(?:\'s)?(?:\s+(?:recommendations?|suggestions?|advice|guidance|insights?))?(?:\s+(?:found|reported|suggests?|recommends?|advises?))?(?:\s+that)?[,:]?\s*',
        # Snake_case版本 互联网档案馆的存檔，存档日期2014-09-02.
        r'(?:The |According to (?:the )?|Based on (?:the )?)?data_science_agent(?:\'s)?(?:\s+(?:analysis|findings))?(?:\s+(?:found|reported|calculated))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?domain_expert_agent(?:\'s)?(?:\s+(?:analysis|findings))?(?:\s+(?:found|reported))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?health_coach_agent(?:\'s)?(?:\s+(?:recommendations?|suggestions?))?(?:\s+(?:found|reported))?(?:\s+that)?[,:]?\s*',
    ]
    for pattern in agent_ref_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 清理任何残存的孔隙/白空间文物
    text = re.sub(r'^\s*[,;:]\s*', '', text)  # 删除主标
    # 只规范水平白空间( 空格/tabs), 保留新线以标记下移
    text = re.sub(r'[^\S\n]+', ' ', text)  # 折叠空格/tabs但保留 \n
    text = re.sub(r' *\n *', '\n', text)    # 新线周围的空格
    text = re.sub(r'\n{3,}', '\n\n', text)  # 2 的连续空白行
    
    # 如果第一个字母是小写，请大写
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
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
            # 记录终端调试的漏水情况，但也会在响应中显示
            print(f"[clean_agent_response] PROMPT LEAK detected: matched '{indicator}'")
            print(f"[clean_agent_response] Raw text (first 300 chars): {text[:300]}")
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


# 尝试导入 LLM 后端
_GENAI_AVAILABLE = False
try:
    import google.genai as genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    pass

from ..prompts.orchestrator_prompts import (
    AGENT_NAME_DATA_SCIENCE,
    AGENT_NAME_DOMAIN_EXPERT,
    AGENT_NAME_HEALTH_COACH,
    ORCHESTRATOR_PREAMBLE,
    ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT,
    TEAM_STRUCTURE_PROMPT,
    TOPIC_PROMPT,
    GOAL_PROMPT,
    FINISH_DESC_PROMPT,
    REPHRASE_PROMPT_SUPPORTING_AGENTS,
    SUPPORTING_AGENT_ADDITIONAL_PROMPT,
    UPDATE_TEAM_AGENT_INSIGHTS_PROMPT_TEMPLATE,
    REFLECT_PROMPT,
    FINAL_RESPONSE_PROMPT,
)


class MultiAgentOrchestrator:
    """Multi-Agent Orchestrator for PHA.
    
    Coordinates between Data Science, Domain Expert, and Health Coach agents
    to provide comprehensive, personalized health insights.
    
    The orchestrator:
    1. Determines which agents should handle a query
    2. Calls supporting agents (potentially in parallel)
    3. Synthesizes insights through the main agent
    4. Optionally reflects and refines the response
    """
    
    def __init__(
        self,
        name: str = "Personal Health Insights Agent Team",
        system_prompt: Optional[str] = None,
        debug_verbose: bool = False,
    ):
        """Initialize the orchestrator.
        
        Args:
            name: Name for the orchestrator.
            system_prompt: Custom system prompt (uses default if None).
            debug_verbose: If True, print debug information.
        """
        self.name = name
        self.system_prompt = system_prompt or ORCHESTRATOR_PREAMBLE
        self.debug_verbose = debug_verbose
        
        # 代理参考
        self.data_science_agent = None
        self.domain_expert_agent = None
        self.health_coach_agent = None
        self.multi_agent_team: Dict[str, Any] = {}
        self.agent_name_list: List[str] = []
        
        # 团队结构状态
        self.main_agent = ""
        self.supporting_agents = ""
        self.supporting_agents_complete = ""
        self.collaboration_workflow = ""
        
        # 对话状态
        self.conv_context = ""
        self.conv_utterances = ""
        self.question_last = ""
        
        # 内存状态
        self.memory = {
            "topic": "",
            "goal": "",
            "finish_desc": "",
            "data_science_agent_insights": "",
            "domain_expert_agent_insights": "",
            "health_coach_agent_insights": "",
        }
        
        # 反射状态
        self.flag_reflection_mode = False
        self.reflection_round_counter = 0
        self.reflection_round_upper_bound = 1
        self.rephrase_prompt_record: Dict[str, List[str]] = {}
        
        # 点数
        self.log_number_of_turns = 0
        
        # LLM 配置
        self._client = None
        self._model_name = "models/gemini-3.1-flash-lite-preview"
        self._temperature = 0.7
    
    def configure(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        # 向后兼容的遗留参数
        gemini_api_key: Optional[str] = None,
    ) -> "MultiAgentOrchestrator":
        """Configure the LLM backend.
        
        Args:
            api_key: API key for the selected provider.
            provider: "gemini", "openai", or "anthropic".
            model_name: Model to use (uses provider defaults if not specified).
            temperature: Sampling temperature.
            gemini_api_key: DEPRECATED - use api_key instead.
            
        Returns:
            Self for chaining.
        """
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
            
        # 更新全球配置
        from ..llm.config import configure_global
        configure_global(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        
        # 初始化后端
        from ..llm.backend_multimodel import get_llm_backend
        self._backend = get_llm_backend(
            backend_type=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        
        return self
    
    def set_agents(
        self,
        data_science_agent=None,
        domain_expert_agent=None,
        health_coach_agent=None,
    ) -> "MultiAgentOrchestrator":
        """Set the agent team.
        
        Args:
            data_science_agent: DataScienceAgent instance.
            domain_expert_agent: DomainExpertAgent instance.
            health_coach_agent: HealthCoachAgent instance.
            
        Returns:
            Self for chaining.
        """
        self.multi_agent_team = {}
        
        if data_science_agent is not None:
            self.data_science_agent = data_science_agent
            self.multi_agent_team[AGENT_NAME_DATA_SCIENCE] = data_science_agent
        
        if domain_expert_agent is not None:
            self.domain_expert_agent = domain_expert_agent
            self.multi_agent_team[AGENT_NAME_DOMAIN_EXPERT] = domain_expert_agent
        
        if health_coach_agent is not None:
            self.health_coach_agent = health_coach_agent
            self.multi_agent_team[AGENT_NAME_HEALTH_COACH] = health_coach_agent
            # 为导师保存初始系统提示
            self._health_coach_initial_prompt = copy.deepcopy(
                health_coach_agent.system_prompt
            )
        
        self.agent_name_list = list(self.multi_agent_team.keys())
        
        # 总是记录注册的代理
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"🤖 [{timestamp}] ORCHESTRATOR AGENTS REGISTERED")
        print(f"   Available agents: {self.agent_name_list}")
        print(f"   data_science_agent: {'✓' if data_science_agent else '✗'}")
        print(f"   domain_expert_agent: {'✓' if domain_expert_agent else '✗'}")
        print(f"   health_coach_agent: {'✓' if health_coach_agent else '✗'}")
        print(f"{'═' * 60}\n")
        
        return self
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the orchestrator's LLM.
        
        Args:
            prompt: The prompt to send.
            max_retries: Maximum number of retries.
            
        Returns:
            Model response text.
        """
        for attempt in range(max_retries):
            try:
                if self._backend:
                    return self._backend.generate(prompt)
                else:
                    return "Error: LLM not configured. Call configure() first."
            except Exception as e:
                if self.debug_verbose:
                    print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    traceback.print_exc()
                    return f"Error: {e}"
        
        return "Error: All retry attempts failed."
    
    def _clear_json_format(self, text: str) -> str:
        """Clean JSON response from LLM.
        
        Args:
            text: Raw LLM response.
            
        Returns:
            Cleaned JSON string.
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def get_conversation_history(self) -> str:
        """Get the full conversation history."""
        return self.conv_context + "\n" + self.conv_utterances
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation."""
        self.conv_utterances += f"\nUser: {message}"
        self.question_last = message
    
    def add_response(self, response: str) -> None:
        """Add an orchestrator response to the conversation."""
        self.conv_utterances += f"\nAssistant: {response}"
    
    def _update_memory(self) -> None:
        """Update memory states based on conversation.
        
        Runs topic, goal, and finish_desc extraction in parallel since
        they are independent of each other.
        """
        conv_history = self.get_conversation_history()
        
        # 所有3提取都是独立的——并行运行。
        with ThreadPoolExecutor(max_workers=3) as executor:
            topic_future = executor.submit(self._call_llm, TOPIC_PROMPT + conv_history)
            goal_future = executor.submit(self._call_llm, GOAL_PROMPT + conv_history)
            finish_future = executor.submit(self._call_llm, FINISH_DESC_PROMPT + conv_history)
            
            self.memory["topic"] = topic_future.result()
            self.memory["goal"] = goal_future.result()
            self.memory["finish_desc"] = finish_future.result()
    
    def _update_agent_team_structure(self) -> None:
        """Determine which agents should handle the current query."""
        prompt = (
            self.system_prompt + "\n\n" +
            TEAM_STRUCTURE_PROMPT +
            "\n[CONVERSATION]\n" + self.get_conversation_history() +
            "\n[LAST_QUESTION]\n" + self.question_last +
            "\n[TOPIC]\n" + self.memory.get("topic", "")
        )
        
        for attempt in range(5):
            response = self._call_llm(prompt)
            try:
                # 记录调试的原始响应
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Team structure LLM response (attempt {attempt+1}):")
                print(f"   {response[:300]}..." if len(response) > 300 else f"   {response}")
                
                response = self._clear_json_format(response)
                result = json.loads(response)
                
                # 注意： dict.get( 密钥，默认) 仅在返回时返回 `default`
                # 键缺失，而不是值为“”时。一些模型(特别是)
                # Claude Sonnet 4.5)偶尔会返回空代理字符串，
                # 之前短路 整个团队，并做了
                # 编排器回归无上下文LLM呼叫。
                raw_main = (result.get("main_agent") or "").strip()
                if raw_main and raw_main in self.agent_name_list:
                    self.main_agent = raw_main
                else:
                    # 没有选择有效的主要代理——如果有的话使用健康导师
                    # (一般是用户发音),否则第一个
                    # 可用代理服务器。
                    self.main_agent = (
                        AGENT_NAME_HEALTH_COACH
                        if AGENT_NAME_HEALTH_COACH in self.agent_name_list
                        else (self.agent_name_list[0] if self.agent_name_list else "")
                    )

                raw_supporting = (result.get("supporting_agents") or "").strip()
                if raw_supporting:
                    self.supporting_agents = ";".join([
                        a for a in raw_supporting.split(";")
                        if a and a != self.main_agent and a in self.agent_name_list
                    ])
                else:
                    # 空支持列表与成功选中的列表对齐
                    # 主干员没事，但如果模型空空归来
                    # 两者都通过 其它所有可用的代理作为支持
                    # 我们不会默默失去数据访问(例如data_science).
                    if not raw_main:
                        self.supporting_agents = ";".join([
                            a for a in self.agent_name_list if a != self.main_agent
                        ])
                    else:
                        self.supporting_agents = ""

                self.collaboration_workflow = result.get("collaboration_workflow", "")

                # 调试时总是日志团队结构
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n{'═' * 60}")
                print(f"📋 [{timestamp}] TEAM STRUCTURE DECISION")
                print(f"   Main Agent: {self.main_agent}")
                print(f"   Supporting Agents: {self.supporting_agents or '(none)'}")
                print(f"   Workflow: {self.collaboration_workflow[:100]}..." if len(self.collaboration_workflow) > 100 else f"   Workflow: {self.collaboration_workflow}")
                print(f"{'═' * 60}\n")
                
                break
            except Exception as e:
                if attempt == 4:
                    # 倒计时:保健教练为主，其他人为辅助
                    self.main_agent = AGENT_NAME_HEALTH_COACH if AGENT_NAME_HEALTH_COACH in self.agent_name_list else self.agent_name_list[0]
                    self.supporting_agents = ";".join([a for a in self.agent_name_list if a != self.main_agent])
                    self.collaboration_workflow = "Supporting agents provide insights, main agent responds."
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n{'═' * 60}")
                    print(f"📋 [{timestamp}] TEAM STRUCTURE (FALLBACK)")
                    print(f"   Main Agent: {self.main_agent}")
                    print(f"   Supporting Agents: {self.supporting_agents}")
                    print(f"   Error: {str(e)[:100]}")
                    print(f"{'═' * 60}\n")
    
    def _log_rephrase_prompt_record(self, phase: str) -> None:
        """Log the current rephrase prompt record for debugging."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"📝 [{timestamp}] REPHRASE PROMPT RECORD ({phase})")
        if not self.rephrase_prompt_record:
            print("   (empty)")
        else:
            for agent_name, prompts in self.rephrase_prompt_record.items():
                prompt_preview = prompts[-1] if prompts else ""
                prompt_preview = (
                    prompt_preview[:200] + "..."
                    if len(prompt_preview) > 200
                    else prompt_preview
                )
                print(f"   {agent_name}: {prompt_preview}")
        print(f"{'═' * 60}\n")

    def _build_supporting_agent_rephrase_prompts(
        self,
        supporting_agent_list: List[str],
    ) -> None:
        """Build rephrased prompts for supporting agents using orchestrator LLM."""
        self.rephrase_prompt_record = {}

        rephrase_prompt = REPHRASE_PROMPT_SUPPORTING_AGENTS(
            original_prompt=self.question_last,
            main_agent=self.main_agent,
            supporting_agents=self.supporting_agents,
            collaboration_workflow=self.collaboration_workflow,
        )

        rephrase_dict: Dict[str, Any] = {}
        for attempt in range(3):
            response = self._call_llm(rephrase_prompt)
            try:
                parsed = json.loads(self._clear_json_format(response))
                if isinstance(parsed, dict):
                    rephrase_dict = parsed
                    break
            except Exception:
                if self.debug_verbose:
                    print(f"Rephrase parsing failed (attempt {attempt + 1}/3)")
                continue

        for agent_name in supporting_agent_list:
            # 保持原始数据科学，以分析忠诚性。
            if agent_name == AGENT_NAME_DATA_SCIENCE:
                self.rephrase_prompt_record[agent_name] = [self.question_last]
                continue

            rephrase_question = str(rephrase_dict.get(agent_name, "") or "").strip()
            if rephrase_question:
                prompt = (
                    f"Answer the user's raw question: {self.question_last}\n\n"
                    f"In addition, focus specifically on: {rephrase_question}"
                )
            else:
                prompt = self.question_last
            self.rephrase_prompt_record[agent_name] = [prompt]

        self._log_rephrase_prompt_record("initial")

    @staticmethod
    def _is_error_like_payload(text: str) -> bool:
        """Return True if text looks like technical error output."""
        if not text:
            return True
        normalized = text.strip().lower()
        if not normalized:
            return True
        error_signals = ("error:", "traceback", "exception:", "failed:", "technical issue", "⚠️", "stack trace")
        for err in error_signals:
            if normalized.startswith(err):
                return True
        return False

    def _merge_and_store_agent_insights(self, agent_name: str, new_insights: str) -> None:
        """Merge new insight text into persisted per-agent insights."""
        memory_key = f"{agent_name}_insights"
        old_insights = str(self.memory.get(memory_key, "") or "").strip()
        cleaned_new = clean_agent_response(str(new_insights or "")).strip()

        # 忽略辅助剂的空或技术错误有效载荷。
        if self._is_error_like_payload(cleaned_new):
            return

        agent_token = agent_name.upper()
        update_prompt = (
            UPDATE_TEAM_AGENT_INSIGHTS_PROMPT_TEMPLATE(agent_name)
            + f"\n[{agent_token}_INSIGHTS]\n{old_insights}"
            + f"\n[NEW_{agent_token}_INSIGHTS]\n{cleaned_new}"
        )

        merged_response = clean_agent_response(self._call_llm(update_prompt)).strip()
        if not merged_response:
            # 如果合并调用不当，请保留旧的见解。
            return

        if re.search(r"\bEMPTY\b", merged_response.upper()):
            self.memory[memory_key] = ""
            return

        if self._is_error_like_payload(merged_response):
            # 防范合并失败污染内存。
            if not old_insights:
                self.memory[memory_key] = cleaned_new
            return

        self.memory[memory_key] = merged_response

    def _build_supporting_agent_input_prompt(self, agent_name: str, question_prompt: str) -> str:
        """Build a supporting-agent prompt with existing insights context."""
        prior_insights = str(self.memory.get(f"{agent_name}_insights", "") or "").strip()
        additional_context = SUPPORTING_AGENT_ADDITIONAL_PROMPT(agent_name).strip()
        agent_token = agent_name.upper()

        prompt_parts = []
        if additional_context:
            prompt_parts.append(additional_context)

        if prior_insights:
            prompt_parts.append(
                "You have previously produced insights for this conversation. "
                "Reuse and extend them instead of repeating the same analysis."
            )
            prompt_parts.append(f"[{agent_token}_INSIGHTS]\n{prior_insights}")

        prompt_parts.append(f"[LAST_QUESTION]\n{question_prompt}")
        return "\n\n".join(prompt_parts)

    def _call_supporting_agents(
        self,
        parallel: bool = True,
        flag_rephrase_prompt: bool = True,
    ) -> None:
        """Call supporting agents to gather insights.
        
        Args:
            parallel: If True, call agents in parallel.
            flag_rephrase_prompt: If True, regenerate prompts for supporting agents.
        """
        if not self.supporting_agents:
            return
        
        supporting_agent_list = [a for a in self.supporting_agents.split(";") if a]
        if flag_rephrase_prompt:
            self._build_supporting_agent_rephrase_prompts(supporting_agent_list)
        
        if self.debug_verbose:
            print(f"Calling supporting agents: {supporting_agent_list}")
        
        def call_agent(agent_name: str) -> Tuple[str, str]:
            """Call a single supporting agent."""
            try:
                question_prompt = self.rephrase_prompt_record.get(agent_name, [self.question_last])[-1]
                if isinstance(question_prompt, str) and question_prompt.startswith("reflection_questions::"):
                    follow_up = question_prompt.split("reflection_questions::", 1)[-1].strip()
                    question_prompt = (
                        f"Answer the user's raw question: {self.question_last}\n\n"
                        f"Additional follow-up requested by orchestrator: {follow_up}"
                    )
                prompt = self._build_supporting_agent_input_prompt(agent_name, question_prompt)
                _log_agent_call(agent_name, query=prompt, flag_full_content=True)
                
                if agent_name == AGENT_NAME_DATA_SCIENCE and self.data_science_agent:
                    result = self.data_science_agent.query(prompt)
                    # _log_agent_call(agent_name, response=str(result)[:500])
                    # 为调试目的日志完整响应
                    _log_agent_call(agent_name, response=str(result), flag_full_content=True)
                    return agent_name, str(result)
                
                elif agent_name == AGENT_NAME_DOMAIN_EXPERT and self.domain_expert_agent:
                    result = self.domain_expert_agent.call_agent(prompt)
                    _log_agent_call(agent_name, response=str(result)[:500])
                    return agent_name, str(result)
                
                elif agent_name == AGENT_NAME_HEALTH_COACH and self.health_coach_agent:
                    # 医疗教练作为支援，只要得到一个回应
                    result = self.health_coach_agent.respond(prompt)
                    _log_agent_call(agent_name, response=str(result)[:500])
                    return agent_name, str(result)
                
                return agent_name, ""
            
            except Exception as e:
                _log_agent_call(agent_name, error=str(e))
                if self.debug_verbose:
                    print(f"Error calling {agent_name}: {e}")
                return agent_name, f"Error: {e}"
        
        if parallel and len(supporting_agent_list) > 1:
            # 并行呼叫代理
            with ThreadPoolExecutor(max_workers=len(supporting_agent_list)) as executor:
                futures = {executor.submit(call_agent, name): name for name in supporting_agent_list}
                for future in as_completed(futures):
                    agent_name, result = future.result()
                    self._merge_and_store_agent_insights(agent_name, result)
        else:
            # 按顺序拨打
            for agent_name in supporting_agent_list:
                _, result = call_agent(agent_name)
                self._merge_and_store_agent_insights(agent_name, result)
    
    def _accumulate_supporting_agents_complete(self) -> None:
        """Update the running union of all supporting agents consulted so far."""
        if self.supporting_agents:
            if self.supporting_agents_complete:
                existing = set(self.supporting_agents_complete.split(";"))
                new = set(self.supporting_agents.split(";"))
                self.supporting_agents_complete = ";".join(existing | new)
            else:
                self.supporting_agents_complete = self.supporting_agents
    
    def _organize_supporting_agent_insights(self) -> str:
        """Organize all supporting agent insights into a formatted string."""
        insights = []

        allowed_agents = [
            a.strip()
            for a in str(self.supporting_agents_complete or "").split(";")
            if a.strip()
        ]

        if AGENT_NAME_DATA_SCIENCE in allowed_agents and self.memory.get("data_science_agent_insights"):
            insights.append(
                f"[DATA_SCIENCE_AGENT_INSIGHTS]\n{self.memory['data_science_agent_insights']}"
            )

        if AGENT_NAME_DOMAIN_EXPERT in allowed_agents and self.memory.get("domain_expert_agent_insights"):
            insights.append(
                f"[DOMAIN_EXPERT_AGENT_INSIGHTS]\n{self.memory['domain_expert_agent_insights']}"
            )

        if AGENT_NAME_HEALTH_COACH in allowed_agents and self.memory.get("health_coach_agent_insights"):
            insights.append(
                f"[HEALTH_COACH_AGENT_INSIGHTS]\n{self.memory['health_coach_agent_insights']}"
            )

        return "\n\n".join(insights)
    
    def _call_main_agent(self) -> Tuple[str, str]:
        """Call the main agent with supporting insights.
        
        Returns:
            Tuple of (main agent response, supporting insights).
        """
        supporting_insights = self._organize_supporting_agent_insights()
        
        if self.main_agent == AGENT_NAME_HEALTH_COACH and self.health_coach_agent:
            # 保健教练作为主要代理人，从他人那里进行分析
            _log_agent_call(f"{self.main_agent} (MAIN)", query=self.question_last)
            if supporting_insights:
                response = self.health_coach_agent.respond_with_analysis(
                    self.question_last,
                    supporting_insights
                )
            else:
                response = self.health_coach_agent.respond(self.question_last)
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            return response, supporting_insights
        
        elif self.main_agent == AGENT_NAME_DOMAIN_EXPERT and self.domain_expert_agent:
            # 主要从事合成工作的域专家
            prompt = self.question_last
            if supporting_insights:
                prompt += f"\n\n[SUPPORTING_AGENT_INSIGHTS]\n{supporting_insights}"
            _log_agent_call(f"{self.main_agent} (MAIN)", query=prompt[:300])
            response = self.domain_expert_agent.call_agent(prompt)
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            return str(response), supporting_insights
        
        elif self.main_agent == AGENT_NAME_DATA_SCIENCE and self.data_science_agent:
            # 数据科学为主
            _log_agent_call(f"{self.main_agent} (MAIN)", query=self.question_last)
            response = self.data_science_agent.query(self.question_last)
            # _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            # 为调试目的日志完整响应
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response), flag_full_content=True)
            return str(response), supporting_insights
        
        else:
            # 倒计时:使用 编排器 LLM (中文(简体) ).
            prompt = FINAL_RESPONSE_PROMPT(
                self.main_agent,
                self.supporting_agents,
                self.supporting_agents_complete,
                self.collaboration_workflow
            )
            prompt += f"\n[CONVERSATION]\n{self.get_conversation_history()}"
            prompt += f"\n[SUPPORTING_AGENT_INSIGHTS]\n{supporting_insights}"
            prompt += f"\n[MAIN_AGENT_RESPONSE]\n(Generate response)"
            
            response = self._call_llm(prompt)
            return response, supporting_insights
    
    def take_turn(self) -> str:
        """Take a single conversation turn.
        
        This is the main orchestration method that:
        1. Updates memory states
        2. Determines agent team structure
        3. Calls supporting agents
        4. Calls main agent
        5. Reflects and optionally refines (catches questions answerable by data)
        
        Returns:
            The orchestrated response.
        """
        from pha.streaming import emit_step
        
        self.log_number_of_turns += 1
        self.reflection_round_counter = 0
        self.flag_reflection_mode = False
        self.rephrase_prompt_record = {}
        
        # === Parallel memory + overlapped team structure ===
        # 主题，目标，finish_desc是独立的LLM呼叫。
        # 团队结构只需要话题，所以我们一开始
        # 专题解决——与仍在运行的目标/最终呼号重叠。
        emit_step("Orch", "Analyzing query and updating memory...")
        conv_history = self.get_conversation_history()
        
        with ThreadPoolExecutor(max_workers=3) as mem_executor:
            topic_future = mem_executor.submit(
                self._call_llm, TOPIC_PROMPT + conv_history
            )
            goal_future = mem_executor.submit(
                self._call_llm, GOAL_PROMPT + conv_history
            )
            finish_future = mem_executor.submit(
                self._call_llm, FINISH_DESC_PROMPT + conv_history
            )
            
            # 主题块(按团队结构要求)
            self.memory["topic"] = topic_future.result()
            
            if len(self.agent_name_list) > 1:
                # 重叠:在目标/完成仍在计算时确定团队结构
                emit_step("Orch", "Determining which agents to use...")
                self._update_agent_team_structure()
            
            # 收集剩余内存结果(可能已经完成)
            self.memory["goal"] = goal_future.result()
            self.memory["finish_desc"] = finish_future.result()
        
        if len(self.agent_name_list) > 1:
            emit_step("Orch", f"Team: main={self.main_agent}, supporting={self.supporting_agents or 'none'}")
            
            # 呼叫辅助人员
            if self.supporting_agents:
                emit_step("Orch", f"Calling supporting agents: {self.supporting_agents}")
            self._call_supporting_agents(flag_rephrase_prompt=True)
            self._accumulate_supporting_agents_complete()
            
            # 调用主代理
            emit_step("Orch", f"Calling main agent: {self.main_agent}")
            main_response, supporting_insights = self._call_main_agent()

            # 坚持主剂输出作为该剂不断发展的洞察力记忆。
            # 这保留了同一剂日后成为辅助剂时的连续性。
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)
            
            # 在最后答复之前反映(渔获物问题按数据回答)
            emit_step("Orch", "Reflecting on response quality...")
            reflection_result = self._reflect_before_final_response(
                main_response, supporting_insights
            )
            
            # 如果反思发现需要改进，请再次拨打支助代理
            if reflection_result.get("decision") == "YES":
                self.flag_reflection_mode = True
                self.reflection_round_counter += 1
                
                if self.reflection_round_counter <= self.reflection_round_upper_bound:
                    emit_step("Orch", "Reflection found improvements needed — re-querying agents...")
                    # 更新带有反射问题的提示
                    for agent_name, questions in reflection_result.get("reflection_questions", {}).items():
                        if isinstance(questions, list):
                            questions = ". ".join(questions)
                        if agent_name not in self.rephrase_prompt_record:
                            self.rephrase_prompt_record[agent_name] = [self.question_last]
                        self.rephrase_prompt_record[agent_name].append(
                            f"reflection_questions::{questions}"
                        )
                    self._log_rephrase_prompt_record("reflection update")
                    
                    # 再给支援人员打电话 问些新问题
                    self._call_supporting_agents(flag_rephrase_prompt=False)
                    self._accumulate_supporting_agents_complete()
                    
                    # 再给主干员打个电话 说出新的见解
                    main_response, supporting_insights = self._call_main_agent()
                    if self.main_agent in self.agent_name_list:
                        self._merge_and_store_agent_insights(self.main_agent, main_response)
            
            emit_step("Orch", "Composing final response...")
        
        elif len(self.agent_name_list) == 1:
            # 单一代理模式
            self.main_agent = self.agent_name_list[0]
            emit_step("Orch", f"Single-agent mode: calling {self.main_agent}")
            main_response, _ = self._call_main_agent()
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)
        
        else:
            # 没有配置代理 - 直接使用 编排器
            main_response = self._call_llm(
                self.system_prompt + 
                "\n[CONVERSATION]\n" + self.get_conversation_history() +
                "\n\nRespond to the user's question: " + self.question_last
            )
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)


        main_response_new = self._call_llm(
            ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT +
            "\n[CONVERSATION]\n" + self.get_conversation_history() +
            "\n\nRespond to the user's question: " + self.question_last +
            "\n[FINAL_RESPONSE]\n" + main_response
        )
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"[Orch:{ts}] updating main response from '{main_response}'")
        print(f"\n{'═' * 10}")
        print(f"[Orch:{ts}] to new main response: '{main_response_new}'")
        print(f"\n{'═' * 60}")
        main_response = main_response_new

        # 检查完成
        if "FINISH" in self.memory.get("finish_desc", ""):
            return main_response + "\n\n I look forward to speaking with you again soon!"
        
        return main_response
    
    def _reflect_before_final_response(
        self,
        main_agent_response: str,
        supporting_agent_insights: str,
    ) -> Dict[str, Any]:
        """Reflect on whether the response can be improved by additional agent queries.
        
        This is critical for catching when the health coach asks questions that
        can be answered by the data science agent from the user's data.
        
        Args:
            main_agent_response: Response from the main agent.
            supporting_agent_insights: Insights from supporting agents.
            
        Returns:
            Dictionary with 'decision' (YES/NO) and 'reflection_questions'.
        """
        from ..prompts.orchestrator_prompts import (
            REFLECT_PROMPT, 
            AGENT_NAME_DATA_SCIENCE,
        )
        
        reflection_results = {
            "decision": "NO",
            "reflection_questions": {},
        }
        
        # 如果没有辅助代理，则跳过
        if not self.supporting_agents:
            return reflection_results
        
        # 如果 数据科学 Agent 不属于辅助代理，请跳过
        # (反应主要抓住数据可回答的问题)
        if AGENT_NAME_DATA_SCIENCE not in self.supporting_agents:
            return reflection_results
        
        reflect_prompt = REFLECT_PROMPT(
            original_prompt=self.question_last,
            main_agent=self.main_agent,
            supporting_agents=self.supporting_agents,
            collaboration_workflow=self.collaboration_workflow,
            main_agent_response=main_agent_response,
            supporting_agent_insights=supporting_agent_insights,
        )
        
        trial_count = 0
        while trial_count < 3:
            trial_count += 1
            try:
                reflect_response = self._call_llm(reflect_prompt)
                reflect_response = self._clear_json_format(reflect_response)
                reflection_results = json.loads(reflect_response)
                
                if "YES" in reflection_results.get("decision", ""):
                    # 验证针对已知代理人的反思问题
                    reflection_agents = list(reflection_results.get("reflection_questions", {}).keys())
                    if not set(reflection_agents).issubset(set(self.agent_name_list)):
                        # 引用无效的代理，跳过反射
                        reflection_results = {"decision": "NO", "reflection_questions": {}}
                    else:
                        # 更新下一轮的辅助代理
                        self.supporting_agents = ";".join(reflection_agents)
                break
            except Exception as e:
                if trial_count >= 3:
                    reflection_results = {"decision": "NO", "reflection_questions": {}}
                continue
        
        return reflection_results
    
    def respond(self, user_message: str, max_retries: int = 2) -> str:
        """Convenience method for single-turn response.
        
        Args:
            user_message: The user's message.
            max_retries: Maximum number of retries for problematic responses.
            
        Returns:
            The orchestrated response (cleaned of internal labels).
        """
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"[Orch:{ts}] ═══ Orchestrator.respond ═══")
        print(f"[Orch:{ts}] Query: {user_message[:100]}...")
        print(f"[Orch:{ts}] Agents: {self.agent_name_list}")
        print(f"{'═' * 60}")
        
        # 重置此请求的数据
        request_stats.reset()
        
        self.add_user_message(user_message)
        
        for attempt in range(max_retries):
            print(f"\n[Orch:{ts}] --- Attempt {attempt+1}/{max_retries} ---")
            response = self.take_turn()
            
            print(f"[Orch:{ts}] take_turn returned ({len(response) if response else 0} chars): {repr(response[:200] if response else 'None')}...")
            
            # 清理前检查有问题的反应模式
            response_lower = response.lower() if response else ""
            
            # 检测即时泄漏(模式呼应指示)
            prompt_leak_patterns = [
                'you must do so safely',
                'you will analyze and synthesize',
                'with the goal of delivering',
                'clinically accurate, evidence-based',
                'you are a domain expert',
            ]
            is_prompt_leak = any(p in response_lower for p in prompt_leak_patterns)
            
            # 检测原始搜索结果(未合成)
            is_raw_search = response_lower.count('url:') >= 2 or response_lower.count('relevance score:') >= 2
            
            if is_prompt_leak:
                print(f"[Orch:{ts}] *** PROMPT LEAK DETECTED in response")
            if is_raw_search:
                print(f"[Orch:{ts}] *** RAW SEARCH RESULTS detected in response")
            
            if (is_prompt_leak or is_raw_search) and attempt < max_retries - 1:
                print(f"[Orch:{ts}] Retrying due to problematic response...")
                
                # 添加明确的综合请求
                synthesis_prompt = (
                    "Please provide a direct, synthesized answer to the user's question. "
                    "Do not output raw data or search results. "
                    "Summarize the key findings in natural language."
                )
                
                # 原始搜索结果请尝试合成
                if is_raw_search and self.domain_expert_agent:
                    try:
                        synthesis_request = f"Based on these search results, answer the user's question: {user_message}\n\nSearch results:\n{response[:2000]}\n\n{synthesis_prompt}"
                        synthesized = self.domain_expert_agent.call_agent(synthesis_request)
                        if synthesized and len(synthesized) > 50:
                            cleaned = clean_agent_response(synthesized)
                            if cleaned and "⚠️" not in cleaned:
                                self.add_response(cleaned)
                                print(f"[Orch:{ts}] ✓ Synthesized response ({len(cleaned)} chars)")
                                request_stats.print_summary()
                                return cleaned
                    except Exception as e:
                        print(f"[Orch:{ts}] Synthesis failed: {e}")

                # 处理在原始搜索+域-专家路径以外的问题案件。
                # 保持现有逻辑上方不变，并增加退后修补通行证。
                if is_prompt_leak:
                    try:
                        fallback_request = (
                            f"User question: {user_message}\n\n"
                            f"Draft response (may contain leaked prompt or raw output):'''\n{response}'''\n\n"
                            f"{synthesis_prompt}\n"
                            "Do not include role or instruction text in the final answer."
                        )
                        repaired = self._call_llm(fallback_request)
                        if repaired and len(repaired) > 50:
                            cleaned_repaired = clean_agent_response(str(repaired))
                            if cleaned_repaired and "⚠️" not in cleaned_repaired:
                                self.add_response(cleaned_repaired)
                                print(f"[Orch:{ts}] ✓ Fallback repaired response ({len(cleaned_repaired)} chars)")
                                request_stats.print_summary()
                                return cleaned_repaired
                    except Exception as e:
                        print(f"[Orch:{ts}] Fallback repair failed: {e}")
                
                continue
            
            self.add_response(response)
            cleaned = clean_agent_response(response)
            print(f"[Orch:{ts}] ✓ Final response ({len(cleaned)} chars): {cleaned[:200]}...")
            request_stats.print_summary()
            return cleaned
        
        # 如果全部重试失败，请返回上次清理的响应
        cleaned = clean_agent_response(response)
        print(f"[Orch:{ts}] ✗ All retries exhausted, returning last response ({len(cleaned)} chars)")
        request_stats.print_summary()
        return cleaned
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conv_context = ""
        self.conv_utterances = ""
        self.question_last = ""
        self.memory = {k: "" for k in self.memory}
        self.main_agent = ""
        self.supporting_agents = ""
        self.supporting_agents_complete = ""
        self.collaboration_workflow = ""
        self.log_number_of_turns = 0
        self.rephrase_prompt_record = {}
        
        # 重置健康导师对话(如果有的话)
        if self.health_coach_agent:
            self.health_coach_agent.reset_conversation()
    
    def get_team_structure(self) -> Dict[str, str]:
        """Get the current agent team structure."""
        return {
            "main_agent": self.main_agent,
            "supporting_agents": self.supporting_agents,
            "collaboration_workflow": self.collaboration_workflow,
        }
    
    def get_agent_insights(self) -> Dict[str, str]:
        """Get insights from each agent."""
        return {
            "data_science": self.memory.get("data_science_agent_insights", ""),
            "domain_expert": self.memory.get("domain_expert_agent_insights", ""),
            "health_coach": self.memory.get("health_coach_agent_insights", ""),
        }


def create_orchestrator(
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    data_science_agent=None,
    domain_expert_agent=None,
    health_coach_agent=None,
    debug_verbose: bool = False,
    # 向后兼容的遗留参数
    gemini_api_key: Optional[str] = None,
) -> MultiAgentOrchestrator:
    """Factory function to create a configured orchestrator.
    
    Args:
        api_key: API key for the selected provider.
        provider: "gemini", "openai", or "anthropic".
        model_name: Model to use (uses provider defaults if not specified).
        data_science_agent: DataScienceAgent instance.
        domain_expert_agent: DomainExpertAgent instance.
        health_coach_agent: HealthCoachAgent instance.
        debug_verbose: If True, print debug information.
        gemini_api_key: DEPRECATED - use api_key instead.
        
    Returns:
        Configured MultiAgentOrchestrator.
    """
    # 处理遗留参数
    if gemini_api_key and not api_key:
        api_key = gemini_api_key
        provider = "gemini"
    
    orchestrator = MultiAgentOrchestrator(debug_verbose=debug_verbose)
    orchestrator.configure(
        api_key=api_key,
        provider=provider,
        model_name=model_name,
    )
    orchestrator.set_agents(
        data_science_agent=data_science_agent,
        domain_expert_agent=domain_expert_agent,
        health_coach_agent=health_coach_agent,
    )
    return orchestrator

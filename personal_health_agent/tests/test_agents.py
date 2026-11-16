"""Tests for PHA agents.

This module tests individual agents:
- DataScienceAgent: Data analysis and code generation
- DomainExpertAgent: Medical interpretation and contextualization  
- HealthCoachAgent: Health coaching and recommendations

Tests are organized into:
- Unit tests: Mock-based, no API keys required
- Integration tests: Require GEMINI_API_KEY (marked with @pytest.mark.integration)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


# =============================================================================
# DataScienceAgent 测试
# =============================================================================

class TestDataScienceAgentUnit:
    """Unit tests for DataScienceAgent (no API key required)."""
    
    def test_import(self):
        """DataScienceAgent can be imported."""
        from pha.agents import DataScienceAgent
        assert DataScienceAgent is not None
    
    def test_instantiation(self):
        """DataScienceAgent can be instantiated."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        assert agent is not None
    
    def test_has_configure_method(self):
        """DataScienceAgent has configure method."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        assert hasattr(agent, 'configure')
        assert callable(agent.configure)
    
    def test_has_load_dataframes_method(self):
        """DataScienceAgent has load_dataframes method."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        assert hasattr(agent, 'load_dataframes')
        assert callable(agent.load_dataframes)
    
    def test_load_dataframes(self, all_sample_dataframes):
        """DataScienceAgent can load dataframes."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        
        # 不应该加薪
        agent.load_dataframes(all_sample_dataframes)
    
    def test_has_query_method(self):
        """DataScienceAgent has query method."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        assert hasattr(agent, 'query')
        assert callable(agent.query)
    
    def test_has_query_with_details_method(self):
        """DataScienceAgent has query_with_details method."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        assert hasattr(agent, 'query_with_details')
        assert callable(agent.query_with_details)


class TestDataScienceAgentPrompts:
    """Tests for DataScienceAgent prompt generation."""
    
    def test_approach_prompt_generation(self):
        """Agent generates approach prompts correctly."""
        from pha.prompts.data_science_prompts import generate_approach_prompt
        
        dfs_str = """
        summary_df columns: date, steps, sleep_minutes, resting_heart_rate
        activities_df columns: start_time, activity_name, duration, calories
        profile_df columns: age, gender, weight, height
        """
        
        prompt = generate_approach_prompt(
            query="What are my average steps?",
            dfs_str=dfs_str,
        )
        
        assert "steps" in prompt.lower()
        assert len(prompt) > 100  # 应该是实质性的
    
    def test_codegen_prompt_generation(self):
        """Agent generates code generation prompts correctly."""
        from pha.prompts.data_science_prompts import generate_codegen_prompt
        
        prompt = generate_codegen_prompt(
            query="What are my average steps?",
            approach_str="Calculate average steps from summary_df",
            dfs_str="summary_df has columns: date, steps, sleep_minutes",
            dfs_info_str="summary_df: 365 rows x 3 columns",
        )
        
        assert "summary_df" in prompt or "steps" in prompt.lower()
        assert len(prompt) > 50


@pytest.mark.integration
@pytest.mark.agents
@pytest.mark.requires_api_key("gemini")
class TestDataScienceAgentIntegration:
    """Integration tests for DataScienceAgent (requires GEMINI_API_KEY)."""
    
    def test_configure_with_api_key(self, gemini_api_key):
        """Agent can be configured with API key."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        
        # 不应该加薪
        agent.configure(gemini_api_key=gemini_api_key)
    
    def test_simple_query(self, gemini_api_key, all_sample_dataframes):
        """Agent responds to simple data query."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        agent.configure(gemini_api_key=gemini_api_key)
        agent.load_dataframes(all_sample_dataframes)
        
        response = agent.query("What are my average daily steps?")
        
        assert response is not None
        assert len(response) > 0
        # 答复应提及步骤或编号
        assert "step" in response.lower() or any(c.isdigit() for c in response)
    
    def test_trend_query(self, gemini_api_key, all_sample_dataframes):
        """Agent can analyze trends."""
        from pha.agents import DataScienceAgent
        agent = DataScienceAgent()
        agent.configure(gemini_api_key=gemini_api_key)
        agent.load_dataframes(all_sample_dataframes)
        
        response = agent.query("How has my sleep changed over the past month?")
        
        assert response is not None
        assert len(response) > 0


# =============================================================================
# DomainExpertAgent 测试
# =============================================================================

class TestDomainExpertAgentUnit:
    """Unit tests for DomainExpertAgent (no API key required)."""
    
    def test_import(self):
        """DomainExpertAgent can be imported."""
        from pha.agents import DomainExpertAgent
        assert DomainExpertAgent is not None
    
    def test_instantiation(self):
        """DomainExpertAgent can be instantiated."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        assert agent is not None
    
    def test_has_get_agent_method(self):
        """DomainExpertAgent has get_agent method."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        assert hasattr(agent, 'get_agent')
        assert callable(agent.get_agent)
    
    def test_has_call_agent_method(self):
        """DomainExpertAgent has call_agent method."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        assert hasattr(agent, 'call_agent')
        assert callable(agent.call_agent)
    
    def test_has_set_user_health_data_method(self):
        """DomainExpertAgent has set_user_health_data method."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        assert hasattr(agent, 'set_user_health_data')
        assert callable(agent.set_user_health_data)
    
    def test_react_availability_check(self):
        """Can check if ReAct is available."""
        from pha.agents.domain_expert_agent import is_react_available
        result = is_react_available()
        assert isinstance(result, bool)


class TestDomainExpertPrompts:
    """Tests for DomainExpertAgent prompts and tools."""
    
    def test_reference_range_function(self):
        """Reference range check function works."""
        from pha.prompts.domain_expert_prompts import check_reference_ranges
        
        # 范围内的数值
        result = check_reference_ranges([[50, 40, 60]])
        assert len(result) == 1
        assert "within" in result[0][0].lower()
        
        # 数值大于幅度
        result = check_reference_ranges([[70, 40, 60]])
        assert "outside" in result[0][0].lower() or "above" in result[0][1].lower()
        
        # 数值低于范围
        result = check_reference_ranges([[30, 40, 60]])
        assert "outside" in result[0][0].lower() or "below" in result[0][1].lower()


@pytest.mark.integration
@pytest.mark.agents
@pytest.mark.requires_api_key("gemini")
class TestDomainExpertAgentIntegration:
    """Integration tests for DomainExpertAgent (requires GEMINI_API_KEY)."""
    
    def test_configure_with_api_key(self, gemini_api_key):
        """Agent can be configured with API key."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        
        # 不应该加薪
        result = agent.get_agent(gemini_api_key=gemini_api_key)
        assert result is not None
    
    def test_health_interpretation_query(self, gemini_api_key):
        """Agent responds to health interpretation query."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        agent.get_agent(gemini_api_key=gemini_api_key)
        
        response = agent.call_agent("What is a healthy resting heart rate for a 35 year old?")
        
        assert response is not None
        assert len(response) > 0
        # 应提及 心率 或 bpm
        assert "heart" in response.lower() or "bpm" in response.lower() or "beat" in response.lower()
    
    def test_with_user_data(self, gemini_api_key, sample_profile_df):
        """Agent uses user health data in responses."""
        from pha.agents import DomainExpertAgent
        agent = DomainExpertAgent()
        agent.get_agent(gemini_api_key=gemini_api_key)
        agent.set_user_health_data_from_df(sample_profile_df)
        
        response = agent.call_agent("Based on my profile, what should my target heart rate zones be?")
        
        assert response is not None
        assert len(response) > 0


# =============================================================================
# HealthCoachAgent 测试
# =============================================================================

class TestHealthCoachAgentUnit:
    """Unit tests for HealthCoachAgent (no API key required)."""
    
    def test_import(self):
        """HealthCoachAgent can be imported."""
        from pha.agents import HealthCoachAgent
        assert HealthCoachAgent is not None
    
    def test_instantiation(self):
        """HealthCoachAgent can be instantiated."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent()
        assert agent is not None
    
    def test_simple_mode_instantiation(self):
        """HealthCoachAgent can be instantiated in simple mode."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent(simple_mode=True)
        assert agent is not None
    
    def test_has_configure_method(self):
        """HealthCoachAgent has configure method."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent()
        assert hasattr(agent, 'configure')
        assert callable(agent.configure)
    
    def test_has_respond_method(self):
        """HealthCoachAgent has respond method."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent()
        assert hasattr(agent, 'respond')
        assert callable(agent.respond)


class TestHealthCoachPrompts:
    """Tests for HealthCoachAgent prompts."""
    
    def test_health_coach_prompts_exist(self):
        """Health coach prompts are defined."""
        from pha.prompts.health_coach_prompts import (
            HEALTH_COACH_SYSTEM_PROMPT,
            GOAL_PROMPT,
            FINISH_DESC_PROMPT,
        )
        
        assert len(HEALTH_COACH_SYSTEM_PROMPT) > 100
        assert len(GOAL_PROMPT) > 10
        assert len(FINISH_DESC_PROMPT) > 10
    
    def test_prompts_mention_health_coaching(self):
        """Prompts reference health coaching role."""
        from pha.prompts.health_coach_prompts import HEALTH_COACH_SYSTEM_PROMPT
        
        prompt_lower = HEALTH_COACH_SYSTEM_PROMPT.lower()
        assert "health" in prompt_lower or "coach" in prompt_lower or "wellness" in prompt_lower


@pytest.mark.integration
@pytest.mark.agents
@pytest.mark.requires_api_key("gemini")
class TestHealthCoachAgentIntegration:
    """Integration tests for HealthCoachAgent (requires GEMINI_API_KEY)."""
    
    def test_configure_with_api_key(self, gemini_api_key):
        """Agent can be configured with API key."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent(simple_mode=True)
        
        # 不应该加薪
        agent.configure(gemini_api_key=gemini_api_key)
    
    def test_coaching_query(self, gemini_api_key):
        """Agent responds to coaching query."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent(simple_mode=True)
        agent.configure(gemini_api_key=gemini_api_key)
        
        response = agent.respond("I want to improve my sleep quality. What should I do?")
        
        assert response is not None
        assert len(response) > 0
        # 应该提到与睡眠有关的咨询
        assert "sleep" in response.lower() or "bed" in response.lower() or "rest" in response.lower()
    
    def test_goal_setting_query(self, gemini_api_key):
        """Agent can help with goal setting."""
        from pha.agents import HealthCoachAgent
        agent = HealthCoachAgent(simple_mode=True)
        agent.configure(gemini_api_key=gemini_api_key)
        
        response = agent.respond("Help me set a realistic step goal for the next month.")
        
        assert response is not None
        assert len(response) > 0


# =============================================================================
# LLM 后端测试
# =============================================================================

class TestLLMBackendUnit:
    """Unit tests for LLM backend abstraction."""
    
    def test_import_get_llm_backend(self):
        """get_llm_backend can be imported."""
        from pha.llm import get_llm_backend
        assert get_llm_backend is not None
        assert callable(get_llm_backend)
    
    def test_backend_classes_exist(self):
        """Backend classes are defined."""
        from pha.llm.backend import (
            LLMBackend,
            FallbackGeminiBackend,
        )
        
        assert LLMBackend is not None
        assert FallbackGeminiBackend is not None
    
    def test_onetwo_availability_check(self):
        """Can check if onetwo is available."""
        from pha.llm.backend import is_onetwo_available
        
        result = is_onetwo_available()
        assert isinstance(result, bool)


@pytest.mark.integration
@pytest.mark.requires_api_key("gemini")
class TestLLMBackendIntegration:
    """Integration tests for LLM backend."""
    
    def test_gemini_backend_generation(self, gemini_api_key):
        """Gemini backend can generate text."""
        from pha.llm import get_llm_backend

        # 不要给特定的模型定下钉子——让get_llm_backend选择电流。
        # 默认。硬码名称会快速变质( gemini - 2.0 - flash 已退役)
        # (b) 提前向现有客户2026提供。
        backend = get_llm_backend(
            backend_type="gemini",
            api_key=gemini_api_key,
        )

        response = backend.generate("What is 2 + 2? Answer with just the number.")

        assert response is not None
        assert "4" in response


# =============================================================================
# 平行 基线 测试
# =============================================================================

@pytest.mark.agents
class TestParallelBaselineUnit:
    """Unit tests for ParallelMultiAgentBaseline."""
    
    def test_import(self):
        """Parallel baseline can be imported."""
        from pha.agents import ParallelMultiAgentBaseline, create_parallel_baseline
        
        assert ParallelMultiAgentBaseline is not None
        assert create_parallel_baseline is not None
    
    def test_initialization(self):
        """Parallel baseline can be initialized."""
        from pha.agents import ParallelMultiAgentBaseline
        
        baseline = ParallelMultiAgentBaseline(debug_verbose=True)
        
        assert baseline is not None
        assert baseline.debug_verbose is True
        assert baseline.data_science_agent is None
        assert baseline.domain_expert_agent is None
        assert baseline.health_coach_agent is None
    
    def test_set_agents(self):
        """Agents can be set on the baseline."""
        from pha.agents import ParallelMultiAgentBaseline, DataScienceAgent
        
        baseline = ParallelMultiAgentBaseline()
        ds_agent = DataScienceAgent()
        
        result = baseline.set_agents(data_science_agent=ds_agent)
        
        assert result is baseline  # 方法链
        assert baseline.data_science_agent is ds_agent
    
    def test_reset_conversation(self):
        """Conversation can be reset."""
        from pha.agents import ParallelMultiAgentBaseline
        
        baseline = ParallelMultiAgentBaseline()
        baseline.conversation_history = [{"role": "user", "content": "test"}]
        baseline.last_ds_response = "test response"
        
        baseline.reset_conversation()
        
        assert baseline.conversation_history == []
        assert baseline.last_ds_response == ""
    
    def test_get_individual_responses(self):
        """Individual responses can be retrieved."""
        from pha.agents import ParallelMultiAgentBaseline
        
        baseline = ParallelMultiAgentBaseline()
        baseline.last_ds_response = "ds response"
        baseline.last_de_response = "de response"
        baseline.last_coach_response = "coach response"
        
        responses = baseline.get_individual_responses()
        
        assert responses["data_science"] == "ds response"
        assert responses["domain_expert"] == "de response"
        assert responses["health_coach"] == "coach response"
    
    def test_synthesis_prompt_format(self):
        """Synthesis prompt is properly formatted."""
        from pha.agents.parallel_baseline import PARALLEL_SYNTHESIS_PROMPT
        
        formatted = PARALLEL_SYNTHESIS_PROMPT.format(
            user_context="test context",
            query="test query",
            ds_response="ds response",
            de_response="de response",
            coach_response="coach response",
        )
        
        assert "test query" in formatted
        assert "ds response" in formatted
        assert "de response" in formatted
        assert "coach response" in formatted


# =============================================================================
# PHIABaseline 测试
# =============================================================================

class TestPHIABaselineUnit:
    """Unit tests for PHIABaseline (no API key required)."""
    
    def test_import(self):
        """PHIABaseline can be imported."""
        from pha.agents import PHIABaseline
        assert PHIABaseline is not None
    
    def test_is_onetwo_available_function(self):
        """is_onetwo_available function exists and returns bool."""
        from pha.agents import is_onetwo_available
        result = is_onetwo_available()
        assert isinstance(result, bool)
    
    def test_create_phia_baseline_import(self):
        """create_phia_baseline factory function can be imported."""
        from pha.agents import create_phia_baseline
        assert create_phia_baseline is not None
        assert callable(create_phia_baseline)
    
    def test_preamble_exists(self):
        """PHIA preamble prompt exists."""
        from pha.agents.phia_baseline import PHIA_PREAMBLE
        assert PHIA_PREAMBLE is not None
        assert "summary_df" in PHIA_PREAMBLE
        assert "activities_df" in PHIA_PREAMBLE
        assert "profile" in PHIA_PREAMBLE
    
    def test_question_prefix_exists(self):
        """QUESTION_PREFIX exists."""
        from pha.agents.phia_baseline import QUESTION_PREFIX
        assert QUESTION_PREFIX is not None
        assert "tool_code" in QUESTION_PREFIX
    
    def test_helper_functions_exist(self):
        """Helper functions exist."""
        from pha.agents.phia_baseline import (
            simple_python_executor,
            tavily_search_func,
        )
        assert callable(simple_python_executor)
        assert callable(tavily_search_func)
    
    def test_simple_python_executor(self):
        """simple_python_executor works for basic code."""
        from pha.agents.phia_baseline import simple_python_executor
        import pandas as pd
        import numpy as np
        
        sandbox = {"pd": pd, "np": np}
        
        # 测试简单的表达式
        result = simple_python_executor("2 + 2", sandbox)
        assert "4" in result
        
        # 测试打印语句
        result = simple_python_executor("print('hello')", sandbox)
        assert "hello" in result
    
    def test_tavily_search_no_key(self, monkeypatch):
        """tavily_search_func handles missing API key gracefully."""
        from pha.agents.phia_baseline import tavily_search_func

        # 函数在 api_key 时返回到 os. getenv(' TAVILY_API_KEY]。
        # 没有，所以我们得清空内脏 来测试
        # 缺少密钥路径。没有此选项，开发者将拥有 TAVILY_API_KEY。
        # 他们的壳里静悄悄地 把这个变成一个现场搜索测试
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        result = tavily_search_func("test query", api_key=None)
        assert "Error" in result or "error" in result.lower()
    
    def test_clean_react_format_helper(self):
        """_clean_react_format removes ReAct artifacts."""
        from pha.agents import is_onetwo_available
        
        if not is_onetwo_available():
            pytest.skip("onetwo not available")
        
        from pha.agents import PHIABaseline
        baseline = PHIABaseline()
        
        # 测试前缀删除
        assert baseline._clean_react_format("[Thought]: This is a thought") == "This is a thought"
        assert baseline._clean_react_format("[Finish]: Final answer") == "Final answer"
        
        # 测试内置标记去除 - 应在[法案]中切换：
        text_with_act = "Some analysis here\n[Act]: ```code```"
        result = baseline._clean_react_format(text_with_act)
        assert "[Act]" not in result
        assert "Some analysis here" in result
        
        # 测试代码块删除
        text_with_code = "Analysis\n```python\ncode\n```\nMore text"
        result = baseline._clean_react_format(text_with_code)
        assert "```" not in result
        
        # 测试 JSON 检测
        assert baseline._clean_react_format('{"key": "value"}') == ""
        
        # 测试普通文本通过
        assert baseline._clean_react_format("Normal response text") == "Normal response text"
    
    def test_is_complete_response_helper(self):
        """_is_complete_response detects incomplete responses."""
        from pha.agents import is_onetwo_available
        
        if not is_onetwo_available():
            pytest.skip("onetwo not available")
        
        from pha.agents import PHIABaseline
        baseline = PHIABaseline()
        
        # 完整答复
        assert baseline._is_complete_response("Your average sleep is 7 hours per night, which is within the healthy range.") == True
        
        # 答复不完整
        assert baseline._is_complete_response("I need to analyze the data first") == False
        assert baseline._is_complete_response("[Act]: some code") == False
        assert baseline._is_complete_response("") == False


class TestPHIABaselineWithOnetwo:
    """Tests that require onetwo to be available."""
    
    def test_initialization(self):
        """PHIABaseline can be instantiated when onetwo is available."""
        from pha.agents import PHIABaseline, is_onetwo_available
        
        if not is_onetwo_available():
            pytest.skip("onetwo not available")
        
        baseline = PHIABaseline()
        assert baseline is not None
        assert baseline.conversation_history == []
    
    def test_has_required_methods(self):
        """PHIABaseline has required methods."""
        from pha.agents import PHIABaseline, is_onetwo_available
        
        if not is_onetwo_available():
            pytest.skip("onetwo not available")
        
        baseline = PHIABaseline()
        assert hasattr(baseline, 'configure')
        assert hasattr(baseline, 'load_data')
        assert hasattr(baseline, 'respond')
        assert hasattr(baseline, 'reset_conversation')

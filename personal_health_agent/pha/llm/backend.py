"""LLM backend abstraction layer.

This module provides the LLM backend interface for PHA agents,
supporting Google Gemini, OpenAI, and Anthropic models.

Usage:
    from pha.llm import setup_backend, run_llm
    
    # Set up the Gemini backend
    setup_backend(api_key="your-key")
    
    # Generate text
    response = run_llm("What is the capital of France?")

Note:
    For best results with onetwo ReAct agents, install onetwo:
    pip install git+https://github.com/google-deepmind/onetwo
"""

import os
from typing import Optional, Callable
from abc import ABC, abstractmethod


# =============================================================================
# OpenAI API 兼容性补丁
# =============================================================================
# 较新的OpenAI型号(o1,o3,gpt-4o等)需要`max_completion_tokens`.
# 而不是 `max_tokens`。此补丁确保了与库的兼容性
# 就像还有两个仍然使用旧参数名。

_OPENAI_PATCHED = False

def _patch_openai_for_max_tokens():
    """Patch OpenAI client to convert max_tokens to max_completion_tokens.
    
    This is needed because newer OpenAI models don't support max_tokens,
    but some libraries (like onetwo) still use it.
    """
    global _OPENAI_PATCHED
    if _OPENAI_PATCHED:
        return
    
    try:
        from openai.resources.chat import completions
        
        _original_create = completions.Completions.create
        
        def _patched_create(self, *args, **kwargs):
            # 将 max_tokens 转换为 max_completion_tokens 用于更新的模型
            if 'max_tokens' in kwargs and 'max_completion_tokens' not in kwargs:
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            return _original_create(self, *args, **kwargs)
        
        completions.Completions.create = _patched_create
        _OPENAI_PATCHED = True
    except Exception:
        pass  # 如果补丁失败，继续没有它


# 尝试导入 1 2,如果无法使用，返回到简单的执行中
# 注:由于使用新类型的参数语法，需要1个Python 3.12+
_ONETWO_AVAILABLE = False
_ONETWO_OPENAI_AVAILABLE = False
_ONETWO_GENAI_AVAILABLE = False  # 如果有新的 GoogleGenAIAPI 后端，则为真
_ONETWO_IMPORT_ERROR = None
try:
    from onetwo import ot
    from onetwo.builtins import llm
    
    # 使用 GoogleGenAIAPI 后端(新google.genai SDK,无贬值警告，
    # Gemini2.5+模型的安全默认值为OFF
    try:
        from onetwo.backends import google_genai_api
        _ONETWO_GENAI_AVAILABLE = True
    except ImportError:
        pass  # 将使用 FallbackGeminiBackend (google.genai SDK)代替。
    
    _ONETWO_AVAILABLE = True
    
    # 尝试导入 OpenAI 后端
    try:
        from onetwo.backends import openai_api
        _ONETWO_OPENAI_AVAILABLE = True
        # 当OpenAI 后端可用时应用补丁
        _patch_openai_for_max_tokens()
    except ImportError:
        pass  # OpenAI 后端不可用，将使用倒置
except ImportError as e:
    _ONETWO_IMPORT_ERROR = str(e)
except SyntaxError as e:
    # Python版本太老了(一二需要3.12+).
    _ONETWO_IMPORT_ERROR = f"onetwo requires Python 3.12+, you have {__import__('sys').version}"
except Exception as e:
    _ONETWO_IMPORT_ERROR = str(e)


# =============================================================================
# 后端接口
# =============================================================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            The generated text response.
        """
        pass
    
    @abstractmethod
    def register(self) -> None:
        """Register this backend with onetwo (if applicable)."""
        pass


# =============================================================================
# OneTwo -基于后端(首选)
# =============================================================================

if _ONETWO_AVAILABLE:
    
    class OneTwoGeminiBackend(LLMBackend):
        """Gemini backend using onetwo's GoogleGenAIAPI (new google.genai SDK)."""
        
        def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "models/gemini-3.1-flash-lite-preview",
            temperature: float = 0.0,
            max_tokens: int = 8192,
        ):
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key required. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            
            if not _ONETWO_GENAI_AVAILABLE:
                raise ImportError(
                    "onetwo's GoogleGenAIAPI backend not available. "
                    "Update onetwo: pip install -U git+https://github.com/google-deepmind/onetwo"
                )
            
            self.backend = google_genai_api.GoogleGenAIAPI(
                api_key=self.api_key,
                generate_model_name=model_name,
                # OneTwo 验证两个生成AND 聊天模式名称对齐
                # 客户端. models.list (); chat_model_name 默认为
                # DEFAULT_GENERATE_MODEL ( `models/gemini-2.5-flash` 即为
                # 不在每个密钥上显示。绑定在用户的选中位置。
                chat_model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                threadpool_size=1,
            )
            self._registered = False
        
        def register(self) -> None:
            """Register this backend with onetwo's global registry."""
            if not self._registered:
                self.backend.register()
                self._registered = True
        
        def generate(self, prompt: str) -> str:
            """Generate a response using Gemini via onetwo."""
            self.register()
            response = ot.run(llm.generate_text(prompt=prompt))
            return response


if _ONETWO_OPENAI_AVAILABLE:
    
    class OneTwoOpenAIBackend(LLMBackend):
        """OpenAI backend using onetwo library.
        
        This enables full OneTwo functionality (ReAct agents, caching, etc.)
        with OpenAI models like GPT-4o.
        """
        
        def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "gpt-4o",
            temperature: float = 0.0,
            max_tokens: int = 8192,
        ):
            """Initialize the OpenAI backend via onetwo.
            
            Args:
                api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
                model_name: Model to use (e.g., gpt-4o, gpt-4-turbo).
                temperature: Sampling temperature.
                max_tokens: Maximum tokens in response.
            """
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            # 原因/ GPT-5 模型只接受温度= 1.0 (其
            # API 默认值。我们用在其他地方的0.7最接近的模拟值。
            if model_name.startswith(("o1", "o3", "o4", "gpt-5")):
                temperature = 1.0
            self.backend = openai_api.OpenAIAPI(
                api_key=self.api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=1,
            )
            self._registered = False
        
        def register(self) -> None:
            """Register this backend with onetwo's global registry."""
            if not self._registered:
                self.backend.register()
                self._registered = True
        
        def generate(self, prompt: str) -> str:
            """Generate a response using OpenAI via onetwo."""
            self.register()
            response = ot.run(llm.generate_text(prompt=prompt))
            return response


# =============================================================================
# 后端(当无法使用 1 2 时)
# =============================================================================

class FallbackGeminiBackend(LLMBackend):
    """Fallback Gemini backend using google-genai directly.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-3.1-flash-lite-preview",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the Gemini backend.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: "
                "pip install google-genai"
            )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    
    def register(self) -> None:
        """No-op for fallback backend."""
        pass
    
    def generate(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.generation_config,
        )
        return response.text


class FallbackOpenAIBackend(LLMBackend):
    """Fallback OpenAI backend.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the OpenAI backend.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: "
                "pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 原因/ GPT-5 模型拒绝任何非默认温度(只有1)是
        # 支持此选项。关闭这些选项的 kwarg ,因此 API 使用默认值。
        self._is_reasoning_model = (
            self.model_name.startswith("o1")
            or self.model_name.startswith("o3")
            or self.model_name.startswith("o4")
            or self.model_name.startswith("gpt-5")
        )

    def register(self) -> None:
        """No-op for fallback backend."""
        pass

    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI."""
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if not self._is_reasoning_model:
            kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class FallbackAnthropicBackend(LLMBackend):
    """Fallback Anthropic Claude backend.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the Anthropic backend.
        
        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: "
                "pip install anthropic"
            )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def register(self) -> None:
        """No-op for fallback backend."""
        pass
    
    def generate(self, prompt: str) -> str:
        """Generate a response using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# =============================================================================
# 工厂功能
# =============================================================================

def get_llm_backend(
    backend_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    prefer_onetwo: bool = True,
) -> LLMBackend:
    """Factory function to create an LLM backend.
    
    Args:
        backend_type: "gemini", "openai", or "anthropic".
        api_key: API key for the backend.
        model_name: Model name (uses sensible defaults if not provided).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        prefer_onetwo: If True, use onetwo when available (recommended for
                      Gemini and OpenAI to enable ReAct agents and caching).
    
    Returns:
        Configured LLMBackend instance.
    
    Example:
        >>> backend = get_llm_backend("gemini", api_key="your-key")
        >>> response = backend.generate("Hello!")
    """
    backend_type = backend_type.lower()
    
    # 默认模型名称
    defaults = {
        "gemini": "models/gemini-3.1-flash-lite-preview",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5-20250929",
    }
    
    if backend_type not in defaults:
        raise ValueError(f"Unknown backend type: {backend_type}. Choose from: {list(defaults.keys())}")
    
    model_name = model_name or defaults[backend_type]
    
    # 如果有的话，请使用其中的一二个，并优先使用Gemini(要求GoogleGenAIAPI)
    if prefer_onetwo and _ONETWO_AVAILABLE and _ONETWO_GENAI_AVAILABLE and backend_type == "gemini":
        return OneTwoGeminiBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    # 可用且首选 OpenAI 时使用 12
    if prefer_onetwo and _ONETWO_OPENAI_AVAILABLE and backend_type == "openai":
        return OneTwoOpenAIBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    # 否则会使用后端
    if backend_type == "gemini":
        return FallbackGeminiBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "openai":
        return FallbackOpenAIBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "anthropic":
        return FallbackAnthropicBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# =============================================================================
# 方便职能
# =============================================================================

# 全球后端实例
_global_backend: Optional[LLMBackend] = None


def setup_backend(
    backend_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> LLMBackend:
    """Set up and register the global LLM backend.
    
    This matches the onetwo pattern of registering a backend globally.
    
    Args:
        backend_type: "gemini", "openai", or "anthropic".
        api_key: API key for the backend.
        model_name: Model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    
    Returns:
        The configured backend.
    
    Example:
        >>> setup_backend("gemini", api_key="your-key")
        >>> response = run_llm("Hello!")
    """
    global _global_backend
    _global_backend = get_llm_backend(
        backend_type=backend_type,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _global_backend.register()
    return _global_backend


def run_llm(
    prompt: str,
    parse_function: Optional[Callable[[str], str]] = None,
) -> str:
    """Run the LLM with the global backend.
    
    Args:
        prompt: The input prompt.
        parse_function: Optional function to parse the response.
    
    Returns:
        The (optionally parsed) response.
    
    Example:
        >>> setup_backend("gemini")
        >>> response = run_llm("What is 2+2?")
    """
    if _global_backend is None:
        raise RuntimeError("No backend set up. Call setup_backend() first.")
    
    response = _global_backend.generate(prompt)
    
    if parse_function is not None:
        return parse_function(response)
    return response


def is_onetwo_available() -> bool:
    """Check if onetwo is available."""
    return _ONETWO_AVAILABLE


def is_onetwo_openai_available() -> bool:
    """Check if onetwo OpenAI backend is available."""
    return _ONETWO_OPENAI_AVAILABLE


def get_onetwo_import_error() -> str:
    """Get the error message if onetwo import failed."""
    return _ONETWO_IMPORT_ERROR or "No error"

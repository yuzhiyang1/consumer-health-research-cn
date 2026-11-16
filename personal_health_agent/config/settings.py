"""Centralized configuration settings for PHA.

This module provides a single place to configure all paths, API keys,
and model settings used throughout the PHA system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# =============================================================================
# 默认路径 -通过设置类或环境变量可覆盖
# =============================================================================

# 获取工程的根目录( assumes config/是从根向下一个关卡)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 默认数据目录
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "sample"

# 样本数据文件的默认路径
DEFAULT_SUMMARY_PATH = DEFAULT_DATA_DIR / "summary.csv"
DEFAULT_ACTIVITIES_PATH = DEFAULT_DATA_DIR / "activities.csv"
DEFAULT_PROFILE_PATH = DEFAULT_DATA_DIR / "profile.csv"
DEFAULT_POPULATION_PATH = DEFAULT_DATA_DIR / "population_percentiles.csv"


# =============================================================================
# LLM 配置
# =============================================================================

# 支持的 LLM 后端
LLMBackend = Literal["gemini", "openai", "anthropic"]

# 每个后端的默认模型名称
DEFAULT_MODELS = {
    "gemini": "models/gemini-3.1-flash-lite-preview",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}


# =============================================================================
# 代理配置常数
# =============================================================================

# 代码执行重试设置
DEFAULT_NUM_CODE_RETRIES = 3
DEFAULT_TOTAL_CODE_RETRIES = 5


@dataclass
class Settings:
    """Configuration settings for PHA.
    
    This class centralizes all configuration for the PHA system.
    
    Example usage:
        # Use defaults
        settings = Settings()
        
        # Custom data directory
        settings = Settings(data_dir="./my_data")
        
        # Use OpenAI instead of Gemini
        settings = Settings(llm_backend="openai")
        
        # Load API key from environment
        settings = Settings()  # Reads GEMINI_API_KEY, OPENAI_API_KEY, etc.
    """
    
    # 数据路径
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    summary_path: Optional[Path] = None
    activities_path: Optional[Path] = None
    profile_path: Optional[Path] = None
    population_path: Optional[Path] = None
    
    # LLM 设置
    llm_backend: LLMBackend = "gemini"
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192
    
    # API 密钥(如果未提供，则从环境装入)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # 搜索 API 领域专家 Agent 键
    tavily_api_key: Optional[str] = None
    
    # 代理设置
    num_code_retries: int = DEFAULT_NUM_CODE_RETRIES
    total_code_retries: int = DEFAULT_TOTAL_CODE_RETRIES
    
    # 是否在时间上将数据定位( 转换日期为“今天 ” )
    temporally_localize: bool = True
    
    # 调试设置
    debug_verbose: bool = False
    
    def __post_init__(self):
        """Initialize paths and load API keys from environment."""
        # 将字符串路径转换为路径对象并解析相对路径
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        # 解决data_dir 走向一条绝对的道路
        # 处理路径相对于当前工作目录的情况
        if not self.data_dir.is_absolute():
            self.data_dir = self.data_dir.resolve()
        
        # 如果未明确提供基于 data_dir 的默认文件路径
        if self.summary_path is None:
            self.summary_path = self.data_dir / "summary.csv"
        if self.activities_path is None:
            self.activities_path = self.data_dir / "activities.csv"
        if self.profile_path is None:
            self.profile_path = self.data_dir / "profile.csv"
        if self.population_path is None:
            self.population_path = self.data_dir / "population_percentiles.csv"
        
        # 如果字符串并解析相对路径，则转换为路径对象
        for attr in ['summary_path', 'activities_path', 'profile_path', 'population_path']:
            val = getattr(self, attr)
            if isinstance(val, str):
                val = Path(val)
            if val is not None and not val.is_absolute():
                val = val.resolve()
            setattr(self, attr, val)
        
        # 如果未提供，则根据后端设置默认模式名称
        if self.model_name is None:
            self.model_name = DEFAULT_MODELS.get(self.llm_backend)
        
        # 如果不提供，从环境中装入 API 密钥
        if self.gemini_api_key is None:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.tavily_api_key is None:
            self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key for the current LLM backend."""
        if self.llm_backend == "gemini":
            return self.gemini_api_key
        elif self.llm_backend == "openai":
            return self.openai_api_key
        elif self.llm_backend == "anthropic":
            return self.anthropic_api_key
        return None
    
    def validate(self) -> None:
        """Validate that required settings are present.
        
        Raises:
            ValueError: If required settings are missing.
        """
        # 选中后端的 API 密钥
        api_key = self.get_api_key()
        if api_key is None:
            raise ValueError(
                f"No API key found for {self.llm_backend}. "
                f"Set {self.llm_backend.upper()}_API_KEY environment variable "
                f"or pass {self.llm_backend}_api_key to Settings."
            )
        
        # 检查数据文件是否存在( 警告但不要出错 -它们可能稍后创建)。
        for path_attr in ['summary_path', 'activities_path', 'profile_path', 'population_path']:
            path = getattr(self, path_attr)
            if path and not path.exists():
                print(f"Warning: {path_attr} not found at {path}")


# 为方便创建默认设置实例
default_settings = Settings()

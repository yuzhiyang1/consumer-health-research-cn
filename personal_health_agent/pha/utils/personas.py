"""Persona management for easy switching between user health profiles.

This module provides utilities for discovering, listing, and loading
different user personas (health data profiles) for PHA.

Directory Structure:
    data/
        sample/           # Default persona
            summary.csv
            activities.csv
            profile.csv
            population_percentiles.csv
            description.txt  # Optional
        persona_alice/    # Additional persona
            summary.csv
            activities.csv
            profile.csv
            description.txt
        ...

Usage:
    from pha.utils.personas import list_personas, load_persona_data
    
    # List available personas
    personas = list_personas()
    
    # Load a specific persona
    data = load_persona_data("sample")
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd


# =============================================================================
# 配置
# =============================================================================

# 默认数据目录( 相对于项目根)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class PersonaInfo:
    """Information about an available persona."""
    id: str
    name: str
    description: str = ""
    data_dir: Optional[Path] = None
    
    # 人口概况(载自profile.csv).
    age: Optional[int] = None
    gender: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    
    # 数据摘要
    days_of_data: Optional[int] = None
    num_activities: Optional[int] = None


def _load_persona_metadata(persona_dir: Path) -> PersonaInfo:
    """Load metadata for a persona from its directory.
    
    Args:
        persona_dir: Path to the persona's data directory.
        
    Returns:
        PersonaInfo with loaded metadata.
    """
    persona_id = persona_dir.name
    display_name = '合成示例档案' if persona_id == 'sample' else persona_id.replace('_', ' ').replace('-', ' ').title()
    
    # 档案说明使用 UTF-8，可包含中文。
    description = ""
    desc_file = persona_dir / "description.txt"
    if desc_file.exists():
        try:
            description = desc_file.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            description = desc_file.read_text().strip()
    
    # 创建基础信息
    info = PersonaInfo(
        id=persona_id,
        name=display_name,
        description=description,
        data_dir=persona_dir,
    )
    
    # 尝试装入剖面人口统计
    profile_path = persona_dir / "profile.csv"
    if profile_path.exists():
        try:
            profile = pd.read_csv(profile_path)
            if len(profile) > 0:
                row = profile.iloc[0]
                info.age = int(row.get('age', row.get('Age', None))) if 'age' in row or 'Age' in row else None
                info.gender = str(row.get('sex', row.get('gender', row.get('Gender', None))))
                info.height = float(row.get('height', row.get('Height', None))) if 'height' in row or 'Height' in row else None
                info.weight = float(row.get('weight', row.get('Weight', None))) if 'weight' in row or 'Weight' in row else None
        except Exception:
            pass  # 元数据是可选的
    
    # 尝试获取数据摘要
    summary_path = persona_dir / "summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            info.days_of_data = len(summary)
        except Exception:
            pass
    
    activities_path = persona_dir / "activities.csv"
    if activities_path.exists():
        try:
            activities = pd.read_csv(activities_path)
            info.num_activities = len(activities)
        except Exception:
            pass
    
    return info


# =============================================================================
# 发现者
# =============================================================================

def discover_personas(data_dir: Optional[Path] = None) -> List[PersonaInfo]:
    """Discover all available personas in the data directory.
    
    A valid persona directory must contain at minimum:
    - summary.csv
    - activities.csv  
    - profile.csv
    
    Args:
        data_dir: Root data directory. Defaults to project's data/ folder.
        
    Returns:
        List of PersonaInfo objects for each valid persona.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    
    personas = []
    required_files = ['summary.csv', 'activities.csv', 'profile.csv']
    
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 跳过隐藏和缓存目录
        if subdir.name.startswith('.') or subdir.name == '__pycache__':
            continue
        
        # 检查所需文件
        has_required = all((subdir / f).exists() for f in required_files)
        
        if has_required:
            info = _load_persona_metadata(subdir)
            personas.append(info)
    
    return personas


def list_personas(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List available personas formatted for UI display.
    
    Args:
        data_dir: Root data directory.
        
    Returns:
        List of dicts with persona info for UI dropdowns.
    """
    personas = discover_personas(data_dir)
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "demographics": {
                "age": p.age,
                "gender": p.gender,
                "height": p.height,
                "weight": p.weight,
            },
            "data_summary": {
                "days": p.days_of_data,
                "activities": p.num_activities,
            },
        }
        for p in personas
    ]


def get_persona_ids(data_dir: Optional[Path] = None) -> List[str]:
    """Get list of available persona IDs.
    
    Args:
        data_dir: Root data directory.
        
    Returns:
        List of persona ID strings.
    """
    return [p.id for p in discover_personas(data_dir)]


def get_default_persona() -> str:
    """Get the default persona ID.
    
    Returns:
        Default persona ID ("sample").
    """
    return "sample"


# =============================================================================
# 正在装入
# =============================================================================

def load_persona_data(
    persona_id: str,
    data_dir: Optional[Path] = None,
    enforce_schema: bool = True,
    temporally_localize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data for a specific persona.
    
    Args:
        persona_id: The persona identifier (directory name).
        data_dir: Root data directory.
        enforce_schema: Whether to enforce column dtypes.
        temporally_localize: Whether to shift dates to appear as "today".
        
    Returns:
        Tuple of (summary_df, activities_df, profile_df, population_df)
        
    Raises:
        ValueError: If persona not found.
    """
    from .data_loader import load_persona
    
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    persona_dir = data_dir / persona_id
    
    if not persona_dir.exists():
        available = get_persona_ids(data_dir)
        raise ValueError(
            f"Persona '{persona_id}' not found. "
            f"Available personas: {available}"
        )
    
    # 查找人口百分数文件
    # 先在个人目录中检查，然后寻找共享的目录
    population_path = persona_dir / "population_percentiles.csv"
    if not population_path.exists():
        # 寻找其他人的共享人口文件a dirs
        for other_dir in data_dir.iterdir():
            if other_dir.is_dir():
                other_pop = other_dir / "population_percentiles.csv"
                if other_pop.exists():
                    population_path = other_pop
                    break
    
    return load_persona(
        summary_path=persona_dir / "summary.csv",
        activities_path=persona_dir / "activities.csv",
        profile_path=persona_dir / "profile.csv",
        population_path=population_path,
        enforce_schema=enforce_schema,
        temporally_localize=temporally_localize,
    )


def validate_persona_directory(persona_dir: Path) -> Tuple[bool, List[str]]:
    """Validate a persona directory has all required files.
    
    Args:
        persona_dir: Path to the persona directory.
        
    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    required_files = ['summary.csv', 'activities.csv', 'profile.csv']
    missing = [f for f in required_files if not (persona_dir / f).exists()]
    return len(missing) == 0, missing


# =============================================================================
# 创建者
# =============================================================================

def create_persona_template(
    persona_id: str,
    data_dir: Optional[Path] = None,
    description: str = "",
) -> Path:
    """Create a new persona directory with template files.
    
    This creates empty CSV files with the correct schema that can be
    populated with user health data.
    
    Args:
        persona_id: ID for the new persona (will be directory name).
        data_dir: Root data directory.
        description: Optional description for the persona.
        
    Returns:
        Path to the created persona directory.
        
    Raises:
        ValueError: If persona already exists.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    persona_dir = data_dir / persona_id
    
    if persona_dir.exists():
        raise ValueError(f"Persona '{persona_id}' already exists at {persona_dir}")
    
    # 创建目录
    persona_dir.mkdir(parents=True)
    
    # 创建 summary.csv 模板
    summary_columns = [
        "datetime", "resting_heart_rate", "heart_rate_variability",
        "steps", "active_zone_minutes", "sleep_minutes", 
        "deep_sleep_minutes", "rem_sleep_minutes", "light_sleep_minutes",
        "awake_minutes", "stress_management_score", "bed_time", "wake_up_time"
    ]
    pd.DataFrame(columns=summary_columns).to_csv(
        persona_dir / "summary.csv", index=False
    )
    
    # 创建 activities.csv 模板
    activities_columns = [
        "start_time", "end_time", "activity_name", "activityName",
        "duration", "calories", "average_heart_rate", "steps", "distance"
    ]
    pd.DataFrame(columns=activities_columns).to_csv(
        persona_dir / "activities.csv", index=False
    )
    
    # 创建 profile.csv 模板
    profile_data = {
        "age": [30],
        "sex": ["unknown"],
        "height": [170],
        "weight": [70],
        "bmi": [24.2],
    }
    pd.DataFrame(profile_data).to_csv(
        persona_dir / "profile.csv", index=False
    )
    
    # 创建 description.txt
    if description:
        (persona_dir / "description.txt").write_text(description, encoding="utf-8")
    else:
        (persona_dir / "description.txt").write_text(
            f"{persona_id} 的健康档案。\n可编辑此文件补充档案说明。",
            encoding="utf-8",
        )
    
    return persona_dir


def copy_persona(
    source_id: str,
    new_id: str,
    data_dir: Optional[Path] = None,
) -> Path:
    """Copy an existing persona to create a new one.
    
    Args:
        source_id: ID of the persona to copy.
        new_id: ID for the new persona.
        data_dir: Root data directory.
        
    Returns:
        Path to the created persona directory.
        
    Raises:
        ValueError: If source doesn't exist or new ID already exists.
    """
    import shutil
    
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    
    data_dir = Path(data_dir)
    source_dir = data_dir / source_id
    new_dir = data_dir / new_id
    
    if not source_dir.exists():
        raise ValueError(f"Source persona '{source_id}' not found")
    
    if new_dir.exists():
        raise ValueError(f"Persona '{new_id}' already exists")
    
    shutil.copytree(source_dir, new_dir)
    
    # 更新说明
    desc_file = new_dir / "description.txt"
    desc_file.write_text(f"从 {source_id} 复制的档案。\n可修改这段说明。", encoding="utf-8")
    
    return new_dir

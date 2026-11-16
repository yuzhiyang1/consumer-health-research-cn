"""Utility functions for loading and parsing health data.

This module provides functions for loading personal health record (PHR) data
from CSV files and preparing it for analysis by PHA agents.
"""

import datetime
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .data_schemas import (
    SUMMARY_SCHEMA,
    ACTIVITIES_SCHEMA,
    POPULATION_SCHEMA,
    PhdaDataFrame,
    enforce_persona_schema,
)


def get_personal_df_from_phr(phr_df: pd.DataFrame) -> pd.DataFrame:
  """Extract profile information from a PHR (Personal Health Record) DataFrame."""
  age = phr_df["age"].to_list()[0]
  weight = phr_df["weight"].to_list()[0]
  height = phr_df["height"].to_list()[0]
  gender = phr_df["sex"].to_list()[0]
  bmi = phr_df["bmi"].to_list()[0]
  return pd.DataFrame({
      "age": [age],
      "weight": [weight],
      "height": [height],
      "gender": [gender],
      "bmi": [bmi],
  })


def load_persona(
    summary_path: Optional[Union[str, Path]] = None,
    activities_path: Optional[Union[str, Path]] = None,
    profile_path: Optional[Union[str, Path]] = None,
    population_path: Optional[Union[str, Path]] = None,
    settings: Optional["Settings"] = None,  # type: ignore
    enforce_schema: bool = True,
    temporally_localize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Loads user health data from CSV files.

  Args:
    summary_path: Path to summary CSV file (daily health metrics).
    activities_path: Path to activities CSV file (exercise records).
    profile_path: Path to profile CSV file (user demographics).
    population_path: Path to population percentiles CSV file.
    settings: Optional Settings object. If provided, paths are read from it.
    enforce_schema: Whether to enforce column dtypes and filter columns.
    temporally_localize: Whether to shift dates so data appears to end "today".

  Returns:
    Tuple of (summary_df, activities_df, profile_df, population_df)
  """
  # 如果提供了设置，请使用设置中的路径
  if settings is not None:
    summary_path = summary_path or settings.summary_path
    activities_path = activities_path or settings.activities_path
    profile_path = profile_path or settings.profile_path
    population_path = population_path or settings.population_path
    if temporally_localize is True:
      temporally_localize = settings.temporally_localize

  # 验证我们所有必要的路径
  if not all([summary_path, activities_path, profile_path, population_path]):
    raise ValueError(
        "All data paths must be provided either directly or via Settings. "
        "Missing paths: " + ", ".join([
            name for name, path in [
                ("summary_path", summary_path),
                ("activities_path", activities_path),
                ("profile_path", profile_path),
                ("population_path", population_path),
            ] if path is None
        ])
    )

  # 转换为路径对象
  summary_path = Path(summary_path)
  activities_path = Path(activities_path)
  profile_path = Path(profile_path)
  population_path = Path(population_path)

  print(f"Loading data from {summary_path.parent}")

  # 装入 CSVs
  summary = pd.read_csv(summary_path)
  activities = pd.read_csv(activities_path)
  profile = pd.read_csv(profile_path)
  population = pd.read_csv(population_path)

  # 转换为 PhdaDataFrame ,用于 during () 支持
  summary = PhdaDataFrame(summary)
  activities = PhdaDataFrame(activities)
  population = PhdaDataFrame(population)

  # 必要时重新命名人口栏
  population_renames = {
      "age_group": "age",
      "resting_heart_rate_bpm": "resting_heart_rate",
      "resting_heart_rate_covariance_value": "heart_rate_variability",
      "azm_in_fat_burn_minute_cnt": "fatburn_active_zone_minutes",
      "azm_in_cardio_minute_cnt": "cardio_active_zone_minutes",
      "azm_in_peak_minute_cnt": "peak_active_zone_minutes",
      "step_cnt": "steps",
      "sleep_rem_minute_cnt": "rem_sleep_minutes",
      "sleep_deep_minute_cnt": "deep_sleep_minutes",
      "sleep_light_minute_cnt": "light_sleep_minutes",
      "sleep_awake_minute_cnt": "awake_minutes",
      "stress_management_score": "stress_management_score",
  }
  # 只重命名已存在的列
  rename_cols = {k: v for k, v in population_renames.items() if k in population.columns}
  if rename_cols:
    population = population.rename(columns=rename_cols)

  # 强制计划(dtypes和列过滤)
  if enforce_schema:
    summary, activities, population = enforce_persona_schema(
        summary, activities, population
    )

  # 临时本地化( 变换日期导致数据似乎结束“今天 )
  if temporally_localize:
    if isinstance(temporally_localize, bool):
      temporally_localize = "today"
    summary, activities = localize_to_date(
        summary, activities, temporally_localize
    )

  return summary, activities, profile, population


def localize_to_date(
    summary_df: pd.DataFrame,
    activities_df: pd.DataFrame,
    date: Optional[Union[datetime.datetime, datetime.date, str]] = "today",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Modifies the timestamp columns so that it appears the data ended today.
  
  This also updates the index to stay in sync with the datetime columns.
  """

  # 在两个数据框中指明最近的日期
  # 使用日期时间栏，而非索引(索引可以是整数)
  if "start_time" in activities_df.columns:
    latest_activity_date = pd.to_datetime(activities_df["start_time"]).max().date()
  elif isinstance(activities_df.index, pd.DatetimeIndex):
    latest_activity_date = activities_df.index.max().date()
  else:
    latest_activity_date = datetime.date.today()
  
  if "datetime" in summary_df.columns:
    latest_summary_date = pd.to_datetime(summary_df["datetime"]).max().date()
  elif "date" in summary_df.columns:
    latest_summary_date = pd.to_datetime(summary_df["date"]).max().date()
  elif isinstance(summary_df.index, pd.DatetimeIndex):
    latest_summary_date = summary_df.index.max().date()
  else:
    latest_summary_date = datetime.date.today()
    
  latest_date = max(latest_activity_date, latest_summary_date)

  # 计算从今天算起的天数差异
  date_zero = pd.to_datetime(date).date()
  days_difference = (date_zero - latest_date).days

  # `summary_df`. 熊猫4更严格地指定DatetimeArray.
  # (例如，一个系列+Timedelta的结果)直接进入一个列槽，所以
  # 我们先提取下限值，然后再分配。
  delta = pd.Timedelta(days=days_difference)
  time_columns_summary = [
      "bed_time",
      "wake_up_time",
      "datetime",
      "date",
  ]
  for col in time_columns_summary:
    if col in summary_df.columns:
      shifted = pd.to_datetime(summary_df[col]) + delta
      summary_df[col] = shifted.values

  # 同时更新索引以保持与日期时间列同步
  if isinstance(summary_df.index, pd.DatetimeIndex):
    summary_df.index = summary_df.index + delta

  time_columns_activity = ["start_time", "end_time"]
  activities_df = activities_df.copy()
  for col in time_columns_activity:
    if col in activities_df.columns:
      shifted = pd.to_datetime(activities_df[col]) + delta
      # 使用plain df[col]=(不是.loc[:,col]=.),所以pandas 4允许。
      # 从str到 日期时 64 的列 Dtype 更改。.loc 设置器
      # 试图保留 dtype 并拒绝日期性任务。
      activities_df[col] = shifted.values
  
  # 同时更新索引以保持与 start_time 列的同步
  if isinstance(activities_df.index, pd.DatetimeIndex):
    activities_df.index = activities_df.index + pd.Timedelta(days=days_difference)

  return PhdaDataFrame(summary_df), PhdaDataFrame(activities_df)

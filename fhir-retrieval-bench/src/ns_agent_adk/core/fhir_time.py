"""Clinical time extraction logic for FHIR resources."""

import calendar
import datetime
import re
from typing import Any, NamedTuple

# 数据类更简单 NamedTuples 或类来避免外部阻隔


class ClinicalInstant(NamedTuple):
  start_time: datetime.datetime
  end_time: datetime.datetime
  fhir_path: str


class ClinicalRange(NamedTuple):
  start: ClinicalInstant
  end: ClinicalInstant | None = None


class ClinicalTime(NamedTuple):
  instant: ClinicalInstant | None = None
  range: ClinicalRange | None = None

  @property
  def start_time(self) -> datetime.datetime | None:
    if self.instant:
      return self.instant.start_time
    if self.range and self.range.start:
      return self.range.start.start_time
    return None

  @property
  def end_time(self) -> datetime.datetime | None:
    if self.instant:
      return self.instant.end_time
    if self.range:
      if self.range.end:
        return self.range.end.end_time
      # 如果射程没有终点，则倒转开始的end_time或仅仅是start_time?.
      # 通常如果它是一个只有起始的射程，则结束是开着的或未知的。
      # 为了进行分类，可以使用start_time.
      # 但让我们检查一下用法。
      return self.range.start.end_time
    return None


# 逻辑的像 Enum 的常数
TYPE_INSTANT = 'ClinicalInstant'
TYPE_RANGE = 'ClinicalRange'


# 地图 FHIR 资源类型 -> (临床时间类型，顺序列表 FHIR 路径)
_CLINICAL_TIME_CONFIG = {
    'Patient': (None, []),
    'Encounter': (
        TYPE_RANGE,
        [
            'actualPeriod.start',
            'actualPeriod.end',
            'plannedStartDate',
            'plannedEndDate',
            'period.start',
            'period.end',
        ],
    ),
    'Condition': (
        TYPE_RANGE,
        ['onsetDateTime', 'onsetPeriod.start', 'recordedDate'],
    ),
    'Procedure': (
        TYPE_RANGE,
        [
            'occurrenceDateTime',
            'occurrencePeriod.start',
            'occurrencePeriod.end',
            'recorded',
            'performedDateTime',
            'performedPeriod.start',
            'performedPeriod.end',
        ],
    ),
    'Observation': (
        TYPE_RANGE,
        [
            'effectiveDateTime',
            'effectivePeriod.start',
            'effectiveInstant',
            'issued',
        ],
    ),
    'AllergyIntolerance': (
        TYPE_RANGE,
        [
            'recordedDate',
            'onsetDateTime',
            'onsetPeriod.start',
            'lastOccurrence',
        ],
    ),
    'Immunization': (
        TYPE_INSTANT,
        ['occurrenceDateTime', 'occurrenceDateTime', 'recorded', 'date'],
    ),
    'MedicationRequest': (
        TYPE_INSTANT,
        [
            'authoredOn',
            'effectiveDosePeriod.start',
            'dispenseRequest.validityPeriod.start',
        ],
    ),
    'MedicationStatement': (
        TYPE_INSTANT,
        ['dateAsserted', 'effectiveDateTime', 'effectivePeriod.start'],
    ),
    'Medication': (None, []),
    'Location': (None, []),
    'Organization': (None, []),
    'PractitionerRole': (None, []),
    'Practitioner': (None, []),
}


def _get_or_none(data: dict[str, Any], *keys: str) -> Any:
  """Helper to traverse nested dictionary."""
  curr = data
  for key in keys:
    if isinstance(curr, dict) and key in curr:
      curr = curr[key]
    else:
      return None
  return curr


def _parse_partial_date(
    groups: tuple[str, str | None, str | None], fhir_path: str
) -> ClinicalInstant:
  """Parses date (YYYY, YYYY-MM, or YYYY-MM-DD) into a ClinicalInstant."""
  year_str, month_str, day_str = groups
  year = int(year_str)

  # 计算起始时间的默认值
  month = int(month_str) if month_str else 1
  day = int(day_str) if day_str else 1

  start_time = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)

  # 根据所提供的精确度计算结束时间
  if day_str:
    # Day (End of that specific day)
    end_time = start_time.replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
  elif month_str:
    # Month (End of that specific month)
    _, last_day = calendar.monthrange(year, month)
    end_time = start_time.replace(
        day=last_day, hour=23, minute=59, second=59, microsecond=999999
    )
  else:
    # Year (End of that specific year)
    end_time = start_time.replace(
        month=12, day=31, hour=23, minute=59, second=59, microsecond=999999
    )

  return ClinicalInstant(
      start_time=start_time,
      end_time=end_time,
      fhir_path=fhir_path,
  )


def _parse_iso_datetime(date_str: str, fhir_path: str) -> ClinicalInstant:
  dt = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=datetime.timezone.utc)
  return ClinicalInstant(
      start_time=dt,
      end_time=dt,
      fhir_path=fhir_path,
  )


def _parse_fhir_datetime(
    date_str: str, fhir_path: str
) -> ClinicalInstant | None:
  """Parses a FHIR date/time string into a ClinicalInstant."""
  try:
    # YYYY,YYYY-MM,或YYYY-MM-DD的备注。
    if match := re.fullmatch(r'^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$', date_str):
      return _parse_partial_date(match.groups(), fhir_path)

    return _parse_iso_datetime(date_str, fhir_path)

  except (ValueError, IndexError):
    print(f'Warning: Failed to parse FHIR date string: {date_str}')
    return None


def get_clinical_time(
    fhir_resource: dict[str, Any],
) -> ClinicalTime:
  """Extracts ClinicalTime from a FHIR resource."""
  resource_type = fhir_resource.get('resourceType')
  if not resource_type:
    return ClinicalTime()

  config = _CLINICAL_TIME_CONFIG.get(resource_type)
  if not config:
    # 倒计时/ 过失行为？
    # 原始代码返回空 ClinicalTime如果找不到配置
    # 但我们想要安全
    return ClinicalTime()

  time_type, paths = config
  if time_type is None:
    return ClinicalTime()

  for path in paths:
    keys = path.split('.')
    value = _get_or_none(fhir_resource, *keys)
    if isinstance(value, str):
      instant = _parse_fhir_datetime(value, path)
      if instant:
        if time_type == TYPE_INSTANT:
          return ClinicalTime(instant=instant)
        elif time_type == TYPE_RANGE:
          return ClinicalTime(range=ClinicalRange(start=instant))

  # 检查一下是否要登录失败？
  return ClinicalTime()

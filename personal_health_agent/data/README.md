# PHA 数据格式文档

本目录记录个人健康 Agent（PHA）的输入数据格式。 `sample/` 提供用于测试和演示的合成数据。

## 概览

PHA 读取四个 CSV 文件。

| 文件 | 用途 | 行数 |
|------|---------|------|
| `summary.csv` | 每日汇总的健康指标 | 每天一行 |
| `activities.csv` | 个人锻炼/活动记录 | 每个活动一行 |
| `profile.csv` | 用户人口学信息 | 一行 |
| `population_percentiles.csv` | 人群参考数据 | 每个百分位数、年龄组和性别组合一行 |

## 数据字段

### summary.csv（每日摘要）

每日健康指标汇总自可穿戴设备。每行代表一天。

| 列名 | 类型 | 说明 |
|--------|------|-------------|
| `datetime` | datetime | 记录日期，用作索引 |
| `steps` | float | 当天步数总计 |
| `sleep_minutes` | float | 总睡眠时长（分钟） |
| `bed_time` | datetime | 就寝时间 |
| `wake_up_time` | datetime | 起床时间 |
| `resting_heart_rate` | float | 平均静息心率（次/分钟） |
| `heart_rate_variability` | float | 心率变异性（毫秒） |
| `active_zone_minutes` | float | 活跃区间分钟数合计 |
| `deep_sleep_minutes` | float | 深睡时长（分钟） |
| `rem_sleep_minutes` | float | 快速眼动（REM）睡眠时长（分钟） |
| `light_sleep_minutes` | float | 浅睡时长（分钟） |
| `awake_minutes` | float | 睡眠期间清醒的时长（分钟） |
| `stress_management_score` | float | 压力管理评分，越高表示管理得越好 |
| `fatburn_active_zone_minutes` | float | 脂肪燃烧心率区间的时长（分钟） |
| `cardio_active_zone_minutes` | float | 有氧运动心率区间的时长（分钟） |
| `peak_active_zone_minutes` | float | 峰值心率区间的时长（分钟） |
| `sleep_score` | float | 总体睡眠质量评分（可选） |
| `readiness_score` | float | 每日准备度评分（可选） |
| `cardio_load_total` | float | 总心肺负荷，单位为 TRIMP（可选） |

**示例：**
```csv
datetime,steps,sleep_minutes,bed_time,wake_up_time,resting_heart_rate,heart_rate_variability
2024-01-15,8523,425,2024-01-14 23:15:00,2024-01-15 06:20:00,62,45
2024-01-16,10241,390,2024-01-15 23:45:00,2024-01-16 06:15:00,60,48
```

### activities.csv（活动记录）

每行记录一次锻炼或活动。

| 列名 | 类型 | 说明 |
|--------|------|-------------|
| `start_time` | datetime | 活动开始时间，用作索引 |
| `end_time` | datetime | 活动结束时间 |
| `activity_name` | string | 活动类型，例如 `RUNNING`、`WALKING` 或 `CYCLING` |
| `duration` | float | 时长（分钟） |
| `distance` | float | 距离（英里） |
| `averageHeartRate` | float | 活动期间的平均心率（次/分钟） |
| `calories` | float | 消耗的热量 |
| `steps` | float | 活动期间的步数（若适用） |
| `active_zone_minutes` | float | 活动获得的活跃区间分钟数 |
| `speed` | float | 平均速度（英里/小时，可选） |
| `cardio_load` | float | 心肺负荷，单位为 TRIMP（可选） |
| `elevation_gain` | float | 累计爬升（米，可选） |

**支持的活动名称：**
`WALKING` , `RUNNING` , `OUTDOOR_BIKE` , `BIKING` , `CYCLING` , `WORKOUT` , `TREADMILL` , `SWIMMING` , `YOGA` , `STRENGTH_TRAINING` , `HIIT` , `ELLIPTICAL` , `ROWING_MACHINE` , `HIKING` 还有其他的

**示例：**
```csv
start_time,end_time,activity_name,duration,distance,averageHeartRate,calories,steps
2024-01-15 07:30:00,2024-01-15 08:00:00,RUNNING,30,3.1,145,320,4200
2024-01-15 18:00:00,2024-01-15 18:45:00,WALKING,45,2.2,95,180,4800
```

### profile.csv（用户档案）

一行记录用户的年龄、性别、身高、体重等资料。

| 列名 | 类型 | 说明 |
|--------|------|-------------|
| `age` | int | 年龄（岁） |
| `gender` | string | `male` 或 `female` |
| `height_cm` | int | 身高（厘米） |
| `weight_kg` | int | 体重（千克） |
| `averageDailySteps` | int | 历史日均步数（可选） |
| `cluster` | string | 用户所属的画像分组（可选） |
| `elderly` | string | `Yes` 或 `No`（可选） |

**示例：**
```csv
age,gender,height_cm,weight_kg,averageDailySteps
34,male,178,75,8156
```

### population_percentiles.csv（人群参考数据）

人群统计数据，用于比较用户和人口学特征相近的人群。例如，Agent 可以说明静息心率在同年龄组中处于第 75 百分位。

| 列名 | 类型 | 说明 |
|--------|------|-------------|
| `percentile` | int | 百分位数（1–100） |
| `age_group` 或 `age` | string | 年龄区间，例如 `18-24`、`25-34`、`35-44`、`45-54`、`55-64` 或 `65+` |
| `gender` | string | `male` 或 `female` |
| `resting_heart_rate` | float | 该百分位数的静息心率 |
| `heart_rate_variability` | float | 该百分位数的心率变异性 |
| `steps` | float | 该百分位数的日步数 |
| `rem_sleep_minutes` | float | 该百分位数的 REM 睡眠时长 |
| `deep_sleep_minutes` | float | 该百分位数的深睡时长 |
| `light_sleep_minutes` | float | 该百分位数的浅睡时长 |
| `stress_management_score` | float | 该百分位数的压力管理评分 |
| `fatburn_active_zone_minutes` | float | 该百分位数处于脂肪燃烧区间的活跃分钟数 |
| `cardio_active_zone_minutes` | float | 该百分位数处于有氧运动区间的活跃分钟数 |
| `peak_active_zone_minutes` | float | 该百分位数处于峰值区间的活跃分钟数 |

**示例：**
```csv
percentile,age_group,gender,resting_heart_rate,heart_rate_variability,steps
1,25-34,male,51,15,1200
50,25-34,male,68,32,7500
99,25-34,male,85,65,15000
```

## 使用您自己的数据

### 方法 1：替换样本文件

用自己的 CSV 文件替换 `data/sample/` 中对应的文件，保持字段结构一致。

### 方法 2：指定数据目录

```python
from config.settings import Settings

settings = Settings(data_dir="/path/to/your/data")
```

### 方法 3：直接传入 DataFrame

```python
from pha.agents import DataScienceAgent
import pandas as pd

agent = DataScienceAgent()
agent.load_dataframes({
    'summary_df': pd.read_csv('my_summary.csv'),
    'activities_df': pd.read_csv('my_activities.csv'),
    'profile_df': pd.read_csv('my_profile.csv'),
    'population_df': pd.read_csv('my_population.csv'),
})
```

## 对齐样本数据的日期

PHA 默认平移所有记录的日期，使最新一条记录对齐到今天，同时保留记录之间的时间间隔。如果需要使用原始日期，请设置：

```python
settings = Settings(temporally_localize=False)
```

## 样本数据

`sample/` 包含为演示构造的健康数据。档案中的人物设定为 34 岁男性。

## 导入其他数据源

可以先从设备或应用导出，再转换成上面的 CSV 格式。
- **Fitbit**：使用 Google Takeout 或 Fitbit 的数据导出功能。
- **Apple 健康**：使用「导出所有健康数据」。
- **Garmin**：通过 Garmin Connect 导出。
- **其他设备**：查看设备或配套应用是否提供数据导出。

转换时请核对时间格式、字段名称和计量单位。

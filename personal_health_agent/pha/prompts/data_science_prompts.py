"""数据科学 Agent 的提示词，覆盖分析计划、代码生成、修正和结果说明。"""

from typing import Any, Dict, Final, List
import jinja2


# =============================================================================
# 快速渲染工具
# =============================================================================

def render_prompt(template: str, **kwargs) -> str:
    """使用给定变量渲染 Jinja2 模板。"""
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    jinja_template = env.from_string(template)
    return jinja_template.render(**kwargs)


# =============================================================================
# 分析方案评价标准
# =============================================================================

RUBRIC_QUESTIONS: Dict[str, Dict[str, str]] = {
    "data_answerable": {
        "answerable": "若过去一年的各表数据都完整，现有字段能否回答这个问题？例如，人群表没有体重字段时，无法比较用户和人群的体重。",
    },
    "timeframe": {
        "identify": "方案是否界定了具体的时间范围，例如最近几个月？",
        "identify_reasonableness": "如果界定了时间范围，范围是否合理？",
        "identify_clear": "如果界定了时间范围，起止点是否明确？不要加入额外假设。",
        "improvement": "补充时间范围能否让回答更有用？",
    },
    "data_transformations": {
        "appropriate_columns": "所用字段是否存在且适合分析？即使没有写出字段名，也应能明确判断使用哪张表的哪一列。",
        "hardcoded_values": "除时间范围外，方案是否写死了筛选值或阈值，例如用固定心率识别节奏跑？",
        "hardcoded_grounded": "如果有固定值，该值是否来自用户问题或已有公认标准？",
        "made_transformations": "除选择时间范围和计算统计量外，方案是否进行了数据变换？",
        "made_transformations_necessary": "如果没有，回答问题是否需要数据变换？",
        "made_transformations_clear": "如果有，是否写明变换办法，让不同分析人员可以得到同样结果？不要加入额外假设。",
        "made_transformations_logical_errors": "变换是否有逻辑错误？例如，问题问步数与前一晚睡眠的关系，却把当日睡眠与当日步数匹配。",
        "made_transformations_reference_nonexistant_columns": "变换是否引用了不存在的字段？",
    },
    "data_distribution_availability": {
        "columns_data_availability": "需要统计分析时，方案是否检查记录行数是否充足？",
        "columns_data_availability_reasonable": "假设所用字段数据齐全，记录行数的检查是否合理？",
        "columns_specific_data_availability": "方案是否检查用于计算统计量或检验的具体字段中，可用值是否充足？",
        "columns_specific_data_availability_clear": "是否明确指出要检查哪些列，以及如何计数或判断数据是否充足？",
        "columns_specific_data_availability_reasonable": "检查条件是否合理，有没有过严或过宽？",
        "columns_specific_data_availability_sufficient": "是否检查了所有相关字段？",
    },
    "data_statistics": {
        "summary_statistics": "方案是否计算均值、中位数、众数、极值、标准差、百分位数、计数或总和等描述性统计量？",
        "summary_statistics_reasonable": "如果计算了，这些统计量是否适合该问题？",
        "summary_statistics_necessary": "如果没有，加入描述性统计量能否改善分析？",
        "summary_statistics_compared_to_hardcoded_values": "是否将统计量与固定值比较？例如，用标准差小于等于 1000 判断步数是否稳定。",
        "summary_statistics_compared_to_hardcoded_values_grounded": "如果使用固定值，比较依据是否来自用户问题或公认标准？",
        "summary_statistics_data_distribution": "是否考虑了数据分布？例如，偏态数值可能更适合用中位数，对分类数据求均值通常不合适。",
        "summary_statistics_data_distribution_reasonable": "如果考虑了分布，处理方式是否合理？",
        "summary_statistics_data_distribution_suitable": "计算这些统计量时，是否需要考虑数据分布？",
        "statistical_test": "方案是否使用了 t 检验、方差分析或相关分析等统计方法？",
        "statistical_test_reasonable": "如果使用了，该方法是否适合当前问题？",
        "statistical_test_necessary": "如果没有，增加统计检验能否改善分析？",
        "statistical_test_clear": "是否给出清晰可复现的检验步骤？不要加入额外假设。",
        "statistical_test_data_distribution": "是否考虑了统计方法对数据分布的假设？",
        "statistical_test_data_distribution_suitable": "对这些分布条件的考虑是否适当？",
    },
    "summary": {
        "align_intents": "整体方案是否符合用户意图，能否回答用户真正想了解的问题？",
    },
}


def _serialize_rubric_questions(num_tabs: int = 1) -> str:
    """将分析方案的评价问题排版为文本。"""
    sections = []
    indent = "\t" * num_tabs

    for section, questions in RUBRIC_QUESTIONS.items():
        if section == "data_answerable":
            continue
        section_text = f"\n{indent}**{section.replace('_', ' ').title()}**\n"
        for q_text in questions.values():
            section_text += f"\n{indent}- {q_text}"
        sections.append(section_text)

    return "\n".join(sections)


RUBRIC_QUESTIONS_STR: Final[str] = _serialize_rubric_questions()


# =============================================================================
# 少样本示例(来自few_shots/approach.py).
# =============================================================================

APPROACH_EXEMPLARS: List[Dict[str, str]] = [
    {
        "query": "从应用里的记录看，我的骑行水平比六个月前有进步吗？",
        "discussion": (
            "「有进步」需要量化。可以用单次里程、平均速度和心率对比两个时段："
            "一年前至六个月前，以及最近六个月。计算统计量前，需检查各时段的数据量和缺失值。"
        ),
        "approach": (
            "1. 用单次骑行里程、速度和心率衡量表现。"
            "2. 筛选 OUTDOOR_BIKE 或 BIKING，按两个时段分组。"
            "3. 每组至少需要五次活动，否则报告数据不足。"
            "4. 检查缺失值，排除缺失比例超过 50% 的指标。"
            "5. 计算可用指标的描述性统计量。"
            "6. 用 Shapiro-Wilk 检验检查正态性。"
            "7. 正态数据用 t 检验，其他数据用 Mann-Whitney U 检验。"
            "8. 返回统计量、缺失值数量和检验结果，并标记被排除的指标。"
        ),
    },
    {
        "query": "我的心率和朋友相比怎样？",
        "discussion": "现有记录没有朋友的心率，无法比较。",
        "approach": "1. 说明只有用户本人及参考人群的数据，未提供朋友的数据。",
    },
]


# =============================================================================
# APPROACH PROMPTS(来自approach_preamble.py).
# =============================================================================

APPROACH_PREAMBLE_BASELINE: Final[str] = """\
你是数据科学专家。根据用户问题制定逐步分析方案。只给出一段讨论和一段方法说明。

问题：{{query}}
== DISCUSSION ==
"""

APPROACH_PREAMBLE_V3: Final[str] = """\
你熟悉 Python 和 Fitbit 健康时间序列。仅根据用户问题和下面的数据结构制定明确、可复现的分析计划，供后续生成代码。计划中不要写代码。

**可用数据结构：**
{{data_summary}}

**用户问题：**
{{query}}

先给出 == Discussion ==，讨论问题能否用现有数据回答。找出「更好」「活跃」「规律」「经常」等含义不明确的词。说明要用哪些字段、怎么算，才能量化这些词。例如，用骑行记录的平均速度（distance / duration）与 active_zone_minutes 衡量。随后概括总体策略。如果缺少数据（例如只有用户本人的记录，没有朋友的数据），或概念无法明确定义，说明为什么无法回答。

再给出 == Approach ==。无法回答时只列一步，明确缺少的数据或其他原因。可以回答时，按顺序列出以下步骤：

1. **时间范围**：给出确切的起止日期或时间。没有指定时，使用相关 DataFrame 中最近一年可用数据，并明确分析区间。
2. **数据选择和筛选**：明确需要哪些 DataFrame 和字段，以及时间、活动类型等筛选条件。例如活动类型为 OUTDOOR_BIKE 或 BIKING，日期落在给定范围。
3. **数据变换**：说明计算公式、对齐方式和单位。例如用 distance / (duration / 60) 计算速度，将日步数按周汇总，或将 sleep_minutes 滞后一天与下日步数对齐。
4. **数据充足性**：指定并说明最低数据要求，例如每个对比区间至少有五条记录。明确如何检查关键字段的空值。例如某区间 resting_heart_rate 缺失超过 50% 时，排除该区间的这一指标并说明原因。数据不足时返回说明。
5. **统计分析**：说明需要哪些统计量和选择理由，例如分布可能偏斜时使用中位数。如果需要 t 检验、相关分析或方差分析，写明检验、变量、适用条件及检查方法。例如按组进行 Shapiro-Wilk 正态性检验。
6. **返回结果**：明确应返回的数据、统计量或检验结果，例如每月步数中位数或平均心率比较的 p 值。计划本身无需解释或评判结果。

让其他分析人员能依照步骤得到相同数值。只用给定的数据和问题提供的信息，避免额外假设。阈值无法确定时，说明限制，或使用确有依据且可用的标准。分析仅返回回答问题所需的数据或统计量。

输出恰好一个 == Discussion == 和一个 == Approach ==，不输出代码。
问题：{{query}}
== Discussion ==
"""


# =============================================================================
# CODEGEN PROMPTS
# =============================================================================

# DataFrame 嵌入在即时的图案
DATAFRAME_SCHEMAS: Final[str] = """
# summary_df：每日摘要
汇总活动、睡眠与其他健康记录，每日一行。
索引名为 `datetime`，值为记录日期（datetime64[ns]）。
包含以下字段：
- steps: 步数（float64）。
- sleep_minutes: 总睡眠时长（分钟，float64）。
- bed_time: 就寝时间（datetime64[ns]）。
- wake_up_time: 起床时间（datetime64[ns]）。
- resting_heart_rate: 静息心率（bpm，float64）。
- heart_rate_variability: 心率变异性（float64）。
- active_zone_minutes: 活跃区间分钟数（float64）。
- deep_sleep_minutes: 深睡时长（分钟，float64）。
- rem_sleep_minutes: REM 睡眠时长（分钟，float64）。
- light_sleep_minutes: 浅睡时长（分钟，float64）。
- awake_minutes: 清醒时长（分钟，float64）。
- stress_management_score: 压力管理得分，越高表示管理得越好（float64）。
- fatburn_active_zone_minutes: 脂肪燃烧心率区间的时长（分钟，float64）。
- cardio_active_zone_minutes: 有氧运动心率区间的时长（分钟，float64）。
- peak_active_zone_minutes: 峰值心率区间的时长（分钟，float64）。

# activities_df：活动记录
每行记录一次活动。
索引名为 `start_time`，值为活动开始时间（datetime64[ns]）。
包含以下字段：
- start_time: 活动开始时间（datetime64[ns]）。
- end_time: 活动结束时间（datetime64[ns]）。
- activity_name: 活动类型。示例："WALKING", "RUNNING", "OUTDOOR_BIKE", "STRENGTH_TRAINING", "WORKOUT", "ELLIPTICAL", "YOGA", "TREADMILL", "HIKING", "SWIMMING".
- distance: 活动距离（英里，float64）。
- duration: 活动时长（分钟，float64）。
- calories: 活动消耗的热量（float64）。
- steps: 步数（float64）。
- active_zone_minutes: 活跃区间分钟数（float64）。

# profile_df：用户档案
只有一行，包含以下字段：
- age: 年龄（int）。
- gender: 性别（字符串 "male" 或 "female"）。
- height_cm: 身高（厘米，int）。
- weight_kg: 体重（千克，int）。
- averageDailySteps: 历史日均步数（int）。
- cluster: 用户活动水平的分组标签（object）。
- elderly: 是否归入老年人群（bool）。

# population_df：人群参考数据
按年龄段、性别和百分位提供模拟的人群健康与活动数据。
对比参考人群前，用执行环境预先加载的 `age_to_age_group(age)` 将年龄映射到年龄段，例如 34 对应 "25-34"。不需要重新定义该函数。
包含以下字段：
- percentile: 百分位数（1–100，int）。
- age: 年龄（int）。
- gender: 性别（字符串 "male" 或 "female"）。
- resting_heart_rate: 静息心率（bpm，float64）。
- heart_rate_variability: 心率变异性（float64）。
- steps: 步数（float64）。
- fatburn_active_zone_minutes: 脂肪燃烧心率区间的时长（分钟，float64）。
- cardio_active_zone_minutes: 有氧运动心率区间的时长（分钟，float64）。
- peak_active_zone_minutes: 峰值心率区间的时长（分钟，float64）。
- rem_sleep_minutes: REM 睡眠时长（分钟，float64）。
- deep_sleep_minutes: 深睡时长（分钟，float64）。
- light_sleep_minutes: 浅睡时长（分钟，float64）。
- stress_management_score: 压力管理得分，越高表示管理得越好（float64）。
"""

# V2 代码提示
CODEGEN_PREAMBLE_V2: Final[str] = """你熟悉 Python、pandas 和 Fitbit 时间序列。根据以下数据结构和方案，编写完成分析任务的 Python 函数。留意列名、类型和时间索引。

<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>

<Approach>
{{approach}}
</Approach>

- 严格实现 <Approach> 中的分析，不构造额外数据。
- 使用已有 DataFrame：summary_df、activities_df、profile_df、population_df。
- 用日期时间索引做筛选和汇总，除非结构明确给出同名列。
- 在开头添加必要导入。将所需结果整理成 DataFrame、计算值或列表，并从 analysis 函数返回。
- 函数签名如下。生成的代码中不写注释或 docstring。尽量使用 pandas 向量化操作。
- 调用预加载的 age_to_age_group(age: int) -> str，无需重写。它将年龄转换为年龄段，如 34 转为 "25-34"。
- 函数体少于 80 行，避免冗余检查。缺少对应人群数据时，只返回用户自己的数据，不因缺少参考值而报错。

生成代码：

```python
from typing import Any, Dict
import pandas as pd
import numpy as np
import datetime

def analysis(summary_df: pd.DataFrame | None = None,
    activities_df: pd.DataFrame | None = None,
    profile_df: pd.DataFrame | None = None,
    population_df: pd.DataFrame | None = None,) -> Dict[str, Any]:
"""

# V1 代码提示( 默认情况下没有使用)
CODEGEN_PREAMBLE: Final[str] = """\
你负责用 Python 分析 Fitbit 健康时间序列。依据用户问题和分析方案写出所需代码，保留方案的原意和回答所需的计算结果与背景。

可用数据：
{{data_summary}}

除 Python 内置库外，只可使用 matplotlib、pandas、numpy、scipy。系统不会读取打印内容，只读取代码末尾的数据结构。将答案和相关信息存入 result。

问题：{{query}}
方案：{{approach}}
可用 DataFrame 变量如下，请直接使用，不要构造示例数据：
{{dfs_info}}
代码：
```python
"""


CODEGEN_PREAMBLE_BASELINE: Final[str] = """\
根据数据结构和分析方案编写 Python，回答有关 Fitbit 健康和运动时间序列的问题。阅读 analysis 函数的 docstring，遵照其中的任务说明。

<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>

问题：{{query}}
方案：{{approach}}

只返回修正后的代码，封装在 ```python ... ``` 中，无需解释。

生成的代码：
```python
"""

DEBUG_CODEGEN_PREAMBLE_BASELINE: Final[str] = """\
根据以下说明、待修正代码和错误信息修复代码。

代码目标：
{{code_instructions}}
代码：
{{code}}
错误：
{{error}}

只返回完整的修正代码，使用 ```python ... ``` 代码块，无需解释。

修正后的代码：
```python
"""

CODE_REFINEMENT_PREAMBLE: Final[str] = """你负责根据反馈改进已经成功运行的 Fitbit 数据分析代码。只修改反馈指出的内容，保留其他正确逻辑。

- 使用原有 analysis 签名。代码中不添加注释，原代码没有 try/except 时也无需添加。
- 函数体少于 80 行。直接调用预加载的 age_to_age_group，不要重写。
- 无需引入额外分析或功能。

<Dataframe schemas>
""" + DATAFRAME_SCHEMAS + """
</Dataframe schemas>

**1. 分析目标**
{{code_instructions}}

**2. 已成功运行的代码**
{{code}}

**3. 执行输出**
{{execution_results}}

**4. 修正反馈**
{{code_update_feedback}}

只返回修正后的代码，封装在 ```python ... ``` 中，无需解释。

修正后的 Python 代码：
```python
"""


DEBUG_CODEGEN_PREAMBLE: Final[str] = """\
你负责修复分析 Fitbit 时间序列的 Python 代码。参考用户的问题、分析方案、程序和执行报错，保留原有目标，返回完整的修正代码。

可用数据：
{{data_summary}}

- 回答所需的数据放在代码最后一行对应的 result 变量里，该行不能在函数内。
- 除 Python 内置库外，仅可使用 matplotlib、pandas、numpy、scipy。
- 系统不读取打印内容，只读取 result。
- 只修复当前错误，避免添加原程序没有的分析、图表或功能。

代码目标：
{{code_instructions}}
代码：
{{code}}
错误：
{{error}}
修正后的代码：
```python
"""


DEBUG_CODEGEN_PREAMBLE_V2: Final[str] = """
你负责调试分析 Fitbit 时间序列的 Python 代码。检查原始代码目标、代码和真实执行结果，定位错误或结果偏差，只返回完整的修正代码。

- 实现原始提示词规定的分析，直接解决执行结果显示的问题。
- 尽量保留原有意图、逻辑、结构和风格。
- 只做所需修复，避免添加原程序没有的分析、可视化、try/except 或其他功能。原程序过长时，可在保留正确性的前提下简化。

**1. 原始代码目标**
{{code_instructions}}

**2. 待修正代码**
{{code}}

**3. 执行结果或错误堆栈**
{{code_result}}

**4. 代码修正反馈（如有）**
{{code_update_feedback}}

修正后的 Python 代码：
```python

"""


# =============================================================================
# INTERPRETATION PROMPT(来自interpretation_preamble.py).
# =============================================================================

INTERPRETATION_PREAMBLE: Final[str] = """\
你负责核对 Fitbit 健康时间序列的计算结果，并用用户能理解的语言说明数值。

- 如果分析正确且无需改代码，在原问题的背景下说明结果。用户看不到原始执行输出，需要用清楚的语言概括。只解释计算所得数值，不作医学解读。以 [SUMMARY] 开头，使用用户最近一轮发言的语言，除非用户另有要求。
- 如果结果疑似由代码逻辑错误引起，例如本应有数值却出现 NaN，说明问题，用英文提出代码修正建议。以 [CODE UPDATE] 开头。

问题：{{query}}
分析方案：
{{approach}}
代码：
{{code}}
执行结果：
{{execution_results}}
回复：
"""


# =============================================================================
# 即时生成函数(简化自build_prompt.py)
# =============================================================================

def generate_approach_prompt_baseline(
    query: str,
) -> str:
    """Create the prompt for generating the approach (baseline / eval).

    Used by the codegen eval harness when the harness wants a minimal,
    schema-free approach generation step.
    """
    return render_prompt(
        APPROACH_PREAMBLE_BASELINE,
        query=query,
    )


def generate_approach_prompt(
    query: str,
    dfs_str: str,
) -> str:
    """Create the prompt for generating the approach."""
    return render_prompt(
        APPROACH_PREAMBLE_V3,
        query=query,
        data_summary=dfs_str,
        rubric=RUBRIC_QUESTIONS_STR,
    )


def generate_codegen_prompt(
    query: str,
    approach_str: str,
    dfs_str: str,
    dfs_info_str: str,
) -> str:
    """Create the prompt for generating the code.

    Uses V2 prompt which requires def analysis() function wrapper.
    The V2 prompt has DataFrame schemas embedded, so we only need approach.
    """
    return render_prompt(
        CODEGEN_PREAMBLE_V2,
        approach=approach_str,
    )


def generate_codegen_prompt_baseline(
    query: str,
    approach_str: str,
) -> str:
    """Create the prompt for generating the code (baseline / eval).

    Used when *approach_str* is a Python function stub (docstring_prompt
    format) — the baseline preamble treats the stub as the authoritative
    spec rather than re-imposing V2's return-type/signature constraints.
    """
    return render_prompt(
        CODEGEN_PREAMBLE_BASELINE,
        query=query,
        approach=approach_str,
    )


def generate_debug_prompt(
    approach_str: str,
    code: str,
    error: str,
    dfs_str: str,
) -> str:
    """Create the prompt for debugging the code."""
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE,
        code_instructions=approach_str,
        data_summary=dfs_str,
        code=code,
        error=error,
    )


def generate_debug_prompt_baseline(
    approach_str: str,
    code: str,
    error: str,
    dfs_str: str,
) -> str:
    """Create the prompt for debugging the code (baseline / eval).

    Neutral debug prompt that shows the stub, the code, and the error
    without imposing a competing function signature.
    """
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE_BASELINE,
        code_instructions=approach_str,
        data_summary=dfs_str,
        code=code,
        error=error,
    )


def generate_debug_prompt_v2(
    approach_str: str,
    code: str,
    code_result: str,
    code_update_feedback: str = "",
) -> str:
    """Create the prompt for debugging the code (v2)."""
    return render_prompt(
        DEBUG_CODEGEN_PREAMBLE_V2,
        code_instructions=approach_str,
        code=code,
        code_result=code_result,
        code_update_feedback=code_update_feedback,
    )


def generate_code_refinement_prompt(
    approach_str: str,
    code: str,
    execution_results: str,
    code_update_feedback: str,
) -> str:
    """Create the prompt for refining successfully-executing code.

    Used when code ran without error but the interpreter identified a logical
    issue. Unlike generate_debug_prompt_v2, this prompt correctly frames the
    task as refinement rather than error-fixing.
    """
    return render_prompt(
        CODE_REFINEMENT_PREAMBLE,
        code_instructions=approach_str,
        code=code,
        execution_results=execution_results,
        code_update_feedback=code_update_feedback,
    )


def generate_result_communication_prompt(
    query: str,
    approach_str: str,
    code: str,
    execution_results: str,
) -> str:
    """Create the prompt for generating the result communication."""
    return render_prompt(
        INTERPRETATION_PREAMBLE,
        query=query,
        approach=approach_str,
        code=code,
        execution_results=execution_results,
    )

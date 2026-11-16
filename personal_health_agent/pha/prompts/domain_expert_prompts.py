"""领域专家 Agent 的医学背景、工具说明和回答模板。"""

from typing import Final, Dict, List
import jinja2


def render_prompt(template: str, **kwargs) -> str:
    """使用给定变量渲染 Jinja2 模板。"""
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    jinja_template = env.from_string(template)
    return jinja_template.render(**kwargs)


# =============================================================================
# 领域专家系统提示
# =============================================================================

DOMAIN_EXPERT_SYSTEM_PROMPT: Final[str] = """\
你是内科和健康领域专家，擅长解释来自不同来源的健康数据。为用户说明指标所处的背景，并给出全面、个性化的回答。

## 可用工具
1. 网页搜索（search）：查找权威健康资料。
2. Data Commons 查询（datacommons_query）：查询公共卫生统计。
3. Python 代码（python_code）：执行计算。
4. 参考范围检查（check_reference_range）：将结果与临床参考范围比较。

## 人群参考数据
按年龄段和性别划分的百分位数可用于解读可穿戴设备及生活方式数据。参考指标包括静息心率（RHR）、心率变异性（HRV）、每日步数、睡眠时长、体质指数（BMI）和胰岛素抵抗指标（HOMA-IR）。比较时选择符合用户年龄段和生理性别的统计数据。

## 回答要求
- 给出回答问题所需的背景，结合用户的年龄、性别、BMI 和生活方式。
- 依据可信来源（如 CDC、WHO、Mayo Clinic 或同行评议期刊），不编造资料，尽可能注明出处。
- 使用准确的医学术语并解释术语含义。区分临床参考范围与理想范围。
- 说明具体数值或阈值，避免只写「略高」。按证据说明风险和收益，保持适度。
- 考虑用户是否有条件执行建议。使用用户最近一轮发言的语言，除非用户另有要求。

## 总结结构
- **总体概览**：概括不同来源的发现，指出最重要和需要关注的事项。
- **详细分析与背景**：依据医学研究说明不同数据之间的关系，保持简洁。
- **缺失数据**：仅当补充数据能明显改善分析时才列出，并且应是用户能够合理取得的数据。
- **行动建议**：依据分析给出可执行、贴合用户情况的建议。
- **参考资料**：列出本次实际使用的搜索结果链接。

你是多 Agent 团队的一员。复杂计算可建议编排器请数据科学 Agent 处理，详细行为计划可请健康教练 Agent 制定。搜索时优先使用权威来源，不编造链接或引用。
"""


# =============================================================================
# 工具描述
# =============================================================================

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "search": """搜索健康问题，返回权威来源的链接和相关摘要。
示例：search("normal HDL cholesterol range")""",

    "datacommons_query": """从 Data Commons 查询公共卫生统计，返回 CDC、WHO 等权威机构的数据。
示例：datacommons_query("heart disease mortality rate by age in US")""",

    "python_code": """运行 Python 代码进行计算和分析。
Example: python_code("bmi = 75 / (1.75 ** 2); print(f'BMI: {bmi:.1f}')")
返回代码的执行结果。""",

    "check_reference_range": """检查健康指标是否处于临床参考范围。
Example: check_reference_range([(150, 0, 200), (45, 40, None)])  # LDL, HDL
返回各数值是否在范围内，以及与边界的接近程度。""",
}


# =============================================================================
# 领域专家少样本示例
# =============================================================================

DOMAIN_EXPERT_EXAMPLES: List[Dict[str, str]] = [
    {
        "query": "My LDL cholesterol is 142 mg/dL. Is this concerning?",
        "thought": "I need to check the clinical reference ranges for LDL cholesterol and consider the user's overall cardiovascular risk profile.",
        "action": "search('LDL cholesterol reference ranges cardiovascular risk')",
        "observation": "According to the American Heart Association, LDL levels are categorized as: Optimal (<100 mg/dL), Near optimal (100-129 mg/dL), Borderline high (130-159 mg/dL), High (160-189 mg/dL), Very high (≥190 mg/dL).",
        "response": """Your LDL cholesterol of 142 mg/dL falls in the **borderline high** category according to American Heart Association guidelines.

**Context**: LDL (Low-Density Lipoprotein) is often called "bad cholesterol" because elevated levels can lead to plaque buildup in arteries, increasing cardiovascular disease risk.

**Clinical Significance**: While 142 mg/dL isn't in the high-risk category, it's above the optimal range of <100 mg/dL. The significance depends on your overall cardiovascular risk profile, including:
- Age and sex
- Blood pressure
- Smoking status
- Family history of heart disease
- HDL cholesterol and triglyceride levels

**Recommendations**:
1. Discuss with your healthcare provider about your 10-year cardiovascular risk score
2. Consider lifestyle modifications: Mediterranean diet, regular exercise, weight management
3. Your provider may recommend statins depending on your overall risk profile

**Sources**: American Heart Association cholesterol guidelines"""
    },
    {
        "query": "I'm a 45-year-old male. My resting heart rate is 72 bpm. How does this compare to others my age?",
        "thought": "I should look up normal resting heart rate ranges by age and sex, and put the user's value in context.",
        "action": "search('resting heart rate normal range by age male')",
        "observation": "Normal resting heart rate for adults ranges from 60-100 bpm. Athletes may have lower rates (40-60 bpm). Studies show average RHR for men 45-54 is approximately 70-72 bpm.",
        "response": """Your resting heart rate of 72 bpm is **within the normal range** and typical for a 45-year-old male.

**Reference Ranges**:
- Normal adult RHR: 60-100 bpm
- Average for men age 45-54: ~70-72 bpm
- Well-trained athletes: 40-60 bpm

**Your Position**: Your RHR of 72 bpm is right at the population average for your demographic, indicating normal cardiovascular function.

**What RHR Indicates**: Resting heart rate reflects cardiovascular efficiency. Lower RHR (within normal limits) often indicates better cardiovascular fitness, as the heart pumps more blood per beat.

**Optimization Opportunities**:
Regular aerobic exercise can gradually lower RHR by 1-2 bpm per month of consistent training. A RHR in the 60-65 range would indicate above-average cardiovascular fitness.

**Sources**: American Heart Association, Cleveland Clinic"""
    },
]


# =============================================================================
# 查询提示模板
# =============================================================================

DOMAIN_EXPERT_QUERY_PROMPT: Final[str] = """\
{{system_prompt}}

{% if user_profile %}
## 用户档案
{{user_profile}}
{% endif %}

{% if user_health_data %}
## 用户健康数据
{{user_health_data}}
{% endif %}

{% if population_context %}
## 人群参考数据
{{population_context}}
{% endif %}

## 用户问题
{{query}}

结合已有数据和专业知识分析问题。如需更多信息，说明可使用哪些工具。按上文要求给出全面、个性化的回答。
"""


# =============================================================================
# 参考范围检查逻辑
# =============================================================================

def check_reference_ranges(
    values_and_intervals: List[tuple],
    threshold_factor: float = 0.05,
) -> List[tuple]:
    """Check if values are within clinical reference ranges.
    
    This function checks a list of values against their reference intervals
    and provides information about proximity to boundaries.
    
    Args:
        values_and_intervals: List of tuples (value, interval_start, interval_end).
            Use None for one-sided intervals.
        threshold_factor: Factor for determining "closeness" to boundaries (default 5%).
    
    Returns:
        List of tuples: (in_range_status, proximity_description)
    
    Example:
        >>> check_reference_ranges([(150, 0, 200), (45, 40, None)])
        [('Within Clinical Reference Range', 'Within interval'),
         ('Within Clinical Reference Range', 'Within interval')]
    """
    results = []
    for value, interval_start, interval_end in values_and_intervals:
        if interval_start is None:
            # 单侧间隔( 仅绑上)
            delta_to_end = abs(value - interval_end)
            within_interval = value <= interval_end
            if delta_to_end <= threshold_factor * abs(interval_end) if interval_end != 0 else delta_to_end < threshold_factor:
                proximity = "Close to upper bound"
            elif value > interval_end:
                proximity = "Above upper bound"
            else:
                proximity = "Within interval"
        elif interval_end is None:
            # 单边间隔( 仅下方绑定)
            delta_to_start = abs(value - interval_start)
            within_interval = value >= interval_start
            if delta_to_start <= threshold_factor * abs(interval_start) if interval_start != 0 else delta_to_start < threshold_factor:
                proximity = "Close to lower bound"
            elif value < interval_start:
                proximity = "Below lower bound"
            else:
                proximity = "Within interval"
        else:
            # 双面间隔
            within_interval = interval_start <= value <= interval_end
            delta_to_start = abs(value - interval_start)
            delta_to_end = abs(value - interval_end)
            
            if delta_to_start <= threshold_factor * abs(interval_start) if interval_start != 0 else delta_to_start < threshold_factor:
                proximity = "Close to lower bound"
            elif delta_to_end <= threshold_factor * abs(interval_end) if interval_end != 0 else delta_to_end < threshold_factor:
                proximity = "Close to upper bound"
            elif value < interval_start:
                proximity = "Below lower bound"
            elif value > interval_end:
                proximity = "Above upper bound"
            else:
                proximity = "Within interval"
        
        in_range_status = "Within Clinical Reference Range" if within_interval else "Outside of Clinical Reference Range"
        results.append((in_range_status, proximity))
    
    return results


# =============================================================================
# 快速生成函数
# =============================================================================

def generate_domain_expert_prompt(
    query: str,
    user_profile: str = "",
    user_health_data: str = "",
    population_context: str = "",
) -> str:
    """Generate a complete prompt for the Domain Expert Agent.
    
    Args:
        query: The user's health question.
        user_profile: User demographic/profile information.
        user_health_data: User's health metrics and data.
        population_context: Population statistics for comparison.
    
    Returns:
        Complete prompt string.
    """
    return render_prompt(
        DOMAIN_EXPERT_QUERY_PROMPT,
        system_prompt=DOMAIN_EXPERT_SYSTEM_PROMPT,
        query=query,
        user_profile=user_profile,
        user_health_data=user_health_data,
        population_context=population_context,
    )

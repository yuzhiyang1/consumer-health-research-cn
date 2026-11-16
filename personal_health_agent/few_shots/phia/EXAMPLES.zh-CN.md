# PHIA 单 Agent 少样本示例中文导读

每节给出原问题的中文翻译和原回复的摘要。完整的 ReAct 步骤、工具调用及输出可打开标题中的英文 Notebook 查看。

这些问题、数值和回复用于演示 Agent 行为。原例中的医学判断、参考范围和计算可能有局限，不能作为诊断或治疗建议。

## [10k_adjust_run_goal](10k_adjust_run_goal.ipynb)

**问题**　我想在 45 分钟内跑完 10 公里，怎样调整训练？

**示例摘要**　示例指出跑步速度的标准差为 1.08 英里/小时，并把建议按速度稳定性、长跑和恢复分组。原文举例使用 6–8 公里节奏跑、长跑后半程加速及每周休息时间。

## [10k_run_goal](10k_run_goal.ipynb)

**问题**　我想在 45 分钟内跑完 10 公里。上个月开始的新训练有效吗？

**示例摘要**　示例对比上月与之前的平均速度（3.08 对 3.67 英里/小时），同时提及周跑量约 30 公里、目标配速约 4 分 30 秒/公里。原例的速度换算与训练改善结论前后有冲突，复核时应以原始数据和计算步骤为准。

## [HRV_bedtime](HRV_bedtime.ipynb)

**问题**　上个月我的入睡时间和 HRV 有什么关系？

**示例摘要**　示例算得相关系数约为 −0.085，表示线性关联很弱。应把这理解为记录中的相关性，不能据此断定晚睡使 HRV 下降。

## [REM_sleep_minutes_and_calories_burned_during_workouts](REM_sleep_minutes_and_calories_burned_during_workouts.ipynb)

**问题**　我的快速眼动（REM）睡眠时长与运动消耗的热量有什么关系？

**示例摘要**　示例得到相关系数 0.65，说明在该样本里两项指标倾向于一起升高。相关性本身无法说明哪一项导致了另一项变化。

## [REM_sleep_percentage_yesterday](REM_sleep_percentage_yesterday.ipynb)

**问题**　我昨天的 REM 睡眠占比是多少？

**示例摘要**　示例从昨晚的记录算出 19.06%，随后解释该阶段在睡眠记录中的含义，并讨论保持规律作息。

## [aerobic_workout_duration_by_sleep_condition](aerobic_workout_duration_by_sleep_condition.ipynb)

**问题**　睡眠超过 59 分钟的日子，我的有氧运动时长是多少？

**示例摘要**　示例先按睡眠时长筛选日期，再给出有氧运动时长 24.733 分钟。59 分钟是原问题给出的筛选条件。

## [average_HRV_conditioned_by_biking](average_HRV_conditioned_by_biking.ipynb)

**问题**　骑行超过 12 公里的日子，我的平均 HRV 是多少？

**示例摘要**　示例筛选骑行记录并对齐相应日期的睡眠或健康摘要，得到平均 HRV 为 14.355 ms。该值只描述符合条件的日期。

## [average_HR_and_calorie_burn_activities](average_HR_and_calorie_burn_activities.ipynb)

**问题**　不同运动的平均心率和平均热量消耗分别是多少？

**示例摘要**　按原例的「心率 bpm / 热量消耗」顺序，骑行 119.857 / 125.857，徒步 111.250 / 165.500，跑步 104.000 / 79.375，力量训练 114.500 / 110.167，有氧训练 133.667 / 39.000。

## [avg_rhr_in_two_weeks](avg_rhr_in_two_weeks.ipynb)

**问题**　过去两周我的平均静息心率是多少？

**示例摘要**　示例在过去两周的时间窗内求得均值 65.21 bpm，然后给出一般成人参考区间。

## [awake_percent_standard_deviation](awake_percent_standard_deviation.ipynb)

**问题**　我的清醒时间占比的标准差是多少？

**示例摘要**　示例得到约 3.1 个百分点，用这个统计量描述清醒占比的波动程度。

## [awake_percent_standard_deviation_7_days](awake_percent_standard_deviation_7_days.ipynb)

**问题**　最近 7 天的清醒时间占比标准差是多少？

**示例摘要**　示例先截取最近七天的记录，再得到约 3.133 个百分点，说明这一周的清醒占比波动较小。

## [bedtime_consistency](bedtime_consistency.ipynb)

**问题**　我的就寝时间规律吗？

**示例摘要**　示例算得最近 30 天就寝时间的标准差约为 54 分钟，并用它说明每天入睡时间的变化。

## [best_activities_to_manage_stress_score](best_activities_to_manage_stress_score.ipynb)

**问题**　哪些运动与我的压力管理分数变化关系最大？

**示例摘要**　示例发现户外骑行与最大降幅相关，平均下降 5.8 分。原回答把降分解释成压力改善，但分数方向需结合指标定义判断，也不能从这组记录推断因果关系。

## [best_workout_time](best_workout_time.ipynb)

**问题**　考虑吃饭时间，一天中什么时候锻炼比较合适？

**示例摘要**　示例根据日程、个人感受和训练前后的饮食安排组织答案，对比早餐前、午餐前和稍晚时段等选择。

## [calorie_intake](calorie_intake.ipynb)

**问题**　我每天应摄入多少热量？

**示例摘要**　示例使用年龄 63 岁、男性、身高 166 cm、体重 90 kg，并假设轻体力活动，估算维持体重的日总能量消耗约为 2383 千卡。活动水平是估算假设。

## [cardiovascular_health](cardiovascular_health.ipynb)

**问题**　我能做些什么来改善心血管健康？

**示例摘要**　示例以平均静息心率 71.8 bpm 为切入点，讨论规律运动、饮食、体重管理和戒烟等方向。

## [consistent_weekly_fitness](consistent_weekly_fitness.ipynb)

**问题**　怎样让每周运动更均衡，避免某天运动很多、其他天几乎不动？

**示例摘要**　示例用每日步数标准差约 1389.77 步和活跃区间分钟数标准差约 5.51 分钟说明波动，然后建议分散运动量并设定每日目标。

## [correct_form](correct_form.ipynb)

**问题**　做力量训练时，我的动作标准吗？

**示例摘要**　示例明确说明现有数据无法观察动作姿态，建议由训练教练或有经验的人当面观察，或参考可靠的示范视频。

## [days_slept_better](days_slept_better.ipynb)

**问题**　列出我睡得较好的日子。

**示例摘要**　示例将睡眠总时长在 7–9 小时、深睡和 REM 睡眠超过个人平均值定义为「睡得较好」，筛出 2024 年 1 月 3、4、23 和 31 日。

## [exercise_effects_on_sleep](exercise_effects_on_sleep.ipynb)

**问题**　我运动较多的日子，睡得更好吗？

**示例摘要**　示例计算活跃分钟数、消耗热量与深睡分钟数的相关系数，分别为 0.52 和 0.48，并提示相关性不能证明因果关系。

## [fatigue](fatigue.ipynb)

**问题**　哪些营养补充剂可能有助于缓解疲劳？

**示例摘要**　示例列出多种补充剂并提及使用需谨慎。原文是一般资料问答示例，未做缺乏筛查或给出个体适用判断。

## [greeting](greeting.ipynb)

**问题**　你好。

**示例摘要**　示例介绍 PHIA 可以分析 Fitbit 里的活动、睡眠和心率，并邀请用户提出与个人数据有关的问题。

## [heart_rate_health_effects](heart_rate_health_effects.ipynb)

**问题**　我的心率如何影响健康？

**示例摘要**　示例从平均静息心率约 72 bpm 入手，解释静息心率与日常活动、心肺体能和其他因素的关系。

## [height_and_weight_from_profile](height_and_weight_from_profile.ipynb)

**问题**　我的身高和体重是多少？

**示例摘要**　示例直接读取用户档案，取出身高 181 cm、体重 74 kg。

## [highest_rhr_last_30_days](highest_rhr_last_30_days.ipynb)

**问题**　最近 30 天的最高静息心率是多少？

**示例摘要**　示例在指定窗口内求最大值，得到 63.53 bpm，并提及压力、睡眠和补水情况可能影响记录。

## [improve_sleep_quality](improve_sleep_quality.ipynb)

**问题**　有哪些改善睡眠质量的办法？

**示例摘要**　示例发现每晚平均睡约 6.3 小时、上床和起床时间不固定，围绕规律作息、睡前放松、卧室环境以及睡前饮食给出建议。

## [manage_stress](manage_stress.ipynb)

**问题**　我怎样管理压力？

**示例摘要**　示例按瑜伽、运动、呼吸放松和日常习惯等类别组织建议。

## [maximize_muscle_minimize_fat](maximize_muscle_minimize_fat.ipynb)

**问题**　怎样安排饮食，尽量增肌并减少体脂增长？

**示例摘要**　示例围绕蛋白质、复合碳水化合物和不饱和脂肪列出食物选择，并讨论蛋白质分配和整体饮食安排。

## [mindfulness](mindfulness.ipynb)

**问题**　如何把更多正念练习融入生活？

**示例摘要**　示例从每天五分钟专注呼吸开始，延伸到吃饭时关注感官、觉察走神并把注意力拉回当下。

## [mindfulness_effects](mindfulness_effects.ipynb)

**问题**　正念练习对我的静息心率或睡眠有帮助吗？

**示例摘要**　示例指出缺少与正念练习对应的时间记录，无法从数据直接评估个人影响，提出用日记将练习、睡眠和心率对齐观察。

## [muscle_recovery_and_injury_prevention](muscle_recovery_and_injury_prevention.ipynb)

**问题**　怎样促进肌肉恢复并减少受伤风险？

**示例摘要**　示例把做法分成热身、放松、补水、营养和休息，并讨论循序渐进增加负荷。

## [optimize_exercise_and_wellness](optimize_exercise_and_wellness.ipynb)

**问题**　从我的活动数据看，怎样调整训练和日常习惯？

**示例摘要**　示例汇报平均睡眠约 6.3 小时、静息心率约 71 bpm，并分别对照步数和活跃分钟数。接着讨论增加活动多样性和按疲劳情况安排恢复。

## [overall_health](overall_health.ipynb)

**问题**　我的整体健康情况怎么样？

**示例摘要**　示例按睡眠、心率、活动和压力管理汇报数据。平均睡眠 377.74 分钟，深睡占 22.85%，REM 睡眠占 21.51%，平均静息心率 71.77 bpm，活跃区间分钟数平均为 73.0。

## [rhr_during_olympics](rhr_during_olympics.ipynb)

**问题**　奥运会第一天我的静息心率是多少？

**示例摘要**　示例先把事件对应到 2024 年 7 月 26 日，再从该日记录读出静息心率 73.0 bpm。

## [running_compared_olympian](running_compared_olympian.ipynb)

**问题**　我的跑步速度和奥运选手相比如何？

**示例摘要**　示例将用户的平均速度 3.64 英里/小时与精英选手的速度作对照，并建议主要关注自己的训练进展。两者的距离和跑步条件未必相同。

## [sleep_data_patterns](sleep_data_patterns.ipynb)

**问题**　我的入睡、起床时间和睡眠时长有什么规律？

**示例摘要**　示例算得睡眠时长标准差 45.3 分钟，最常见的上床时间是晚上 10 点，起床时间是早上 7 点。明显偏离常用上床时间和起床时间的日子分别有 15 天和 18 天。

## [sleep_fastest_run](sleep_fastest_run.ipynb)

**问题**　我跑得最快的那次，前一晚睡得怎么样？

**示例摘要**　示例定位到 2024 年 6 月 15 日。前一晚缺少睡眠起止时刻，但记录了总睡眠约 9.7 小时、深睡约 1 小时、REM 睡眠超过 1.3 小时。

## [sleep_patterns](sleep_patterns.ipynb)

**问题**　我的睡眠有什么规律？

**示例摘要**　示例得到上床和起床时间标准差约为 49.39 和 37.61 分钟，平均睡眠约 378 分钟，平均睡眠得分 65，并按睡眠时长、阶段与作息讨论。

## [sleep_stages_resting_heart_rate](sleep_stages_resting_heart_rate.ipynb)

**问题**　各睡眠阶段与静息心率有相关性吗？

**示例摘要**　示例报告深睡、REM 睡眠、浅睡和清醒时长与静息心率的相关系数分别为 0.24、0.36、0.06 和 −0.01。

## [sleep_time_after_last_bike_ride](sleep_time_after_last_bike_ride.ipynb)

**问题**　我上一次骑行之后睡了多久？

**示例摘要**　示例找出最后一次骑行，匹配之后的睡眠记录，得到 436.8 分钟，约 7 小时 17 分钟。

## [strength](strength.ipynb)

**问题**　如何提高力量，同时尽量不增加体重？

**示例摘要**　示例讨论较轻负荷、多次重复、复合动作和自重训练，还提到训练后的休息和营养。

## [strengthen_core](strengthen_core.ipynb)

**问题**　哪些训练适合增强核心力量？

**示例摘要**　示例列出平板支撑、卷腹和俄罗斯转体等动作，简要解释每种动作的姿势。

## [weight_loss](weight_loss.ipynb)

**问题**　怎样减重？

**示例摘要**　示例围绕少加工食物、份量控制、补水、运动和睡眠组织答案，强调能持续执行的小调整。

## [workout_frequency](workout_frequency.ipynb)

**问题**　我应该多久锻炼一次？

**示例摘要**　示例先说每周活动量参考建议（中等强度 150 分钟或高强度 75 分钟、至少两天力量训练），再对比档案中平均每周约锻炼六天。

## [workouts_specific_goals](workouts_specific_goals.ipynb)

**问题**　为了实现不同健身目标，我可以做哪些运动？

**示例摘要**　示例把每天平均约 20.5 个活跃区间分钟与其采用的约 22 分钟日目标比较，再按减重、增肌等目标归类训练选项。

## [worst_HRV_last_month](worst_HRV_last_month.ipynb)

**问题**　上个月我的最低 HRV 是多少？

**示例摘要**　示例在上个月的记录中找出最低值 40 ms，并提醒单次低值不能判断状态，需结合个人波动与近期睡眠、压力等背景。

# `data/`——本地数据根。

`file_io.get_data_dir()`解决此目录(或`$WPM_DATA_DIR`).
版式：

```
data/
  dummy_data.csv          synthetic RHR user-months  (public schema, below;
  dummy_data_hrv.csv      synthetic HRV user-months   written by notebook 00)
  extra_data/             optional reference tables for the maps (not shipped)
  results/<run_tag>/      written by 02_modelling.ipynb
  results/paper/          optional: publication assets (.tex, png/svg/pdf) written
                          by 01 / 03 only when SAVE_PAPER_ASSETS = True
```

## 哑巴 CSVs

`dummy_data*.csv` ** 合成** ( `src/generate_dummy_data.py` 种子 0 ): a
19国家数百个用户的季节性/年龄/性别粗糙
结构，使模型竞争不会恶化。
测试加载和运行笔记本结束;它们中没有任何估计值
它们没有被承诺:用
`notebooks/00_generate_dummy_data.ipynb` 或 减 减

```
python src/generate_dummy_data.py --out_dir=data --n_users=300
```

## 公共投入计划a

每个用户和日历月各一行。信头在大小写中不敏感地匹配。
`cohort_io.load_cohort` 将各列改名为最后一列中的名称，
这些是模型公式、结果CSVs和
系数表。

| 列名               | 单位/值          | 代码中使用的名称        |
| :------------------- | :--------------------- | :--------------------------- |
| `id`                 | 用户标识符        | `user_id`                    |
| `month`              | 日历月         | `activity_period`            |
: : ( `YYYY-MM` ) : :
| `age`                | 年份                  | `age`                        |
| `sex`                | `Male` / `Female`      | `gender`                     |
: (又`M`/`F`) :
| `country`            | ISO-3166 阿尔法 - 2       | `country_2_alpha_cd`         |
| `latitude` ,          | 小数度        | `lat_e7` , `lng_e7` (1e- 7     |
`longitude`: 德格：
| `temperature`        | 每月平均空气量       | `avg_temp_mean`              |
:温度，°C::
| `temperature min`    | 月平均数    | `avg_temp_min`               |
:日最低，°C::
| `temperature max`    | 月平均数    | `avg_temp_max`               |
:日最大，°C::
| `day light`          | 平均日数、小时数 | `dl_monthly`                 |
| `resting heart rate` | bpm (RHR文件)         | `avg_resting_heart_rate_bpm` |
| ` 心率          | ms (HRV文件).          | `avg_hrv`                    |
: variability`         :                        :                              :
| `region` (可选).  | UN 区域标签        | `un_region` ( `unknown` 若为    |
: :缺席) :

`temperature min`/ `temperature max`因为阶段5/6,所以是需要的。
候选人将平均温度与最大/最低温度模型进行比较。

装货机(`cohort_io.load_cohort`)提出一个`ValueError`,列出：
当`sex`不是以上数值之一或当`latitude`/时，违法行。
`longitude` 缺少或非数字。只有空 `country` 字段计为
缺少(纳米比亚的代码`NA`保存).

## 参考表(`extra_data/`,可选)

只有地图数字需要它们;它们不在这里重新分配。下载它们
输入 `data/extra_data/` 依据其规定：

*   `ne_110_admin_0_countries.geojson`——天然地球 1:110m 行政 0——
    国家(https://www.naturalearthdata.com/).) 使用条件:公有领域。
*   `unsd-countries.csv`— UN 统计司"标准国家或地区"
    统计使用代码(M49)”
    ( https://unstats.un.org/unsd/methodology/m49/overview/), 栏` ISO -阿尔法2
    Code`, `ISO-alpha3 Code` and the region columns. Terms of use:
    https://www.un.org/en/about-us/terms-of-use.

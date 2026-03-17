# search.py 方法与实现说明

本文档说明 `src/search.py` 的核心思路、关键函数职责、搜索流程与复杂度特征，便于课程报告和后续维护。

## 1. 模块目标

`search.py` 的目标是：在给定约束下，搜索 2026-2040 年间从地球出发前往水星的可行发射窗口。模块支持两种模式：

- 单序列搜索：固定一条飞掠序列（如 Earth -> Venus -> Venus -> Mercury -> Mercury）。
- 多序列搜索：自动生成候选飞掠序列并逐条搜索，再进行全局汇总。

底层轨道评估由 `calculate.py` 中的 `evaluate_trajectory(...)` 完成，`search.py` 负责“生成候选 + 分层筛选 + 汇总排序”。

## 2. 数据与依赖

- 行星天体：`Earth, Venus, Mars, Mercury`，可选 `Moon`。
- 时间系统：`astropy.time.Time`。
- 组合生成：`itertools.product`。
- 约束结构：`MissionConstraints`。

关键输入约束通常包括：

- 发射能力上限（以 C3 或 v∞ 表示）
- 航天器机动预算（`spacecraft_budget_km_s`）
- 任务总时长上限（`max_duration_days`）

## 3. 核心辅助函数

### 3.1 `_body_name(body)` 与 `_sequence_name(sequence_bodies)`

- 将天体对象转换为可读字符串。
- 将序列格式化成 `Earth -> Venus -> Mercury`，用于日志与结果标注。

### 3.2 `_build_leg_vectors(n_legs, coarse=True)`

作用：构造每一段转移时间（TOF, days）的候选向量。

实现要点：

- 按“第几段”定义不同范围（前几段范围更大，后几段更紧）。
- 粗搜步长更大（40 天），精搜步长更小（20 天）。
- 使用笛卡尔积生成全组合。
- 总时长过滤：
  - 粗搜保留 `700 <= sum(leg_days) <= 3500`
  - 精搜保留 `700 <= sum(leg_days) <= 3650`

意义：先大范围覆盖，再局部细化，平衡搜索广度和计算量。

### 3.3 `_sort_results(results)`

统一排序规则（越靠前越优）：

1. 先看是否可行（`feasible=True` 优先）
2. 航天器 ΔV 更小优先
3. 发射 v∞ 更小优先
4. 任务时长更短优先

### 3.4 `_score_near_candidate(r, constraints)`

作用：为“不完全可行但接近可行”的样本打分，作为精搜种子来源。

评分形式（越小越好）：

- 航天器 ΔV 超预算惩罚（权重 10）
- 发射 v∞ 超限惩罚（权重 4）
- 任务时长超限惩罚（按年计）
- 再加一个极小的 ΔV 本体项（0.01 倍）用于打破并列

这使算法在无严格可行解时仍能沿“最有希望”的方向继续搜索。

### 3.5 `_annotate_result(result, sequence_name)`

给每个结果追加 `sequence_name` 字段，便于多序列汇总追踪来源。

## 4. 候选序列生成（多序列模式）

函数：`_generate_candidate_sequences(max_sequences=24, include_moon=False)`

### 4.1 生成策略

- 固定起点 Earth，终点 Mercury。
- 中间节点池优先 `Venus > Earth > Mars`，可选 Moon。
- 两类来源：
  - 手工高优模板（例如 E-V-M、E-V-V-M、E-V-E-V-M 等）
  - 组合枚举（中间长度 1~4）

### 4.2 启发式过滤

- 至少包含 1 次 Venus
- Mars 最多 1 次
- 额外 Earth 最多 2 次
- Moon 最多 1 次
- 避免不合理相邻重复（除允许的特定情况）

### 4.3 去重与打分

- 用序列名称元组去重。
- 打分函数偏好“金星更多、长度适中”，并惩罚 Earth/Mars/Moon 过多。
- 最终取前 `max_sequences` 条。

## 5. 单序列搜索流程

函数：`find_launch_windows(...)`

### 5.1 粗搜索阶段

- 发射日期：`start_iso` 到 `end_iso`，步长 20 天。
- 每个发射日遍历粗粒度 `leg_vectors_coarse`。
- 对每组参数调用 `evaluate_trajectory(...)`。
- 记录：
  - `coarse_hits`：严格可行解
  - `near_pool`：近可行候选（用于后续种子）
- 输出粗搜统计：`coarse_scanned`。

### 5.2 精搜索种子选择

- 若粗搜已有可行解：选前 12 个可行解作为种子。
- 若粗搜无可行解：选前 12 个近可行候选作为种子。

### 5.3 精搜索阶段

精搜组合由两部分拼接：

- 全局细网格：`_build_leg_vectors(..., coarse=False)`
- 局部扰动网格：围绕种子 `leg_days` 做 `[-15, 0, +15]` 扰动组合

并对每个种子发射时刻在 ±30 天内按 2 天步长扫描。

循环结构可理解为：

- 对每个 seed epoch
- 对局部发射时刻
- 对每个精搜 leg 向量
- 调用 `evaluate_trajectory(...)`

记录：

- `fine_hits`：精搜可行解
- `fine_scanned`：精搜评估次数

### 5.4 结果组织

返回字典包含：

- `coarse_scanned`, `fine_scanned`
- `feasible_windows` 与 `top_feasible`
- `near_feasible`（主要在精搜无可行时提供）
- `sequence_name`

## 6. 多序列总控流程

函数：`find_launch_windows_multi_sequence(...)`

流程：

1. 先生成候选序列集合。
2. 对每条序列调用一次 `find_launch_windows(...)`。
3. 累加每条序列的 `coarse_scanned` 与 `fine_scanned`。
4. 合并所有可行/近可行解，并按全局规则排序。
5. 生成 `per_sequence` 汇总（每条序列的可行数、近可行数、最佳样本）。

返回字典包含：

- `coarse_scanned`（全局粗搜总数）
- `fine_scanned`（全局精搜总数）
- `top_feasible`（全局前 50）
- `near_feasible`（全局前 80）
- `per_sequence`（序列级摘要）

## 7. 复杂度与性能特征

性能瓶颈主要来自 `evaluate_trajectory(...)` 调用次数，数量级近似为：

- 单序列：`N_launch * N_leg_coarse + N_seed * N_launch_local * N_leg_fine`
- 多序列：上述结果再乘以候选序列数

代码层面的性能控制手段：

- 分阶段搜索（粗 -> 精）
- 近可行种子机制（避免“无可行即停止”）
- 局部扰动与去重（减少无效重复组合）
- 进度打印（便于长任务可观测）

## 8. 方法优点与边界

优点：

- 工程实现清晰，便于调参与复现实验。
- 在严格约束下仍可通过近可行引导继续探索。
- 多序列模式可自动比较不同飞掠策略。

边界：

- 仍是课程项目级 patched-conic + Lambert 拼接，不含高保真传播、B-plane 约束和深空机动连续优化。
- 结果质量依赖候选序列启发式与网格离散精度，不保证全局最优。

## 9. 使用建议

- 想快速试跑：减小 `max_sequences`、增大粗搜步长或收缩 leg 范围。
- 想提高命中率：适度增大种子数量、细化局部步长、扩展可选序列模板。
- 想做论文级对比：固定随机/枚举策略并记录 `coarse_scanned`、`fine_scanned`、最优 ΔV、最短时长等指标。
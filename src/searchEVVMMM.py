"""
searchEVVMMM.py
===============
E/V/M 字母序列解析器与针对 EVVMMM（地球–金星–金星–水星–水星–水星）
重力助推飞掠序列调优的发射窗口搜索模块。

背景
----
直接使用 search.py 中的通用腿时间范围时，5 腿（EVVMMM）序列的总飞行时间
上限约为 920 天，远低于该序列所需的 1500–2500 天（参见 MESSENGER 实际
约 1600 天、参 plan1.md）。本模块使用基于真实任务时间线的专项腿范围。

主要导出
--------
LETTER_BODY_MAP : dict
    单字母 (E/V/M) → poliastro 天体对象的映射。

DEFAULT_SEQUENCE : str
    默认序列代码 "EVVMMM"。

parse_letter_sequence(code) -> list
    将字母串（如 "EVVMMM"）解析为 poliastro 天体列表。
    要求首字母为 'E'，末字母为 'M'。

find_windows_evvmmm(sequence_code, start_iso, end_iso, constraints, ...)
    -> dict
    对指定字母序列执行两阶段（粗搜索 + 精搜索）发射窗口搜索，
    返回与 _build_report_multi 兼容的结果字典。
"""

from itertools import product as iter_product
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.time import Time
from poliastro.bodies import Earth, Mercury, Venus

from calculate import MissionConstraints, evaluate_trajectory

# ── 字母代码 → poliastro 天体 ─────────────────────────────────────────────────

LETTER_BODY_MAP: Dict[str, object] = {
    "E": Earth,
    "V": Venus,
    "M": Mercury,
}

_BODY_NAME: Dict[object, str] = {
    Earth: "Earth",
    Venus: "Venus",
    Mercury: "Mercury",
}

DEFAULT_SEQUENCE = "EVVMMM"

# ── 每对天体的腿时间范围表（粗 / 精，单位：天）────────────────────────────────
# 同星飞掠采用共振锁定：
# - Venus -> Venus: 约 1 个金星年（225 天）
# - Mercury -> Mercury: 1 或 2 个水星年（88/176 天）

_PAIR_COARSE: Dict[Tuple, np.ndarray] = {
    (Earth,   Venus  ): np.arange(120, 241, 30),
    (Venus,   Venus  ): np.array([225]),
    (Venus,   Mercury): np.arange(80,  201, 40),
    (Mercury, Mercury): np.array([88, 176]),
    (Earth,   Mercury): np.arange(80,  241, 80),
}

_PAIR_FINE: Dict[Tuple, np.ndarray] = {
    (Earth,   Venus  ): np.arange(100, 261, 15),
    (Venus,   Venus  ): np.arange(222, 229, 2),
    (Venus,   Mercury): np.arange(70,  221, 15),
    (Mercury, Mercury): np.array([88, 176]),
    (Earth,   Mercury): np.arange(80,  241, 30),
}

_GENERIC_COARSE = np.arange(100, 401, 75)
_GENERIC_FINE   = np.arange(80,  451, 35)


# ── 序列解析 ──────────────────────────────────────────────────────────────────

def parse_letter_sequence(code: str) -> List:
    """
    将字母序列字符串解析为 poliastro 天体列表。

    Parameters
    ----------
    code : str
        由 'E'、'V'、'M' 组成的字符串（大小写不敏感），如 "EVVMMM"。
        首字母必须为 'E'（地球出发），末字母必须为 'M'（水星目标）。

    Returns
    -------
    List
        poliastro 天体对象列表。

    Raises
    ------
    ValueError
        若序列含非法字母、长度不足，或不满足首 E 末 M 约束。
    """
    code = code.strip().upper()
    if len(code) < 2:
        raise ValueError(
            f"序列 '{code}' 太短：至少需要 2 个天体（例如 'EM'）。"
        )
    bodies: List = []
    for ch in code:
        if ch not in LETTER_BODY_MAP:
            raise ValueError(
                f"序列 '{code}' 中含未知字母 '{ch}'。"
                f"支持的字母：E（Earth）、V（Venus）、M（Mercury）。"
            )
        bodies.append(LETTER_BODY_MAP[ch])
    if bodies[0] is not Earth:
        raise ValueError(
            f"序列必须以 'E'（地球出发）开头，当前首字母：'{code[0]}'。"
        )
    if bodies[-1] is not Mercury:
        raise ValueError(
            f"序列必须以 'M'（水星）结尾，当前末字母：'{code[-1]}'。"
        )
    return bodies


def _sequence_label(bodies: List) -> str:
    return " -> ".join(_BODY_NAME.get(b, str(b)) for b in bodies)


# ── 腿时间范围生成 ────────────────────────────────────────────────────────────

def _get_leg_range(body_a, body_b, is_last_leg: bool, coarse: bool) -> np.ndarray:
    """返回一对天体之间的飞行时间候选数组（天）。"""
    lookup = _PAIR_COARSE if coarse else _PAIR_FINE
    key = (body_a, body_b)
    return lookup.get(key, _GENERIC_COARSE if coarse else _GENERIC_FINE)


def _build_leg_vectors(bodies: List, coarse: bool = True) -> List[List[float]]:
    """
    为给定天体序列构建腿飞行时间候选向量（笛卡尔积）。

    过滤总飞行时间不在 [600, 3650] 天范围内的向量。
    """
    n_legs = len(bodies) - 1
    if n_legs < 1:
        return []

    ranges: List[np.ndarray] = []
    for i in range(n_legs):
        ranges.append(_get_leg_range(bodies[i], bodies[i + 1], i == n_legs - 1, coarse))

    vectors: List[List[float]] = []
    for combo in iter_product(*ranges):
        total = sum(combo)
        if 600.0 <= total <= 3650.0:
            vectors.append(list(map(float, combo)))
    return vectors


# ── 辅助函数（与 search.py 一致）───────────────────────────────────────────────

def _sort_results(results: List[Dict]) -> List[Dict]:
    return sorted(
        results,
        key=lambda r: (
            0 if r.get("feasible", False) else 1,
            r.get("spacecraft_delta_v_km_s", 1e9),
            r.get("launch_excess_km_s", 1e9),
            r.get("duration_days", 1e9),
        ),
    )


def _score_near(r: Dict, c: MissionConstraints) -> float:
    sc_ex  = max(0.0, r.get("spacecraft_delta_v_km_s", 1e9) - c.spacecraft_budget_km_s)
    lv_ex  = max(0.0, r.get("launch_excess_km_s", 1e9) - np.sqrt(c.launch_c3_max_km2_s2))
    dur_ex = max(0.0, r.get("duration_days", 1e9) - c.max_duration_days) / 365.25
    return 10.0 * sc_ex + 4.0 * lv_ex + dur_ex + 0.01 * r.get("spacecraft_delta_v_km_s", 1e9)


def _rank_near_seed(vectors: List[List[float]], seed: List[float], max_keep: int) -> List[List[float]]:
    """将候选向量按与种子的距离排序，返回最近的 max_keep 个。"""
    if not vectors or max_keep <= 0:
        return []
    if not seed:
        return vectors[:max_keep]
    seed_sum = float(sum(seed))

    def dist(v: List[float]) -> float:
        n = min(len(v), len(seed))
        return sum(abs(v[i] - seed[i]) for i in range(n)) + 0.1 * abs(sum(v) - seed_sum)

    return sorted(vectors, key=dist)[:max_keep]


# ── 主搜索函数 ────────────────────────────────────────────────────────────────

def find_windows_evvmmm(
    sequence_code: str = DEFAULT_SEQUENCE,
    start_iso: str = "2026-01-01",
    end_iso: str = "2040-12-31",
    constraints: Optional[MissionConstraints] = None,
    *,
    coarse_launch_step_days: int = 20,
    max_coarse_top: int = 200,
    fine_eval_budget: int = 200000,
    fine_per_seed_budget: int = 50000,
    fine_early_stop: int = 10000,
    verbose: bool = True,
) -> Dict:
    """
    针对给定 E/V/M 字母序列执行两阶段发射窗口搜索。

    使用基于真实任务时间线（MESSENGER / BepiColombo）调优的腿飞行时间范围，
    覆盖 EVVMMM 族序列所需的 1200–2500 天飞行时段（通用范围仅达 ~920 天）。

    Parameters
    ----------
    sequence_code : str
        E/V/M 字母序列（大小写不敏感），例如 "EVVMMM"、"EVMM"。
        关键字 "single" 等价于默认的 "EVVMMM"。
    start_iso, end_iso : str
        搜索区间的 ISO 起止日期。
    constraints : MissionConstraints, optional
        任务约束；若为 None 则使用 plan1.md 中的标准约束：
        C3 ≤ 16.39 km²/s²（v∞ ≤ 4.05 km/s），ΔV_sc ≤ 2.25 km/s，T ≤ 3650 天。
    coarse_launch_step_days : int        粗搜索阶段发射日期采样步长（天）。
    max_coarse_top : int
        粗搜索保留最优候选数量，用于种子化精搜索。
    fine_eval_budget : int
        精搜索轨迹评估总次数上限。
    fine_per_seed_budget : int
        精搜索每个种子点的评估次数上限。
    fine_early_stop : int
        精搜索无改进的连续评估次数,达到后提前终止当前种子。
    verbose : bool
        是否打印搜索进度。

    Returns
    -------
    dict
        键包括：
        - ``top_feasible``  : 满足全部约束的最优窗口列表（最多 20 条）
        - ``near_feasible`` : 最接近可行的候选列表（无可行解时填充）
        - ``feasible_windows`` : 所有可行窗口
        - ``coarse_scanned`` / ``fine_scanned`` : 各阶段评估次数
        - ``sequence_name`` : 序列标签字符串
        - ``per_sequence``  : 单元素列表，兼容 _build_report_multi 格式
    """
    # ── 规范化输入 ────────────────────────────────────────────────────────────
    code_upper = sequence_code.strip().upper()
    if code_upper == "SINGLE":
        code_upper = DEFAULT_SEQUENCE

    bodies = parse_letter_sequence(code_upper)
    n_legs = len(bodies) - 1
    seq_name = _sequence_label(bodies)

    if constraints is None:
        constraints = MissionConstraints(
            launch_c3_max_km2_s2=4.05 ** 2,
            spacecraft_budget_km_s=2.25,
            max_duration_days=3650.0,
            mercury_orbit_altitude_km=400.0,
        )

    start_t = Time(start_iso)
    end_t   = Time(end_iso)
    n_days  = int((end_t - start_t).to_value("day")) + 1
    launch_dates = start_t + np.arange(0, n_days, coarse_launch_step_days)

    leg_vectors_coarse = _build_leg_vectors(bodies, coarse=True)

    if verbose:
        print(f"\n{'=' * 68}")
        print(f"[EVVMMM-OPT] Sequence : {seq_name}")
        print(f"[EVVMMM-OPT] Period   : {start_iso} → {end_iso}")
        print(
            f"[EVVMMM-OPT] Constraints:"
            f"  C3 ≤ {constraints.launch_c3_max_km2_s2:.1f} km²/s² |"
            f"  ΔV_sc ≤ {constraints.spacecraft_budget_km_s:.2f} km/s |"
            f"  T_max ≤ {constraints.max_duration_days:.0f} d"
        )
        print(
            f"[EVVMMM-OPT] Coarse: {len(launch_dates)} launch dates"
            f" × {len(leg_vectors_coarse)} leg combos"
            f" = {len(launch_dates) * len(leg_vectors_coarse):,} evals"
        )
        print(f"{'=' * 68}")

    # ── 阶段一：粗搜索 ────────────────────────────────────────────────────────
    coarse_feasible: List[Dict] = []
    near_pool: List[Tuple[float, Dict]] = []
    coarse_scanned = 0

    for idx, launch_t in enumerate(launch_dates):
        if verbose and idx % max(1, len(launch_dates) // 8) == 0:
            print(
                f"  [Coarse] {idx:4d}/{len(launch_dates)}"
                f" | feasible={len(coarse_feasible)}"
                f" | near_pool={len(near_pool)}"
            )
        for leg_days in leg_vectors_coarse:
            coarse_scanned += 1
            r = evaluate_trajectory(launch_t, bodies, leg_days, constraints)
            if not r.get("valid", False):
                continue
            r["sequence_name"] = seq_name
            if r.get("feasible", False):
                coarse_feasible.append(r)
            elif "launch_epoch_iso" in r:
                near_pool.append((_score_near(r, constraints), r))

    coarse_feasible = _sort_results(coarse_feasible)[:max_coarse_top]
    near_pool_sorted = sorted(near_pool, key=lambda x: x[0])
    best_near_coarse = [x[1] for x in near_pool_sorted[:max(80, max_coarse_top // 2)]]

    if verbose:
        print(
            f"  [Coarse] Done: {coarse_scanned:,} evals |"
            f" feasible={len(coarse_feasible)} |"
            f" near_seeds={len(best_near_coarse)}"
        )

    # ── 阶段二：精搜索 ────────────────────────────────────────────────────────
    seeds = coarse_feasible[:12] if coarse_feasible else best_near_coarse[:12]
    seed_epochs    = [Time(r["launch_epoch_iso"]) for r in seeds]
    seed_leg_days  = [r["leg_days"] for r in seeds]

    leg_vectors_fine = _build_leg_vectors(bodies, coarse=False)

    # 在各种子点周围生成局部微扰腿向量
    local_vectors: List[List[float]] = []
    for base_legs in seed_leg_days:
        deltas = [-12.0, 0.0, 12.0]
        choices = [[max(20.0, l + d) for d in deltas] for l in base_legs]
        for combo in iter_product(*choices):
            vv = list(map(float, combo))
            if 600.0 <= sum(vv) <= constraints.max_duration_days:
                local_vectors.append(vv)
    if local_vectors:
        local_vectors = [list(x) for x in {tuple(v) for v in local_vectors}]

    all_fine_vectors = [
        list(x) for x in {tuple(v) for v in leg_vectors_fine + local_vectors}
    ]

    if verbose:
        print(
            f"  [Fine]   Seeds={len(seed_epochs)}"
            f" | fine leg combos={len(all_fine_vectors)}"
            f" | budget: total={fine_eval_budget:,} / seed={fine_per_seed_budget:,}"
        )

    fine_feasible: List[Dict] = []
    fine_scanned = 0

    for s_idx, seed_t in enumerate(seed_epochs):
        if fine_scanned >= fine_eval_budget:
            if verbose:
                print(f"  [Fine]   Total budget reached ({fine_scanned:,}). Stop.")
            break

        focused = _rank_near_seed(
            all_fine_vectors,
            seed_leg_days[s_idx] if s_idx < len(seed_leg_days) else [],
            max_keep=2500,
        )
        local_launch = seed_t + np.arange(-30, 31, 2)

        seed_scanned = 0
        no_improve   = 0
        best_score   = 1e18

        if verbose:
            print(
                f"  [Fine]   Seed {s_idx + 1}/{len(seed_epochs)}"
                f" @ {seed_t.iso[:10]}"
                f" | focused combos={len(focused)}"
            )

        for lt in local_launch:
            if lt < start_t or lt > end_t:
                continue
            for leg_days in focused:
                if (
                    fine_scanned >= fine_eval_budget
                    or seed_scanned >= fine_per_seed_budget
                    or no_improve >= fine_early_stop
                ):
                    break

                fine_scanned += 1
                seed_scanned += 1

                r = evaluate_trajectory(lt, bodies, leg_days, constraints)
                if not r.get("valid", False):
                    continue

                score = _score_near(r, constraints)
                if score < best_score:
                    best_score = score
                    no_improve = 0
                else:
                    no_improve += 1

                if r.get("feasible", False):
                    r["sequence_name"] = seq_name
                    fine_feasible.append(r)

            if (
                fine_scanned >= fine_eval_budget
                or seed_scanned >= fine_per_seed_budget
                or no_improve >= fine_early_stop
            ):
                break

        if verbose:
            print(
                f"           Seed done: scanned={seed_scanned:,}"
                f" | best_score={best_score:.3f}"
                f" | feasible_so_far={len(fine_feasible)}"
            )

    fine_feasible = _sort_results(fine_feasible)
    near_result   = [x[1] for x in near_pool_sorted[:40]] if not fine_feasible else []

    if verbose:
        print(f"\n[EVVMMM-OPT] RESULT: coarse={coarse_scanned:,} | fine={fine_scanned:,}")
        print(f"             Feasible windows found: {len(fine_feasible)}")
        if fine_feasible:
            best = fine_feasible[0]
            print(
                f"             Best: launch={best['launch_epoch_iso'][:10]}"
                f"  arrival={best['arrival_epoch_iso'][:10]}"
                f"  sc_dv={best['spacecraft_delta_v_km_s']:.3f} km/s"
            )
        elif near_result:
            top = near_result[0]
            print(
                f"             Near-feasible: launch={top.get('launch_epoch_iso', '?')[:10]}"
                f"  sc_dv={top.get('spacecraft_delta_v_km_s', float('nan')):.3f} km/s"
            )

    per_sequence_entry = {
        "sequence_name":  seq_name,
        "feasible_count": len(fine_feasible),
        "near_count":     len(near_result),
        "best": fine_feasible[0] if fine_feasible else (near_result[0] if near_result else None),
    }

    return {
        "sequence_name":   seq_name,
        "coarse_scanned":  coarse_scanned,
        "fine_scanned":    fine_scanned,
        "feasible_windows": fine_feasible,
        "top_feasible":    fine_feasible[:20],
        "near_feasible":   near_result,
        "per_sequence":    [per_sequence_entry],
    }

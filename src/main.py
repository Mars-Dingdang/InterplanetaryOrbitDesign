import argparse
import os
import sys
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from calculate import MissionConstraints, parse_sequence
from search import find_launch_windows, find_launch_windows_multi_sequence
from searchEVVMMM import DEFAULT_SEQUENCE, find_windows_evvmmm, parse_letter_sequence


def _fmt_row(r):
	return (
		f"| {r['launch_epoch_iso'][:10]} | {r['arrival_epoch_iso'][:10]} | "
		f"{r['duration_days']:.0f} | {r['launch_excess_km_s']:.2f} | "
		f"{r['spacecraft_delta_v_km_s']:.2f} | {r['capture_delta_v_km_s']:.2f} | "
		f"{r['total_delta_v_km_s']:.2f} |"
	)


def _build_report_multi(constraints, result):
	feasible = [r for r in result["top_feasible"] if "launch_epoch_iso" in r]
	near = [r for r in result["near_feasible"] if "launch_epoch_iso" in r]
	per_sequence = result.get("per_sequence", [])

	lines = []
	lines.append("# 前往水星发射窗口分析报告（2026-2040）")
	lines.append("")
	lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append("")
	lines.append("## 任务与方法")
	if len(per_sequence) == 1:
		lines.append(f"- 飞掠序列：{per_sequence[0]['sequence_name']}")
		lines.append("- 搜索算法：EVVMMM 调优两阶段网格搜索（粗搜 + 精搜）")
	else:
		lines.append(f"- 飞掠序列：多序列启发式搜索（共 {len(per_sequence)} 条候选序列）")
		lines.append("- 搜索算法：多序列遍历 + 分阶段网格搜索（粗搜 + 精搜）")
	lines.append("- 动力学模型：真实星历 + Lambert 多腿拼接（patched-conic 近似）")
	lines.append("")
	lines.append("## 约束条件")
	lines.append(f"- 发射能力：$v_\\infty \\leq {np.sqrt(constraints.launch_c3_max_km2_s2):.2f}$ km/s（即 C3 <= {constraints.launch_c3_max_km2_s2:.2f} km²/s²）")
	lines.append(f"- 航天器机动预算：$\\Delta v_{{sc}} \\leq {constraints.spacecraft_budget_km_s:.2f}$ km/s")
	lines.append(f"- 最大任务时长：{constraints.max_duration_days:.0f} 天（约 {constraints.max_duration_days/365.25:.1f} 年）")
	lines.append(f"- 目标轨道：水星近圆轨道，高度 {constraints.mercury_orbit_altitude_km:.0f} km")
	lines.append("")
	lines.append("## 搜索规模")
	if len(per_sequence) > 1:
		lines.append(f"- 候选序列数：{len(per_sequence)}")
	lines.append(f"- 粗搜索轨迹数：{result['coarse_scanned']}")
	lines.append(f"- 精搜索轨迹数：{result['fine_scanned']}")
	lines.append("")
	lines.append("### 各序列搜索结果")
	lines.append("|序列|可行窗口数|近可行数|最优解|")
	lines.append("|---|---:|---:|---|")
	for seq_info in per_sequence:
		best_str = "-"
		if seq_info["best"]:
			best_str = f"{seq_info['best']['launch_epoch_iso'][:10]} ({seq_info['best']['spacecraft_delta_v_km_s']:.2f})"
		lines.append(f"|{seq_info['sequence_name']}|{seq_info['feasible_count']}|{seq_info['near_count']}|{best_str}|")
	lines.append("")

	if feasible:
		lines.append("## 可行窗口（满足全部约束）")
		lines.append("| 发射日期 | 到达日期 | 时长(d) | 发射$v_\\infty$(km/s) | 航天器$\\Delta v$(km/s) | 捕获$\\Delta v$(km/s) | 总$\\Delta v$(km/s) |")
		lines.append("|---|---|---:|---:|---:|---:|---:|")
		for r in feasible[:10]:
			lines.append(_fmt_row(r))
		best = feasible[0]
		lines.append("")
		lines.append("### 最优解（按航天器$\\Delta v$优先）")
		lines.append(f"- 发射：{best['launch_epoch_iso'][:10]}")
		lines.append(f"- 到达：{best['arrival_epoch_iso'][:10]}")
		lines.append(f"- 任务时长：{best['duration_days']:.0f} 天")
		lines.append(f"- 发射$v_\\infty$：{best['launch_excess_km_s']:.2f} km/s")
		lines.append(f"- 航天器$\\Delta v$（飞掠失配 + 捕获）：{best['spacecraft_delta_v_km_s']:.2f} km/s")
	else:
		lines.append("## 可行性结论")
		lines.append("- 在当前序列和约束下，未找到严格满足全部约束的窗口。")
		if near:
			lines.append("- 下表给出最接近可行的候选（用于权衡与放宽约束分析）。")
			lines.append("")
			lines.append("| 发射日期 | 到达日期 | 时长(d) | 发射$v_\\infty$(km/s) | 航天器$\\Delta v$(km/s) | 捕获$\\Delta v$(km/s) | 总$\\Delta v$(km/s) |")
			lines.append("|---|---|---:|---:|---:|---:|---:|")
			for r in near[:10]:
				lines.append(_fmt_row(r))

	lines.append("")
	lines.append("## 与历史任务比较（定性）")
	lines.append("- MESSENGER：约 6.5 年，地球/金星/水星多次飞掠，核心思想是逐步降低日心能量。")
	lines.append("- BepiColombo：序列更长且结合电推进，降低了单次脉冲机动压力，但任务周期更长。")
	lines.append("- 本分析结果与上述思路一致：若限制发射能力和星上$\\Delta v$较紧，则往往需要更多飞掠或更长任务时间。")
	lines.append("")
	lines.append("## 说明")
	lines.append("- 本模型为课程项目级近似：Lambert 分段 + 飞掠速率匹配惩罚，不含完整 B-plane 约束与深空机动优化。")
	lines.append("- 若要工程级设计，建议引入全局优化（GA/PSO）、深空机动变量与高保真动力学传播。")

	return "\n".join(lines)


def _build_report(sequence_text, constraints, result):
	"""Deprecated: kept for backwards compatibility. Use _build_report_multi instead."""
	return _build_report_multi(constraints, result)


def _plot_windows(feasible, near):
	plt.figure(figsize=(10, 6))
	if feasible:
		x = [r["duration_days"] for r in feasible]
		y = [r["spacecraft_delta_v_km_s"] for r in feasible]
		plt.scatter(x, y, s=18, alpha=0.8, label="Feasible")
	if near:
		x2 = [r["duration_days"] for r in near]
		y2 = [r["spacecraft_delta_v_km_s"] for r in near]
		plt.scatter(x2, y2, s=18, alpha=0.8, label="Near-feasible")

	plt.xlabel("Mission Duration (days)")
	plt.ylabel("Spacecraft Delta-V (km/s)")
	plt.title("Mercury Mission Window Candidates")
	plt.grid(alpha=0.25)
	plt.legend()

	os.makedirs("img", exist_ok=True)
	out_png = os.path.join("img", "final_project_window_scatter.png")
	plt.savefig(out_png, dpi=160)
	plt.close()
	return out_png


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="main.py",
		description="水星发射窗口分析（2026–2040）",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=textwrap.dedent("""\
			示例:
			  python main.py                       默认：按 EVVMMM 序列搜索
			  python main.py -s EVVMMM             显式指定 EVVMMM 单序列搜索
			  python main.py -s EVMM               搜索 E→V→M→M 序列
			  python main.py -s single             单序列模式（等价于 -s EVVMMM）
			  python main.py -s multi              多序列启发式搜索（扩展候选池）
		"""),
	)
	p.add_argument(
		"-s", "--sequence",
		default=DEFAULT_SEQUENCE,
		metavar="SEQ",
		help=(
			"搜索模式 / 飞掠序列 (默认: %(default)s):\n"
			"  <E/V/M 字母串>  单序列搜索，例如 EVVMMM / EVMM\n"
			"  single          单序列模式（等价于 -s EVVMMM）\n"
			"  multi           多序列启发式搜索（生成候选序列池）"
		),
	)
	return p


def main():
	args = build_arg_parser().parse_args()
	seq_arg = args.sequence.strip()
	seq_upper = seq_arg.upper()

	constraints = MissionConstraints(
		launch_c3_max_km2_s2=4.05 ** 2,
		spacecraft_budget_km_s=2.25,
		max_duration_days=3650,
		mercury_orbit_altitude_km=400,
	)

	if seq_upper == "MULTI":
		print("\n=== Mercury Mission Window Search (Multi-Sequence Mode) ===")
		print("Starting multi-sequence search...\n")
		result = find_launch_windows_multi_sequence(
			start_iso="2026-01-01",
			end_iso="2040-12-31",
			constraints=constraints,
			max_sequences=24,
			include_moon=False,
		)
	else:
		code = DEFAULT_SEQUENCE if seq_upper == "SINGLE" else seq_upper
		try:
			# Validate sequence early to give a clear error message
			parse_letter_sequence(code)
		except ValueError as exc:
			print(f"错误：{exc}", file=sys.stderr)
			print("有效字母：E（Earth）、V（Venus）、M（Mercury）。示例：-s EVVMMM", file=sys.stderr)
			sys.exit(1)
		print(f"\n=== Mercury Mission Window Search (Single-Sequence: {code}) ===")
		result = find_windows_evvmmm(
			sequence_code=code,
			start_iso="2026-01-01",
			end_iso="2040-12-31",
			constraints=constraints,
		)

	report = _build_report_multi(constraints, result)

	report_path = "final_report.md"
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report)
	print(f"\n✓ Report saved to: {report_path}")

	img_path = _plot_windows(result["top_feasible"], result["near_feasible"])
	print(f"✓ Figure saved to: {img_path}")

	print("\n=== SUMMARY ===")
	print(f"Feasible windows: {len(result['top_feasible'])}")
	print(f"Near-feasible:    {len(result['near_feasible'])}")
	print(f"Total evals:      {result.get('coarse_scanned', 0) + result.get('fine_scanned', 0):,}")
	if result['top_feasible']:
		best = result['top_feasible'][0]
		print("\nBest solution:")
		print(f"  Launch:   {best['launch_epoch_iso'][:10]}")
		print(f"  Arrival:  {best['arrival_epoch_iso'][:10]}")
		print(f"  Duration: {best['duration_days']:.0f} days")
		print(f"  Launch v∞: {best['launch_excess_km_s']:.2f} km/s")
		print(f"  S/C ΔV:   {best['spacecraft_delta_v_km_s']:.2f} km/s")
	print()


if __name__ == "__main__":
	main()

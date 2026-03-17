import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from calculate import MissionConstraints, parse_sequence
from search_v2 import find_launch_windows_v2


def _fmt_row(r):
	return (
		f"| {r['launch_epoch_iso'][:10]} | {r['arrival_epoch_iso'][:10]} | "
		f"{r['duration_days']:.0f} | {r['launch_excess_km_s']:.2f} | "
		f"{r['spacecraft_delta_v_km_s']:.2f} | {r['capture_delta_v_km_s']:.2f} | "
		f"{r['total_delta_v_km_s']:.2f} |"
	)


def _build_report(sequence_text, constraints, result):
	feasible = [r for r in result["top_feasible"] if "launch_epoch_iso" in r]
	near = [r for r in result["near_feasible"] if "launch_epoch_iso" in r]

	lines = []
	lines.append("# 前往水星发射窗口分析报告（2026-2040）")
	lines.append("")
	lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append("")
	lines.append("## 任务与方法")
	lines.append(f"- 飞掠序列：`{sequence_text}`")
	lines.append("- 动力学模型：真实星历 + Lambert 多腿拼接（patched-conic 近似）")
	lines.append("- 搜索算法：多阶段自适应网格搜索（粗搜 → 过渡焦点 → 密集局部搜索）")
	lines.append("")
	lines.append("## 约束条件")
	lines.append(f"- 发射能力：$v_\\infty \\leq {np.sqrt(constraints.launch_c3_max_km2_s2):.2f}$ km/s（即 C3 <= {constraints.launch_c3_max_km2_s2:.2f} km²/s²）")
	lines.append(f"- 航天器机动预算：$\\Delta v_{{sc}} \\leq {constraints.spacecraft_budget_km_s:.2f}$ km/s")
	lines.append(f"- 最大任务时长：{constraints.max_duration_days:.0f} 天（约 {constraints.max_duration_days/365.25:.1f} 年）")
	lines.append(f"- 目标轨道：水星近圆轨道，高度 {constraints.mercury_orbit_altitude_km:.0f} km")
	lines.append("")
	lines.append("## 搜索规模与耗时")
	lines.append(f"- 总搜索耗时：{result.get('total_time_seconds', 0):.1f} 秒")
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


def _plot_windows(feasible, near):
	plt.figure(figsize=(10, 6))
	if feasible:
		x = [r["duration_days"] for r in feasible]
		y = [r["spacecraft_delta_v_km_s"] for r in feasible]
		plt.scatter(x, y, s=40, alpha=0.8, label="Feasible", color="green", marker="*")
	if near:
		x2 = [r["duration_days"] for r in near]
		y2 = [r["spacecraft_delta_v_km_s"] for r in near]
		plt.scatter(x2, y2, s=18, alpha=0.6, label="Near-feasible", color="orange")

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


def main():
	print("\n" + "="*70)
	print("MERCURY MISSION WINDOW SEARCH - MAIN PIPELINE")
	print("="*70 + "\n")
	
	sequence_text = "Earth Venus Venus Mercury Mercury"
	sequence_bodies = parse_sequence(sequence_text)
	
	constraints = MissionConstraints(
		launch_c3_max_km2_s2=12.25,  # 3.5 km/s (FIXED)
		spacecraft_budget_km_s=1.5,  # (FIXED)
		max_duration_days=3650,      # 10 years (FIXED)
		mercury_orbit_altitude_km=400,
	)
	
	print(f"Constraints (PROJECT SPECIFIED):")
	print(f"  • Launch v_inf ≤ {np.sqrt(constraints.launch_c3_max_km2_s2):.2f} km/s")
	print(f"  • S/C ΔV budget ≤ {constraints.spacecraft_budget_km_s:.2f} km/s")
	print(f"  • Max duration: {constraints.max_duration_days/365.25:.1f} years")
	print(f"\nStarting multi-phase adaptive search...\n")
	
	result = find_launch_windows_v2(
		sequence_bodies=sequence_bodies,
		start_iso="2026-01-01",
		end_iso="2040-12-31",
		constraints=constraints,
	)

	report = _build_report(sequence_text, constraints, result)

	report_path = "final_report.md"
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report)
	print(f"\n✓ Report saved to: {report_path}")

	img_path = _plot_windows(result["top_feasible"], result["near_feasible"])
	print(f"✓ Figure saved to: {img_path}")

	print(f"\n{'='*70}")
	print(f"FINAL RESULTS")
	print(f"{'='*70}")
	print(f"Feasible windows found: {len(result['top_feasible'])}")
	print(f"Near-feasible candidates: {len(result['near_feasible'])}")
	print(f"Total search time: {result.get('total_time_seconds', 0):.1f} seconds\n")
	
	if result['top_feasible']:
		best = result['top_feasible'][0]
		print(f"BEST SOLUTION:")
		print(f"  Launch date: {best['launch_epoch_iso'][:10]}")
		print(f"  Arrival date: {best['arrival_epoch_iso'][:10]}")
		print(f"  Mission duration: {best['duration_days']:.0f} days ({best['duration_days']/365.25:.2f} years)")
		print(f"  Launch v_inf: {best['launch_excess_km_s']:.3f} km/s")
		print(f"  S/C ΔV: {best['spacecraft_delta_v_km_s']:.3f} km/s")
		print(f"  Total ΔV: {best['total_delta_v_km_s']:.3f} km/s")
	else:
		if result['near_feasible']:
			print(f"No strictly feasible windows found.")
			best_near = result['near_feasible'][0]
			gap = best_near.get('spacecraft_delta_v_km_s', 1e9) - constraints.spacecraft_budget_km_s
			print(f"Best near-feasible:")
			print(f"  Launch: {best_near['launch_epoch_iso'][:10]}")
			print(f"  ΔV gap: {gap:+.3f} km/s (needs {abs(gap):.3f} km/s reduction)")
	
	print()


if __name__ == "__main__":
	main()

"""
Multi-phase adaptive search for Mercury mission windows.
Phase 1 (COARSE): Fast broad sweep with sparse grids
Phase 2 (TRANSITION): Zoom into promising regions  
Phase 3 (FINE): Dense local search + local optimization
"""

import time
from itertools import product
from typing import Dict, List, Sequence

import numpy as np
from astropy.time import Time

from calculate import MissionConstraints, evaluate_trajectory


def _phase1_coarse_sweep(
	sequence_bodies, start_t: Time, end_t: Time, constraints: MissionConstraints
) -> List[Dict]:
	"""Phase 1: Fast broad grid sweep."""
	n_legs = len(sequence_bodies) - 1
	print("\n" + "="*70)
	print("PHASE 1: COARSE BROAD SWEEP")
	print("="*70)
	
	# Very sparse launch dates: every 45 days
	launch_candidates = start_t + np.arange(0, int((end_t - start_t).to_value("day")) + 1, 45)
	
	# Sparse leg combinations
	ranges = [
		np.arange(100, 321, 70),  # E->V
		np.arange(80, 241, 70),   # V->V
		np.arange(70, 211, 70),   # V->M
		np.arange(60, 181, 70),   # M->M
		np.arange(50, 151, 70),
		np.arange(50, 151, 70),
	]
	use_ranges = ranges[:n_legs]
	leg_vectors = [list(map(float, v)) for v in product(*use_ranges)]
	leg_vectors = [v for v in leg_vectors if 700 <= sum(v) <= 2800]
	
	print(f"Sweeping {len(launch_candidates)} launch dates × {len(leg_vectors)} leg combos")
	print(f"Total evals: ~{len(launch_candidates) * len(leg_vectors)}\n")
	
	candidates = []
	t0 = time.time()
	eval_count = 0
	
	for i, launch_t in enumerate(launch_candidates):
		if i % max(1, len(launch_candidates) // 10) == 0:
			elapsed = time.time() - t0
			eta = elapsed * len(launch_candidates) / (i + 1) if i > 0 else 0
			print(f"  [{i+1:3d}/{len(launch_candidates)}] Elapsed: {elapsed:6.1f}s  ETA: {eta:6.1f}s")
		
		for leg_days in leg_vectors:
			eval_count += 1
			result = evaluate_trajectory(launch_t, sequence_bodies, leg_days, constraints)
			
			if not result.get("valid", False):
				continue
			
			if result.get("feasible", False):
				candidates.append(result)
				print(f"    ✓✓✓ FEASIBLE: {result['launch_epoch_iso'][:10]}  ΔV_sc={result['spacecraft_delta_v_km_s']:.2f}")
			else:
				# Track near-feasible for phase 2 focus
				dv_gap = result.get("spacecraft_delta_v_km_s", 1e9) - constraints.spacecraft_budget_km_s
				if dv_gap < 0.6:  # Within 0.6 km/s of budget
					candidates.append(result)
					print(f"    ◆ Near-feas: {result['launch_epoch_iso'][:10]}  gap={dv_gap:+.2f} km/s")
	
	elapsed = time.time() - t0
	print(f"\nPhase 1 complete: {elapsed:.1f}s, {eval_count} evals, {len(candidates)} promising candidates\n")
	return candidates


def _phase2_transition_focus(
	sequence_bodies, candidates: List[Dict], start_t: Time, end_t: Time, constraints: MissionConstraints
) -> List[Dict]:
	"""Phase 2: Zoom into regions near promising coarse solutions."""
	if not candidates:
		print("\nPHASE 2: Skipped (no candidates from Phase 1)\n")
		return []
	
	print("="*70)
	print("PHASE 2: TRANSITION FOCUS")
	print("="*70)
	
	n_legs = len(sequence_bodies) - 1
	
	# Sample top candidates by proximity to feasibility and diversity
	candidates_sorted = sorted(
		candidates,
		key=lambda r: (r.get("spacecraft_delta_v_km_s", 1e9), r.get("duration_days", 1e9))
	)
	seed_indices = np.linspace(0, len(candidates_sorted) - 1, min(8, len(candidates_sorted)), dtype=int)
	seed_candidates = [candidates_sorted[i] for i in seed_indices]
	
	print(f"Focusing on {len(seed_candidates)} seed regions\n")
	
	# Moderate leg resolution
	ranges = [
		np.arange(100, 321, 35),
		np.arange(80, 241, 35),
		np.arange(70, 211, 35),
		np.arange(60, 181, 35),
		np.arange(50, 151, 35),
		np.arange(50, 151, 35),
	]
	use_ranges = ranges[:n_legs]
	leg_vectors_fine = [list(map(float, v)) for v in product(*use_ranges)]
	leg_vectors_fine = [v for v in leg_vectors_fine if 700 <= sum(v) <= 3400]
	
	refined = []
	t0 = time.time()
	total_evals = 0
	
	for seed_idx, seed_cand in enumerate(seed_candidates):
		seed_epoch = Time(seed_cand["launch_epoch_iso"])
		print(f"  Region {seed_idx+1}: {seed_epoch.iso[:10]}  (ΔV_sc={seed_cand['spacecraft_delta_v_km_s']:.2f})")
		
		# Local search: ±20 days, ±10% leg scaling
		local_launches = seed_epoch + np.arange(-20, 21, 4)
		
		for launch_t in local_launches:
			if launch_t < start_t or launch_t > end_t:
				continue
			
			for leg_days in leg_vectors_fine:
				total_evals += 1
				if total_evals % 100 == 0:
					elapsed = time.time() - t0
					print(f"    Phase 2 progress: {total_evals} evals ({len(refined)} found) {elapsed:.1f}s")
				
				result = evaluate_trajectory(launch_t, sequence_bodies, leg_days, constraints)
				if not result.get("valid", False):
					continue
				
				if result.get("feasible", False):
					refined.append(result)
					print(f"      ✓✓✓ FEASIBLE window found: {result['launch_epoch_iso'][:10]}")
				elif result.get("spacecraft_delta_v_km_s", 1e9) - constraints.spacecraft_budget_km_s < 0.3:
					refined.append(result)
	
	elapsed = time.time() - t0
	print(f"\nPhase 2 complete: {elapsed:.1f}s, {total_evals} evals, {len(refined)} new candidates\n")
	return refined


def _phase3_dense_local_search(
	sequence_bodies, top_candidates: List[Dict], start_t: Time, end_t: Time, constraints: MissionConstraints
) -> List[Dict]:
	"""Phase 3: Dense local search + micro-optimization around best candidates."""
	if not top_candidates:
		print("\nPHASE 3: Skipped (no candidates)\n")
		return []
	
	print("="*70)
	print("PHASE 3: DENSE LOCAL SEARCH")
	print("="*70)
	
	n_legs = len(sequence_bodies) - 1
	top_candidates = sorted(
		top_candidates,
		key=lambda r: (r.get("spacecraft_delta_v_km_s", 1e9))
	)[:5]
	
	print(f"Refining {len(top_candidates)} top candidates with dense local search\n")
	
	# Dense leg resolution
	ranges = [
		np.arange(100, 321, 15),
		np.arange(80, 241, 15),
		np.arange(70, 211, 15),
		np.arange(60, 181, 15),
		np.arange(50, 151, 15),
		np.arange(50, 151, 15),
	]
	use_ranges = ranges[:n_legs]
	leg_vectors_dense = [list(map(float, v)) for v in product(*use_ranges)]
	leg_vectors_dense = [v for v in leg_vectors_dense if 700 <= sum(v) <= 3600]
	
	final_windows = []
	t0 = time.time()
	total_evals = 0
	
	for cand_idx, cand in enumerate(top_candidates):
		print(f"  Candidate {cand_idx+1}: {cand['launch_epoch_iso'][:10]} (ΔV_sc={cand['spacecraft_delta_v_km_s']:.2f})\n")
		seed_epoch = Time(cand["launch_epoch_iso"])
		
		# Tight local search: ±12 days
		local_launches = seed_epoch + np.arange(-12, 13, 1)
		
		for launch_t in local_launches:
			if launch_t < start_t or launch_t > end_t:
				continue
			
			for leg_days in leg_vectors_dense:
				total_evals += 1
				if total_evals % 150 == 0:
					elapsed = time.time() - t0
					print(f"    Phase 3 progress: {total_evals} evals, {len(final_windows)} windows found, {elapsed:.1f}s")
				
				result = evaluate_trajectory(launch_t, sequence_bodies, leg_days, constraints)
				if not result.get("valid", False):
					continue
				
				if result.get("feasible", False):
					final_windows.append(result)
					print(f"      ✓✓✓ FEASIBLE: {result['launch_epoch_iso'][:10]}")
	
	elapsed = time.time() - t0
	print(f"\nPhase 3 complete: {elapsed:.1f}s, {total_evals} evals, {len(final_windows)} feasible windows\n")
	return final_windows


def find_launch_windows_v2(
	sequence_bodies: Sequence,
	start_iso: str = "2026-01-01",
	end_iso: str = "2040-12-31",
	constraints: MissionConstraints = MissionConstraints(),
) -> Dict:
	"""Multi-phase adaptive search with continuous progress feedback."""
	
	start_t = Time(start_iso)
	end_t = Time(end_iso)
	
	print(f"\n{'='*70}")
	print(f"MERCURY MISSION WINDOW SEARCH (Multi-Phase Adaptive)")
	print(f"{'='*70}")
	print(f"Period: {start_iso} to {end_iso}")
	print(f"Launch C3 max: {np.sqrt(constraints.launch_c3_max_km2_s2):.2f} km/s")
	print(f"S/C ΔV budget: {constraints.spacecraft_budget_km_s} km/s")
	print(f"Max duration: {constraints.max_duration_days/365.25:.1f} years")
	
	global_start_t = time.time()
	
	# PHASE 1
	phase1_results = _phase1_coarse_sweep(sequence_bodies, start_t, end_t, constraints)
	
	# PHASE 2
	phase2_results = _phase2_transition_focus(sequence_bodies, phase1_results, start_t, end_t, constraints)
	all_candidates = phase1_results + phase2_results
	
	# PHASE 3
	phase3_results = _phase3_dense_local_search(sequence_bodies, all_candidates, start_t, end_t, constraints)
	
	# Compile results
	feasible_windows = sorted(
		[r for r in phase3_results if r.get("feasible", False)],
		key=lambda r: r.get("spacecraft_delta_v_km_s", 1e9)
	)
	
	near_feasible = sorted(
		[r for r in all_candidates if r.get("valid", False) and not r.get("feasible", False)],
		key=lambda r: abs(r.get("spacecraft_delta_v_km_s", 1e9) - constraints.spacecraft_budget_km_s)
	)[:20]
	
	total_time = time.time() - global_start_t
	print(f"\n{'='*70}")
	print(f"SEARCH COMPLETE in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
	print(f"Feasible windows: {len(feasible_windows)}")
	print(f"Near-feasible: {len(near_feasible)}")
	print(f"{'='*70}\n")
	
	return {
		"coarse_scanned": 0,  # Compatibility
		"fine_scanned": 0,
		"feasible_windows": feasible_windows,
		"top_feasible": feasible_windows[:20],
		"near_feasible": near_feasible,
		"total_time_seconds": total_time,
	}

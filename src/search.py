from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
from astropy.time import Time
from poliastro.bodies import Earth, Mars, Mercury, Venus

from calculate import MissionConstraints, evaluate_trajectory

try:
    from poliastro.bodies import Moon
    HAS_MOON = True
except Exception:
    Moon = None
    HAS_MOON = False


def _body_name(body) -> str:
    name = getattr(body, "name", str(body))
    return str(name).title()


def _sequence_name(sequence_bodies: Sequence) -> str:
    return " -> ".join(_body_name(b) for b in sequence_bodies)


def _build_leg_vectors(n_legs: int, coarse: bool = True) -> List[List[float]]:
    """
    Candidate per-leg time-of-flight vectors (days) for multi-flyby search.
    Designed for sequences like Earth->Venus->...->Mercury.
    """
    if n_legs < 1:
        return []

    if coarse:
        ranges = [
            np.arange(90, 271, 40),
            np.arange(70, 231, 40),
            np.arange(60, 201, 40),
            np.arange(50, 181, 40),
            np.arange(50, 161, 40),
            np.arange(50, 161, 40),
            np.arange(50, 161, 40),
        ]
    else:
        ranges = [
            np.arange(90, 271, 20),
            np.arange(70, 231, 20),
            np.arange(60, 201, 20),
            np.arange(50, 181, 20),
            np.arange(50, 161, 20),
            np.arange(50, 161, 20),
            np.arange(50, 161, 20),
        ]

    use_ranges = ranges[:n_legs]
    vectors = [list(map(float, v)) for v in product(*use_ranges)]

    if coarse:
        vectors = [v for v in vectors if 700 <= sum(v) <= 3500]
    else:
        vectors = [v for v in vectors if 700 <= sum(v) <= 3650]
    return vectors


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


def _score_near_candidate(r: Dict, constraints: MissionConstraints) -> float:
    sc_excess = max(0.0, r.get("spacecraft_delta_v_km_s", 1e9) - constraints.spacecraft_budget_km_s)
    launch_excess = max(0.0, r.get("launch_excess_km_s", 1e9) - np.sqrt(constraints.launch_c3_max_km2_s2))
    duration_excess = max(0.0, r.get("duration_days", 1e9) - constraints.max_duration_days) / 365.25
    return 10.0 * sc_excess + 4.0 * launch_excess + duration_excess + 0.01 * r.get("spacecraft_delta_v_km_s", 1e9)


def _annotate_result(result: Dict, sequence_name: str) -> Dict:
    result["sequence_name"] = sequence_name
    return result


def _rank_leg_vectors_near_seed(
    leg_vectors: List[List[float]],
    seed_leg_days: List[float],
    max_keep: int,
) -> List[List[float]]:
    """
    Keep leg vectors closest to a seed leg vector to focus fine search.
    """
    if not leg_vectors or max_keep <= 0:
        return []
    if not seed_leg_days:
        return leg_vectors[:max_keep]

    seed_sum = float(sum(seed_leg_days))

    def dist(v: List[float]) -> float:
        n = min(len(v), len(seed_leg_days))
        base = sum(abs(v[i] - seed_leg_days[i]) for i in range(n))
        return base + 0.1 * abs(sum(v) - seed_sum)

    ranked = sorted(leg_vectors, key=dist)
    return ranked[:max_keep]


def _generate_candidate_sequences(max_sequences: int = 24, include_moon: bool = False) -> List[List]:
    """
    Heuristic sequence generator:
    - Start at Earth
    - End at Mercury
    - Intermediate pool prioritized: Venus > Earth > Mars, optional Moon
    """
    pool = [Venus, Earth, Mars]
    if include_moon and HAS_MOON:
        pool.append(Moon)

    sequences: List[List] = []

    # Hand-crafted high-priority templates
    templates = [
        [Earth, Venus, Mercury],
        [Earth, Venus, Venus, Mercury],
        [Earth, Venus, Venus, Mercury, Mercury],
        [Earth, Venus, Earth, Venus, Mercury],
        [Earth, Venus, Earth, Venus, Mercury, Mercury],
        [Earth, Earth, Venus, Mercury],
        [Earth, Venus, Mars, Venus, Mercury],
        [Earth, Venus, Mars, Venus, Mercury, Mercury],
    ]

    for seq in templates:
        sequences.append(seq)

    # Combinational generation with heuristic pruning
    for mid_len in [1, 2, 3, 4]:
        for mids in product(pool, repeat=mid_len):
            seq = [Earth] + list(mids) + [Mercury]

            venus_count = sum(1 for b in seq if b is Venus)
            mars_count = sum(1 for b in seq if b is Mars)
            earth_count = sum(1 for b in seq if b is Earth) - 1
            moon_count = sum(1 for b in seq if HAS_MOON and b is Moon)

            # Heuristic filters
            if venus_count < 1:
                continue
            if mars_count > 1:
                continue
            if earth_count > 2:
                continue
            if moon_count > 1:
                continue

            # Avoid too many repeated neighbors except Mercury terminal repetitions (not used here)
            bad_repeat = False
            for i in range(1, len(seq)):
                if seq[i] is seq[i - 1] and seq[i] not in (Venus, Mercury):
                    bad_repeat = True
                    break
            if bad_repeat:
                continue

            sequences.append(seq)

    # Deduplicate and score
    unique = {}
    for seq in sequences:
        key = tuple(_body_name(b) for b in seq)
        unique[key] = seq
    sequences = list(unique.values())

    def seq_score(seq: List) -> float:
        venus_count = sum(1 for b in seq if b is Venus)
        mars_count = sum(1 for b in seq if b is Mars)
        earth_count = sum(1 for b in seq if b is Earth) - 1
        moon_count = sum(1 for b in seq if HAS_MOON and b is Moon)
        length_penalty = 0.25 * len(seq)
        # Lower is better. Favor Venus-rich and moderate length.
        return length_penalty + 0.8 * earth_count + 1.2 * mars_count + 1.0 * moon_count - 1.5 * venus_count

    sequences = sorted(sequences, key=seq_score)
    return sequences[:max_sequences]


def find_launch_windows(
    sequence_bodies: Sequence,
    start_iso: str = "2026-01-01",
    end_iso: str = "2040-12-31",
    constraints: MissionConstraints = MissionConstraints(),
    max_coarse_results: int = 200,
    fine_eval_budget_total: int = 120000,
    fine_eval_budget_per_seed: int = 30000,
    fine_max_leg_vectors_per_seed: int = 1500,
    fine_early_stop_no_improve: int = 10000,
) -> Dict:
    n_legs = len(sequence_bodies) - 1
    if n_legs < 1:
        raise ValueError("Sequence must have at least two bodies.")

    sequence_name = _sequence_name(sequence_bodies)
    start_t = Time(start_iso)
    end_t = Time(end_iso)

    print(f"[SEARCH] Sequence: {sequence_name}")
    print(f"[SEARCH] Period: {start_iso} to {end_iso}")

    launch_candidates = Time(start_iso) + np.arange(0, int((end_t - start_t).to_value("day")) + 1, 20)
    leg_vectors_coarse = _build_leg_vectors(n_legs, coarse=True)

    coarse_hits: List[Dict] = []
    coarse_scanned = 0
    near_pool: List[Tuple[float, Dict]] = []

    print(f"[COARSE] Searching {len(launch_candidates)} launch dates x {len(leg_vectors_coarse)} leg combos")
    for idx, launch_t in enumerate(launch_candidates):
        if idx % max(1, len(launch_candidates) // 10) == 0:
            print(f"  Coarse progress: {idx}/{len(launch_candidates)}")

        for leg_days in leg_vectors_coarse:
            coarse_scanned += 1
            result = evaluate_trajectory(launch_t, sequence_bodies, leg_days, constraints)
            if not result.get("valid", False):
                continue

            if result.get("feasible", False):
                coarse_hits.append(_annotate_result(result, sequence_name))
                if len(coarse_hits) % 5 == 0:
                    print(f"    Feasible found: {len(coarse_hits)}")
            else:
                if "launch_epoch_iso" in result:
                    near_pool.append((_score_near_candidate(result, constraints), _annotate_result(result, sequence_name)))

    coarse_hits = _sort_results(coarse_hits)[:max_coarse_results]
    near_pool = sorted(near_pool, key=lambda x: x[0])
    best_near = [item[1] for item in near_pool[:max(80, max_coarse_results // 2)]]

    print(f"[COARSE] Feasible: {len(coarse_hits)} | Near seeds kept: {len(best_near)}")

    if coarse_hits:
        seed_epochs = [Time(r["launch_epoch_iso"]) for r in coarse_hits[:12]]
        seed_leg_days = [r["leg_days"] for r in coarse_hits[:12]]
        print(f"[FINE] Refining around {len(seed_epochs)} feasible seeds")
    else:
        seed_epochs = [Time(r["launch_epoch_iso"]) for r in best_near[:12]]
        seed_leg_days = [r["leg_days"] for r in best_near[:12]]
        print(f"[FINE] No coarse feasible hits; refining around {len(seed_epochs)} near-feasible seeds")

    leg_vectors_fine_global = _build_leg_vectors(n_legs, coarse=False)

    local_leg_vectors: List[List[float]] = []
    for legs in seed_leg_days:
        deltas = [-15.0, 0.0, 15.0]
        choices = []
        for l in legs:
            choices.append([max(20.0, l + d) for d in deltas])
        for v in product(*choices):
            vv = list(map(float, v))
            if 700 <= sum(vv) <= 3650:
                local_leg_vectors.append(vv)

    if local_leg_vectors:
        local_leg_vectors = [list(x) for x in {tuple(v) for v in local_leg_vectors}]
    local_leg_vectors = local_leg_vectors[:3000]

    leg_vectors_fine = leg_vectors_fine_global + local_leg_vectors
    leg_vectors_fine = [list(x) for x in {tuple(v) for v in leg_vectors_fine}]

    fine_hits: List[Dict] = []
    fine_scanned = 0

    print(
        f"[FINE] Seeds={len(seed_epochs)}, launch local +/-30d step2, global leg combos={len(leg_vectors_fine)}"
    )
    print(
        "[FINE] Adaptive budget: "
        f"total<={fine_eval_budget_total}, per-seed<={fine_eval_budget_per_seed}, "
        f"legs/seed<={fine_max_leg_vectors_per_seed}, early-stop={fine_early_stop_no_improve}"
    )

    for s_idx, seed in enumerate(seed_epochs):
        if fine_scanned >= fine_eval_budget_total:
            print(f"  Fine budget reached ({fine_scanned}), stop remaining seeds.")
            break

        seed_leg = seed_leg_days[s_idx] if s_idx < len(seed_leg_days) else []
        seed_leg_vectors = _rank_leg_vectors_near_seed(
            leg_vectors_fine,
            seed_leg,
            fine_max_leg_vectors_per_seed,
        )

        local_launch = seed + np.arange(-30, 31, 2)
        seed_scanned = 0
        no_improve = 0
        best_seed_score = 1e18

        print(
            f"  Seed {s_idx + 1}/{len(seed_epochs)} at {seed.iso[:10]} | "
            f"leg combos focused={len(seed_leg_vectors)}"
        )

        for launch_t in local_launch:
            if launch_t < start_t or launch_t > end_t:
                continue
            for leg_days in seed_leg_vectors:
                if fine_scanned >= fine_eval_budget_total:
                    break
                if seed_scanned >= fine_eval_budget_per_seed:
                    break

                fine_scanned += 1
                seed_scanned += 1

                if fine_scanned % 500 == 0:
                    print(f"    Fine progress: {fine_scanned} evals, feasible={len(fine_hits)}")

                result = evaluate_trajectory(launch_t, sequence_bodies, leg_days, constraints)
                if not result.get("valid", False):
                    continue

                score = _score_near_candidate(result, constraints)
                if score < best_seed_score:
                    best_seed_score = score
                    no_improve = 0
                else:
                    no_improve += 1

                if result.get("feasible", False):
                    fine_hits.append(_annotate_result(result, sequence_name))

                if no_improve >= fine_early_stop_no_improve:
                    print(
                        f"    Seed early stop (no improvement for {fine_early_stop_no_improve} valid evals)."
                    )
                    break

            if fine_scanned >= fine_eval_budget_total or seed_scanned >= fine_eval_budget_per_seed:
                break
            if no_improve >= fine_early_stop_no_improve:
                break

        print(
            f"    Seed summary: scanned={seed_scanned}, best_near_score={best_seed_score:.3f}, "
            f"feasible_so_far={len(fine_hits)}"
        )

    fine_hits = _sort_results(fine_hits)
    print(f"[FINE] Complete. Found {len(fine_hits)} feasible windows")

    near_candidates: List[Dict] = []
    if not fine_hits:
        near_candidates = [item[1] for item in near_pool[:40]]

    print(
        f"[SUMMARY] Coarse: {coarse_scanned} evals -> {len(coarse_hits)} feasible | "
        f"Fine: {fine_scanned} evals -> {len(fine_hits)} feasible"
    )

    return {
        "coarse_scanned": coarse_scanned,
        "fine_scanned": fine_scanned,
        "feasible_windows": fine_hits,
        "top_feasible": fine_hits[:20],
        "near_feasible": near_candidates,
        "sequence_name": sequence_name,
    }


def find_launch_windows_multi_sequence(
    start_iso: str = "2026-01-01",
    end_iso: str = "2040-12-31",
    constraints: MissionConstraints = MissionConstraints(),
    max_sequences: int = 24,
    include_moon: bool = False,
    max_coarse_results: int = 120,
) -> Dict:
    """
    Heuristic multi-sequence search:
    - Adds a sequence dimension over Earth/Venus/Mars/(optional Moon)/Mercury chains
    - Runs single-sequence search for each candidate sequence
    - Aggregates global feasible windows
    """
    sequences = _generate_candidate_sequences(max_sequences=max_sequences, include_moon=include_moon)
    print(f"[MULTI] Candidate sequences: {len(sequences)}")
    for i, seq in enumerate(sequences, start=1):
        print(f"  [{i:02d}] {_sequence_name(seq)}")

    all_feasible: List[Dict] = []
    all_near: List[Dict] = []
    per_sequence: List[Dict] = []
    total_coarse = 0
    total_fine = 0

    for idx, seq in enumerate(sequences, start=1):
        seq_name = _sequence_name(seq)
        print("\n" + "=" * 72)
        print(f"[MULTI] Sequence {idx}/{len(sequences)}: {seq_name}")
        print("=" * 72)

        result = find_launch_windows(
            sequence_bodies=seq,
            start_iso=start_iso,
            end_iso=end_iso,
            constraints=constraints,
            max_coarse_results=max_coarse_results,
        )

        total_coarse += result.get("coarse_scanned", 0)
        total_fine += result.get("fine_scanned", 0)

        seq_feasible = result.get("top_feasible", [])
        seq_near = result.get("near_feasible", [])

        all_feasible.extend(seq_feasible)
        all_near.extend(seq_near)

        best_seq = _sort_results(seq_feasible)[0] if seq_feasible else (_sort_results(seq_near)[0] if seq_near else None)
        per_sequence.append(
            {
                "sequence_name": seq_name,
                "feasible_count": len(seq_feasible),
                "near_count": len(seq_near),
                "best": best_seq,
            }
        )

    all_feasible = _sort_results(all_feasible)
    all_near = sorted(all_near, key=lambda r: _score_near_candidate(r, constraints))

    print("\n" + "=" * 72)
    print("[MULTI] GLOBAL SUMMARY")
    print("=" * 72)
    print(f"Total coarse evals: {total_coarse}")
    print(f"Total fine evals:   {total_fine}")
    print(f"Global feasible:    {len(all_feasible)}")
    print(f"Global near:        {len(all_near)}")

    return {
        "coarse_scanned": total_coarse,
        "fine_scanned": total_fine,
        "feasible_windows": all_feasible,
        "top_feasible": all_feasible[:50],
        "near_feasible": all_near[:80],
        "per_sequence": per_sequence,
    }

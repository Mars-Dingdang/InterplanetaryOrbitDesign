import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth, Jupiter, Mars, Mercury, Sun, Venus
from poliastro.iod import izzo
from poliastro.twobody import Orbit


BODY_MAP = {
	"mercury": Mercury,
	"venus": Venus,
	"earth": Earth,
	"mars": Mars,
	"jupiter": Jupiter,
}


@dataclass
class MissionConstraints:
	launch_c3_max_km2_s2: float = 16.39  # (4.05 km/s)^2
	spacecraft_budget_km_s: float = 2.25
	max_duration_days: float = 3650.0
	mercury_orbit_altitude_km: float = 400.0


def parse_sequence(sequence_text: str) -> List:
	tokens = re.split(r"[\s,\-\>]+", sequence_text.strip())
	tokens = [t for t in tokens if t]
	if len(tokens) < 2:
		raise ValueError("Flyby sequence must include at least two bodies.")

	bodies = []
	for token in tokens:
		key = token.lower()
		if key not in BODY_MAP:
			valid = ", ".join([k.title() for k in BODY_MAP.keys()])
			raise ValueError(f"Unknown body '{token}'. Supported: {valid}")
		bodies.append(BODY_MAP[key])
	return bodies


def _compute_capture_dv_from_mag_km_s(v_inf_mag: float, altitude_km: float) -> float:
	"""Estimate Mercury orbit insertion Delta-v from scalar arrival V_inf."""
	mu = Mercury.k.to(u.km**3 / u.s**2).value
	r_p = (Mercury.R.to(u.km).value + altitude_km)
	v_circ = np.sqrt(mu / r_p)
	v_hyp_peri = np.sqrt(v_inf_mag ** 2 + 2.0 * mu / r_p)
	return float(max(0.0, v_hyp_peri - v_circ))


def evaluate_trajectory(
	launch_time: Time,
	sequence_bodies: Sequence,
	leg_days: Sequence[float],
	constraints: MissionConstraints,
) -> Dict:
	n_legs = len(sequence_bodies) - 1
	if len(leg_days) != n_legs:
		return {"valid": False, "reason": "leg_days length mismatch"}
	if any(d <= 0 for d in leg_days):
		return {"valid": False, "reason": "non-positive leg duration"}

	epochs = [launch_time]
	for d in leg_days:
		epochs.append(epochs[-1] + d * u.day)

	total_duration = (epochs[-1] - epochs[0]).to(u.day).value
	if total_duration > constraints.max_duration_days:
		return {"valid": False, "reason": "duration exceeds constraint"}

	legs = []
	flyby_mismatch = 0.0
	current_v_inf_mag = 0.0
	launch_excess = 0.0
	launch_c3 = 0.0

	for i in range(n_legs):
		body_a = sequence_bodies[i]
		body_b = sequence_bodies[i + 1]
		t_a = epochs[i]
		t_b = epochs[i + 1]
		tof = (t_b - t_a).to(u.s)

		if body_a is body_b:
			# Resonant same-body flyby: avoid zero-revolution Lambert failures.
			if body_a is Venus:
				flyby_mismatch += 0.10
			elif body_a is Mercury:
				# Mercury flyby acts as heliocentric braking.
				current_v_inf_mag *= 0.70
				flyby_mismatch += 0.15

			legs.append({"is_resonant": True, "tof_days": tof.to(u.day).value})
			continue

		orbit_a = Orbit.from_body_ephem(body_a, t_a)
		orbit_b = Orbit.from_body_ephem(body_b, t_b)

		try:
			(v_depart, v_arrive), *_ = izzo.lambert(Sun.k, orbit_a.r, orbit_b.r, tof)
		except Exception:
			return {"valid": False, "reason": "lambert failed"}

		v_inf_dep = v_depart - orbit_a.v
		v_inf_dep_mag = float(np.linalg.norm(v_inf_dep.to(u.km / u.s).value))

		v_inf_arr = v_arrive - orbit_b.v
		v_inf_arr_mag = float(np.linalg.norm(v_inf_arr.to(u.km / u.s).value))

		if i == 0:
			launch_excess = v_inf_dep_mag
			launch_c3 = launch_excess ** 2
			if launch_c3 > constraints.launch_c3_max_km2_s2:
				return {
					"valid": True,
					"feasible": False,
					"reason": "launch C3 too high",
					"launch_epoch_iso": launch_time.iso,
					"arrival_epoch_iso": epochs[-1].iso,
					"duration_days": float(total_duration),
					"launch_excess_km_s": launch_excess,
					"launch_c3_km2_s2": launch_c3,
					"flyby_mismatch_km_s": 0.0,
					"arrival_v_inf_km_s": v_inf_arr_mag,
					"capture_delta_v_km_s": 0.0,
					"spacecraft_delta_v_km_s": 0.0,
					"total_delta_v_km_s": launch_excess,
					"leg_days": list(float(x) for x in leg_days),
					"leg_count": n_legs,
				}
		else:
			# DSM leverage: penalize only part of V_inf mismatch.
			mismatch = abs(v_inf_dep_mag - current_v_inf_mag)
			flyby_mismatch += mismatch * 0.40

		current_v_inf_mag = v_inf_arr_mag
		legs.append({"is_resonant": False, "tof_days": tof.to(u.day).value})

	# Capture cost at final Mercury arrival.
	if sequence_bodies[-1] is not Mercury:
		return {"valid": False, "reason": "final body is not Mercury"}

	arrive_v_inf = current_v_inf_mag
	capture_dv = _compute_capture_dv_from_mag_km_s(arrive_v_inf, constraints.mercury_orbit_altitude_km)

	spacecraft_dv = flyby_mismatch + capture_dv
	feasible = spacecraft_dv <= constraints.spacecraft_budget_km_s

	return {
		"valid": True,
		"feasible": feasible,
		"reason": "ok" if feasible else "spacecraft Delta-v budget exceeded",
		"launch_epoch_iso": launch_time.iso,
		"arrival_epoch_iso": epochs[-1].iso,
		"duration_days": float(total_duration),
		"launch_excess_km_s": launch_excess,
		"launch_c3_km2_s2": launch_c3,
		"flyby_mismatch_km_s": flyby_mismatch,
		"arrival_v_inf_km_s": arrive_v_inf,
		"capture_delta_v_km_s": capture_dv,
		"spacecraft_delta_v_km_s": spacecraft_dv,
		"total_delta_v_km_s": launch_excess + spacecraft_dv,
		"leg_days": list(float(x) for x in leg_days),
		"leg_count": n_legs,
	}


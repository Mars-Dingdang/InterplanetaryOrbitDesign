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
	launch_c3_max_km2_s2: float = 12.25  # (3.5 km/s)^2
	spacecraft_budget_km_s: float = 1.5
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


def _compute_capture_dv_km_s(v_inf_arrive: u.Quantity, altitude_km: float) -> float:
	"""Estimate Delta-v to capture into a low Mercury orbit from hyperbolic approach."""
	mu = Mercury.k.to(u.km**3 / u.s**2).value
	r_p = (Mercury.R.to(u.km).value + altitude_km)
	v_inf_mag = float(np.linalg.norm(v_inf_arrive.to(u.km / u.s).value))
	v_circ = np.sqrt(mu / r_p)
	v_hyp_peri = np.sqrt(v_inf_mag ** 2 + 2.0 * mu / r_p)
	return float(max(0.0, v_hyp_peri - v_circ))


def evaluate_trajectory(
	launch_time: Time,
	sequence_bodies: Sequence,
	leg_days: Sequence[float],
	constraints: MissionConstraints,
) -> Dict:
	"""
	Evaluate one multi-flyby candidate trajectory.

	Returns a dict with key outputs:
	- total_delta_v_km_s
	- launch_excess_km_s
	- capture_delta_v_km_s
	- duration_days
	- feasible
	"""
	n_legs = len(sequence_bodies) - 1
	if len(leg_days) != n_legs:
		raise ValueError("leg_days length must equal len(sequence)-1")
	if any(d <= 0 for d in leg_days):
		return {"valid": False, "reason": "non-positive leg duration"}

	epochs = [launch_time]
	for d in leg_days:
		epochs.append(epochs[-1] + d * u.day)

	total_duration = (epochs[-1] - epochs[0]).to(u.day).value
	if total_duration > constraints.max_duration_days:
		return {"valid": False, "reason": "duration exceeds constraint"}

	legs = []
	for i in range(n_legs):
		body_a = sequence_bodies[i]
		body_b = sequence_bodies[i + 1]
		t_a = epochs[i]
		t_b = epochs[i + 1]
		tof = (t_b - t_a).to(u.s)

		orbit_a = Orbit.from_body_ephem(body_a, t_a)
		orbit_b = Orbit.from_body_ephem(body_b, t_b)

		try:
			(v_depart, v_arrive), *_ = izzo.lambert(Sun.k, orbit_a.r, orbit_b.r, tof)
		except Exception:
			return {"valid": False, "reason": "lambert failed"}

		legs.append(
			{
				"body_a": body_a,
				"body_b": body_b,
				"orbit_a": orbit_a,
				"orbit_b": orbit_b,
				"v_depart": v_depart,
				"v_arrive": v_arrive,
				"tof_days": tof.to(u.day).value,
			}
		)

	# Earth departure capability (C3) constraint.
	v_inf_launch = legs[0]["v_depart"] - legs[0]["orbit_a"].v
	launch_excess = float(np.linalg.norm(v_inf_launch.to(u.km / u.s).value))
	launch_c3 = launch_excess ** 2
	if launch_c3 > constraints.launch_c3_max_km2_s2:
		return {
			"valid": True,
			"feasible": False,
			"reason": "launch C3 too high",
			"duration_days": total_duration,
			"launch_excess_km_s": launch_excess,
			"launch_c3_km2_s2": launch_c3,
		}

	# Intermediate flyby mismatch. Ideally |v_inf_in| ~= |v_inf_out| for unpowered flyby.
	flyby_mismatch = 0.0
	for i in range(1, n_legs):
		prev_leg = legs[i - 1]
		next_leg = legs[i]
		planet_v = prev_leg["orbit_b"].v
		v_inf_in = prev_leg["v_arrive"] - planet_v
		v_inf_out = next_leg["v_depart"] - planet_v
		mismatch = abs(
			np.linalg.norm(v_inf_out.to(u.km / u.s).value)
			- np.linalg.norm(v_inf_in.to(u.km / u.s).value)
		)
		flyby_mismatch += float(mismatch)

	# Capture cost at final Mercury arrival.
	if sequence_bodies[-1] is not Mercury:
		return {"valid": False, "reason": "final body is not Mercury"}

	v_inf_arrive = legs[-1]["v_arrive"] - legs[-1]["orbit_b"].v
	arrive_v_inf = float(np.linalg.norm(v_inf_arrive.to(u.km / u.s).value))
	capture_dv = _compute_capture_dv_km_s(v_inf_arrive, constraints.mercury_orbit_altitude_km)

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


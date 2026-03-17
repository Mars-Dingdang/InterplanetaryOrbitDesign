import os
import re
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time

from poliastro.bodies import Sun, Mercury, Venus, Earth, Mars, Jupiter
from poliastro.twobody import Orbit
from poliastro.iod import izzo


"""
Q3-2: Multi-flyby pork-chop plot for inward mission to Mercury.

Input sequence example:
	Earth Venus Venus Mercury Mercury

The sequence defines consecutive legs:
	Earth->Venus, Venus->Venus, Venus->Mercury, Mercury->Mercury

This script computes an approximate total Delta-v map over launch/arrival dates
using a patched-conic style metric:
1) Departure Delta-v at first planet
2) Flyby mismatch cost (|v_inf_out| - |v_inf_in|) at intermediate planets
3) Arrival matching Delta-v at final planet
"""


BODY_MAP = {
	"mercury": Mercury,
	"venus": Venus,
	"earth": Earth,
	"mars": Mars,
	"jupiter": Jupiter,
}


def parse_sequence(text: str):
	tokens = re.split(r"[\s,\-\>]+", text.strip())
	tokens = [t for t in tokens if t]
	if len(tokens) < 2:
		raise ValueError("Sequence must include at least two bodies, e.g. 'Earth Mercury'.")

	bodies = []
	for t in tokens:
		key = t.lower()
		if key not in BODY_MAP:
			valid = ", ".join([k.title() for k in BODY_MAP.keys()])
			raise ValueError(f"Unknown body '{t}'. Supported: {valid}")
		bodies.append(BODY_MAP[key])
	return tokens, bodies


def split_times_evenly(t_launch: Time, t_arrival: Time, n_legs: int):
	tof_total = (t_arrival - t_launch).to(u.s)
	if tof_total.value <= 0:
		return None

	step = tof_total / n_legs
	return [t_launch + i * step for i in range(n_legs + 1)]


def multi_flyby_delta_v(body_seq, epoch_seq):
	"""Compute approximate total Delta-v in km/s for one trajectory candidate."""
	n_legs = len(body_seq) - 1
	legs = []

	for i in range(n_legs):
		body_a = body_seq[i]
		body_b = body_seq[i + 1]
		t_a = epoch_seq[i]
		t_b = epoch_seq[i + 1]

		tof = (t_b - t_a).to(u.s)
		if tof.value <= 0:
			return np.nan

		orbit_a = Orbit.from_body_ephem(body_a, t_a)
		orbit_b = Orbit.from_body_ephem(body_b, t_b)

		try:
			(v_depart, v_arrive), *_ = izzo.lambert(Sun.k, orbit_a.r, orbit_b.r, tof)
		except Exception:
			return np.nan

		legs.append({
			"orbit_a": orbit_a,
			"orbit_b": orbit_b,
			"v_depart": v_depart,
			"v_arrive": v_arrive,
			"t_a": t_a,
			"t_b": t_b,
		})

	dv_total = 0.0

	# Departure burn
	dv_depart = np.linalg.norm((legs[0]["v_depart"] - legs[0]["orbit_a"].v).to(u.km / u.s).value)
	dv_total += dv_depart

	# Intermediate flyby mismatch cost
	for i in range(1, n_legs):
		prev_leg = legs[i - 1]
		next_leg = legs[i]

		planet_v = prev_leg["orbit_b"].v
		v_inf_in = prev_leg["v_arrive"] - planet_v
		v_inf_out = next_leg["v_depart"] - planet_v

		# Ideal gravity assist keeps |v_inf| constant; mismatch approximates required DSM/powered flyby.
		mismatch = abs(
			np.linalg.norm(v_inf_out.to(u.km / u.s).value)
			- np.linalg.norm(v_inf_in.to(u.km / u.s).value)
		)
		dv_total += mismatch

	# Arrival matching burn
	dv_arrive = np.linalg.norm((legs[-1]["v_arrive"] - legs[-1]["orbit_b"].v).to(u.km / u.s).value)
	dv_total += dv_arrive

	return dv_total


def main():
	seq_text = input("Enter gravity-assist sequence (e.g. Earth Venus Venus Mercury Mercury): ").strip()
	name_tokens, body_seq = parse_sequence(seq_text)

	# Date windows for Earth->Mercury style design (customize if needed)
	launch_dates = Time("2026-01-01") + np.arange(0, 90, 1) * u.day
	arrival_dates = Time("2026-06-01") + np.arange(0, 200, 1) * u.day

	launch_jd, arrival_jd = np.meshgrid(launch_dates.jd, arrival_dates.jd)
	delta_v_grid = np.full(launch_jd.shape, np.nan)

	n_legs = len(body_seq) - 1

	for i in range(launch_jd.shape[0]):
		for j in range(launch_jd.shape[1]):
			t_launch = Time(launch_jd[i, j], format="jd")
			t_arrival = Time(arrival_jd[i, j], format="jd")

			epoch_seq = split_times_evenly(t_launch, t_arrival, n_legs)
			if epoch_seq is None:
				continue

			delta_v_grid[i, j] = multi_flyby_delta_v(body_seq, epoch_seq)

	plt.figure(figsize=(11, 7))
	contour = plt.contourf(launch_jd, arrival_jd, delta_v_grid, levels=60, cmap="viridis")
	plt.colorbar(contour, label="Total Delta-V (km/s)")
	plt.xlabel("Launch Date (Julian Date)")
	plt.ylabel("Arrival Date (Julian Date)")
	plt.title(f"Multi-flyby Pork-Chop Plot: {' -> '.join([t.title() for t in name_tokens])}")

	os.makedirs("img", exist_ok=True)
	safe_name = "_".join([t.lower() for t in name_tokens])
	out_path = os.path.join("img", f"Q3-2_pork_chop_{safe_name}.png")
	plt.savefig(out_path, dpi=160)
	plt.show()

	print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
	main()

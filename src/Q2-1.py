import os
import numpy as np
import matplotlib.pyplot as plt
from poliastro.bodies import Earth, Mars
from poliastro.twobody import Orbit
from poliastro.iod import izzo
from astropy import units as u
from astropy.time import Time

'''
(Pork-chop plot). Write a Python script that computes ∆v for Earth-Mars transfers as a function of launch and arrival dates. Create a contour plot (pork-chop plot).Use ephemeris data or assume circular orbits; you may solve Lambert's problem (e.g., with poliastro). 
'''


# Define the time range for launch and arrival dates
launch_dates = Time('2026-01-01') + np.arange(0, 90, 1) * u.day
arrival_dates = Time('2026-06-01') + np.arange(0, 150, 1) * u.day

# Build a 2-D grid of Julian-Date floats (avoids AttributeError on numpy object arrays of Time)
launch_jd, arrival_jd = np.meshgrid(launch_dates.jd, arrival_dates.jd)

# Initialize the delta-v grid with NaN
delta_v_grid = np.full(launch_jd.shape, np.nan)

# Compute delta-v for each combination of launch and arrival dates
for i in range(launch_jd.shape[0]):
    for j in range(launch_jd.shape[1]):
        launch_time  = Time(launch_jd[i, j],  format='jd')
        arrival_time = Time(arrival_jd[i, j], format='jd')

        tof = (arrival_time - launch_time).to(u.s)
        if tof.value <= 0:
            continue  # skip non-positive flight times

        try:
            earth_orbit = Orbit.from_body_ephem(Earth, launch_time)
            mars_orbit  = Orbit.from_body_ephem(Mars,  arrival_time)

            # izzo.lambert(k, r0, r, tof) returns a list of (v0, v1) velocity-vector tuples
            (v0, v1), *_ = izzo.lambert(Earth.k, earth_orbit.r, mars_orbit.r, tof)

            dv_dep = np.linalg.norm((v0 - earth_orbit.v).to(u.km / u.s).value)
            dv_arr = np.linalg.norm((v1 - mars_orbit.v).to(u.km / u.s).value)
            delta_v_grid[i, j] = dv_dep + dv_arr
        except Exception:
            pass  # leave as NaN
            
# Create a contour plot (pork-chop plot)
plt.figure(figsize=(10, 6))
plt.contourf(launch_jd, arrival_jd, delta_v_grid, levels=50, cmap='viridis')
plt.colorbar(label='Delta-V (km/s)')
plt.xlabel('Launch Date (Julian Date)')
plt.ylabel('Arrival Date (Julian Date)')
plt.title('Pork-Chop Plot for Earth-Mars Transfers')

# Save BEFORE plt.show() -- show() clears the figure buffer
os.makedirs('img', exist_ok=True)
plt.savefig(os.path.join('img', 'Q2-1_pork_chop_plot.png'), dpi=150)
plt.show()


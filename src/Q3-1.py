'''
(Cost of going inward). Calculate total ∆v for a Hohmann transfer from Earth
to Mercury using: rE = 1.496 × 10^8 km, rM = 5.791 × 10^7 km, µS = 1.327 × 10^11 km³/s².
Find the arrival speed relative to Mercury and compare to Mercury’s orbital speed. What does this imply for orbital insertion?

The formulas are as follows: 
- orbit velocity v = sqrt(µ/r)
- semi-major axis a = (r1 + r2) / 2
- delta-v for departure = sqrt(µ/r1) * (sqrt(2*r2/(r1 + r2)) - 1)
- delta-v for arrival = sqrt(µ/r2) * (1 - sqrt(2*r1/(r1 + r2)))
- total delta-v = delta-v for departure + delta-v for arrival

'''

import numpy as np

# Given values
rE = 1.496e8  # Earth's orbital radius in km
rM = 5.791e7   # Mercury's orbital radius in km
mu_S = 1.327e11  # Solar gravitational parameter in km³/s²

# Calculate the semi-major axis of the transfer orbit
a = (rE + rM) / 2

# Calculate delta-v for departure and arrival
delta_v_departure = np.sqrt(mu_S / rE) * (np.sqrt(2 * rM / (rE + rM)) - 1)
delta_v_arrival = np.sqrt(mu_S / rM) * (1 - np.sqrt(2 * rE / (rE + rM)))

# Calculate total delta-v
total_delta_v = delta_v_departure + delta_v_arrival

print(f"Total delta-v for Hohmann transfer: {total_delta_v:.2f} km/s")
# Calculate arrival speed relative to Mercury
earth_orbital_speed = np.sqrt(mu_S / rE)
mercury_orbital_speed = np.sqrt(mu_S / rM)
arrival_speed = earth_orbital_speed - delta_v_departure
print(f"Arrival speed relative to Mercury: {arrival_speed:.2f} km/s")
print(f"Mercury's orbital speed: {mercury_orbital_speed:.2f} km/s")
print("This means that the spacecraft's arrival speed is lower than Mercury's orbital speed and will be captured by the sun's gravity, requiring additional delta-v for orbital insertion around Mercury.")


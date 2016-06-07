# Überführt Messdaten in auslesbare Textdaten (optional)
import numpy as np

phi_l = 345.9
phi_r = 237.5
omega_l = np.array([117.6, 117.3, 117.1, 116.7, 116.5, 116.4, 115.9, 115.3])
omega_r = np.array([351, 351.2, 351.3, 351.8, 352.1, 352.3, 352.7, 353.2])
l = np.array([643.8, 579.1, 546.1, 508.6, 480.0, 467.81, 435.8, 404.7  ])

#Quelle für Kadnium: http://physics.nist.gov/PhysRefData/Handbook/Tables/cadmiumtable2_a.htm

np.savetxt('messdaten/messung.txt', np.column_stack([omega_l, omega_r, l]), header="omega_l [deg], omega_r [deg], lambda [nm]")

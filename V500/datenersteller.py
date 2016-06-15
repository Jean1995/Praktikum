# Überführt Messdaten in auslesbare Textdaten (optional)
import numpy as np

# Lange Messung, orange
U = np.array([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.02, -0.05, -0.1, -0.2, -0.3, -0.4, -0.5, -0.75, -1, -1.5, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -13.37, -14, -16, -18  ])
I = np.array([-0.036, -0.034, -0.032, -0.03, -0.028, -0.027, -0.025, -0.023, -0.022, -0.02, -0.018, -0.016, -0.014, -0.012, - 0.01, -0.009, -0.008, -0.006, -0.005, -0.004, -0.002, -0.001, 0.002, 0.008, 0.026, 0.052, 0.064, 0.081, 0.15, 0.2, 0.23, 0.27, 0.35, 0.42, 0.52, 0.61, 0.76, 0.9, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.6, 1.6, 1.6, 1.6, 1.7, 1.7])

np.savetxt('messdaten/messung_lang.txt', np.column_stack([I, U]), header="I [nA], V [Volt]")


# grün

I_1 = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
U_1 = np.array([0.45, 0.41, 0.39, 0.36, 0.35, 0.34, 0.32, 0.31, 0.31, 0.30, 0.29])

np.savetxt('messdaten/messung_1.txt', np.column_stack([I_1, U_1]), header="I [nA], V [Volt]")

# zwischen Grün und lila alias Indigo

I_2 = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
U_2 = np.array([0.5, 0.38, 0.28, 0.19, 0.11, 0.05, -0.08, -0.15, -0.21, -0.29, -0.35])

np.savetxt('messdaten/messung_2.txt', np.column_stack([I_2, U_2]), header="I [nA], V [Volt]")

# violett

I_3 = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
U_3 = np.array([1.01, 0.95, 0.91, 0.89, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.75])

np.savetxt('messdaten/messung_3.txt', np.column_stack([I_3, U_3]), header="I [nA], V [Volt]")

# UV-Linie

I_4 = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
U_4 = np.array([1.31, 1.20, 1.10, 1.04, 1.00, 0.8, 0.7, 0.66, 0.63, 0.61, 0.55 ])

np.savetxt('messdaten/messung_4.txt', np.column_stack([I_4, U_4]), header="I [nA], V [Volt]")

# orange

I_5 = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02])
U_5 = np.array([0.45, 0.38, 0.35, 0.31, 0.29, 0.26, 0.24, 0.22, 0.20, 0.19, 0.16])

np.savetxt('messdaten/messung_5.txt', np.column_stack([I_5, U_5]), header="I [nA], V [Volt]")

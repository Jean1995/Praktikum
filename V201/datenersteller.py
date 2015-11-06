import numpy as np
m_w = np.array([477.7, 485.47, 461.57, 430.22, 495.61])
m_k = np.array([385.2, 231.81, 231.81, 231.81, 107.41])
U_w = np.array([0.90, 0.90, 0.95, 0.94, 0.86])
T_w = np.array([22.49, 22.49, 23.73, 23.48, 21.49])
U_k = np.array([2.61, 2.15, 2.92, 3.02, 2.08])
T_k = np.array([64.37, 53.21, 71.84, 74.24, 75.68])
U_m = np.array([0.96, 0.99, 0.99, 1.00, 0.93])
T_m = np.array([23.98, 24.72, 24.72, 24.97, 23.23])

T_w = T_w + 273.2
T_k = T_k + 273.2
T_m = T_m + 273.2

np.savetxt('daten.txt', np.column_stack([m_w, m_k, U_w, T_w, U_k, T_k, U_m, T_m]), header="m_w [g], m_k [g], U_w [mV], T_w [K], U_k [mV], T_k [K], U_m [mV], T_m [K]")

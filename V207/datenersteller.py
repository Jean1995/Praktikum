import numpy as np
T = np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35])
U1 = np.array([0.1752, 0.1540, 0.1340, 0.1149, 0.1104, 0.0853, 0.0776, 0.0697, 0.0530, 0.0455, 0.0360, 0.0273]) # Matt
U2 = np.array([0.9883, 0.8822, 0.7913, 0.7001, 0.6133, 0.5321, 0.4540, 0.3795, 0.3039, 0.2396, 0.1762, 0.1181]) # schwarz
U3 = np.array([0.0687, 0.0641, 0.0521, 0.0495, 0.0446, 0.0443, 0.0364, 0.0361, 0.0202, 0.0177, 0.0159, 0.0149]) # glänzend
U4 = np.array([0.9454, 0.8500, 0.7699, 0.6805, 0.5933, 0.5192, 0.4432, 0.3667, 0.2977, 0.2304, 0.1707, 0.1144]) # weiß

d = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
Uabs = np.array([0.2905, 0.2857, 0.2810, 0.2748, 0.2688, 0.2631, 0.2568, 0.2495, 0.2427, 0.2351, 0.2254]) # Abstände

#Einheiten umrechnen
T = T+273.2
U1 = U1/1000
U2 = U2/1000
U3 = U3/1000
U4 = U4/1000
d = d*0.01
Uabs = Uabs/1000

o = 24.6+273.2

np.savetxt('daten.txt', np.column_stack([T, U1, U2, U3, U4]), header="T, U1, U2, U3, U4")
np.savetxt('daten2.txt', np.column_stack([d, Uabs]), header="abs, Uabs")
Uoffset = ((0.0075/1000) + (0.0088/1000))/2
dU1 = U1-Uoffset
dU2 = U2-Uoffset
dU3 = U3-Uoffset
dU4 = U4-Uoffset
dUabs = Uabs-Uoffset
dt = T**4 - o**4
np.savetxt('tabelle.txt', np.column_stack([T, dU1, dU2, dU3, dU4, dt]), header="T, dU1, dU2, dU3, dU4, dt")
np.savetxt('tabelle2.txt', np.column_stack([d, dUabs]), header="d, dUabs")

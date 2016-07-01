# Überführt Messdaten in auslesbare Textdaten (optional)
# Überführt Messdaten in auslesbare Textdaten (optional)
import numpy as np

# B-Feld-Aufabe

D = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]) # in Zoll! Obacht!
I_1 = np.array([0, 0.28, 0.58, 0.89, 1.19, 1.52, 1.84, 2.18, 2.49]) # 250 V
I_2 = np.array([0, 0.30, 0.64, 1.00, 1.34, 1.68, 2.2, 2.39, 2.73]) # 300 V
I_3 = np.array([0, 0.32, 0.69, 1.07, 1.44, 1.81, 2.2, 2.58, 2.95]) # 350 V
D_kurz = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]) # in Zoll! Obacht!
I_4 = np.array([0, 0.35, 0.74, 1.14, 1.56, 1.94, 2.34, 2.75]) # 400 V
I_5 = np.array([0, 0.4, 0.81, 1.25, 1.66, 2.09, 1.51, 2.94]) # 450 V

np.savetxt('messdaten/messung_B_lang.txt', np.column_stack([D, I_1, I_2, I_3]), header="D [Zoll], I_1 [A], I_2 [A], I_3 [A]")
np.savetxt('messdaten/messung_B_kurz.txt', np.column_stack([D_kurz, I_4, I_5]), header="D [Zoll], I_4 [A], I_5 [A]")

# Erdmangetfeld
I_erd = 0.26 # Kompensationsstrom in Ampere
U_erd = 200 # Beschleunigungsspannung
phi = 70 #grad inklinationswinkel

# E-Feld-Aufgabe

D = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]) # in Zoll! Obacht!
U_1 = np.array([-8.55, -4.58, -1.1, 2.61, 6.18, 9.59, 13.26, 16.8, 20.4]) # 200 V
U_2 = np.array([-11.09, -6.36, -1.87, 2.91, 7.59, 12, 16.6, 20.98, 25.35]) # 250 V
U_3 = np.array([-13.35, -7.82, -1.96, 3.79, 8.92, 14.64, 20.09, 25.31, 30.81]) # 300 V
D_kurz = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]) # in Zoll! Obacht!
U_4 = np.array([-14.23, -7.67, -1.06, 5.38, 11.67, 17.93, 24.32, 30.78]) # 350 V
U_5 = np.array([-17.43, -10.13, -2.4, 5.17, 12.58, 19.83, 27, 34.23]) # 400 V

np.savetxt('messdaten/messung_E_lang.txt', np.column_stack([D, U_1, U_2, U_3]), header="D [Zoll], U_1 [V], U_2 [V], U_3 [V]")
np.savetxt('messdaten/messung_E_kurz.txt', np.column_stack([D_kurz, U_4, U_5]), header="D [Zoll], U_4 [V], U_5 [V]")


# Frequenzkacke

v = np.array([39.74, 79.31, 129.3, 158.46]) # in... kHz (cant remember)
A = np.array([0.25, 9/32, 7/16, 5/16]) # in zöll
np.savetxt('messdaten/frequenzen.txt', np.column_stack([v, A]), header="v [kHz ?], A [zöll]")

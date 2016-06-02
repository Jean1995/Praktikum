# Überführt Messdaten in auslesbare Textdaten (optional)
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)

from uncertainties import ufloat
# Wellenlängenbestimmung

z_1_anfang   = np.array([18, 0, 0, 0, 22, 0, 19, 13, 11, 14]) #zählrate beginn
z_1_ende = np.array([3204, 3183, 3045, 3006, 3359, 3315, 4972, 3345, 4091, 3761])
z_1 = z_1_ende - z_1_anfang # in Impulsen

s_1_anfang = np.array([0, 0, 0.5, 5.59, 0.5, 5.18, 0.5, 14.52, 0.5, 7.31  ])
s_1_ende = np.array([5.33, 5.33, 5.59, 0.56, 5.18, 0.5, 14.52, 0.5, 7.31, 1.05 ])
s_1 = s_1_ende - s_1_anfang
s_1 = np.abs(s_1) # in mm

np.savetxt('messdaten/1.txt', np.column_stack([z_1, s_1]), header="Delta_z [Impulse], Delta_s [mm]")


# ohne kaka-Werte

z_1_anfang   = np.array([18, 0, 0, 0, 22, 0, 11, 14]) #zählrate beginn
z_1_ende = np.array([3204, 3183, 3045, 3006, 3359, 3315, 4091, 3761])
z_1 = z_1_ende - z_1_anfang # in Impulsen

s_1_anfang = np.array([0, 0, 0.5, 5.59, 0.5, 5.18, 0.5, 7.31  ])
s_1_ende = np.array([5.33, 5.33, 5.59, 0.56, 5.18, 0.5, 7.31, 1.05 ])
s_1 = s_1_ende - s_1_anfang
s_1 = np.abs(s_1) # in mm

np.savetxt('messdaten/1_1.txt', np.column_stack([z_1, s_1]), header="Delta_z [Impulse], Delta_s [mm]")





# Für Luft

z_2_anfang = np.array([ 3, 5, 4, 44, 79 ])
z_2_ende = np.array([ 19, 21, 20, 61, 96 ])
z_2 = z_2_ende-z_2_anfang

p_2 = np.array([ 0.4, 0.4, 0.4, 0.4, 0.4 ]) # Druckdifferenz in bar

np.savetxt('messdaten/2.txt', np.column_stack([z_2, p_2]), header="Delta_z [Impulse], Delta_p [bar]")

# Für C4H8

z_3_anfang = np.array([ 17, 20, 345, 578, 795 ])
z_3_ende = np.array([ 110, 139, 471, 706, 929 ])
z_3 = z_3_ende - z_3_anfang

p_3 = np.array([ 0.7, 0.7, 0.7, 0.7, 0.7])

np.savetxt('messdaten/3.txt', np.column_stack([z_3, p_3]), header="Delta_z [Impulse], Delta_p [bar]")

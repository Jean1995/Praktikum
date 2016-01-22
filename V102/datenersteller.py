import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from table import (
    make_table,
    make_SI,
    write,
)
from uncertainties import ufloat

# Vorbereitung

l_1 = np.array([55.4, 55.3, 55.35])
l_2 = np.array([5, 4.9, 5])

write('build/laengen_tabelle.tex', make_table([l_1, l_2], [1,1]))

l_1 = l_1*10**(-2)
l_2 = l_2*10**(-2)

np.savetxt('build/laengen.txt', np.column_stack([l_1, l_2]), header="Länge bis untere Kante [m], Länge von unterer Kante [m]")

r = np.array([0.195, 0.203, 0.199, 0.203, 0.194]) # durchmesser in Millimeter
r = r*10**(3)/2 #micro meter ok
write('build/radius_tabelle.tex', make_table([r], [1])) # micro meter

r = r*10**(-6) # meter

np.savetxt('build/radius.txt', np.column_stack([r]), header="Radius des Torsionsfadens [m]")

# a) Messung ohne alles

t1 = np.array([18.738, 18.726, 18.735, 18.716, 18.735, 18.734, 18.726, 18.730, 18.725, 18.735])

write('build/a_tabelle.tex', make_table([t1], [3]))
np.savetxt('build/a.txt', np.column_stack([t1]), header="Periodendauer ohne alles [s]")

# b) Messung mit Magnetfeld der Erde

t2 = np.array([18.120, 18.114, 18.114, 18.097, 18.118, 18.105, 18.111, 18.081, 18.077, 18.082])

write('build/b_tabelle.tex', make_table([t2], [3]))
np.savetxt('build/b.txt', np.column_stack([t2]), header="Periodendauer mit Erdmagnetfeld [s]")

# c) Messung mit Magnetfeld der Helmholzspulen

T_0_2 = np.array([16.782, 16.779, 16.774, 16.775, 16.768])
T_0_4 = np.array([15.636, 15.635, 15.634, 15.618, 15.609])
T_0_6 = np.array([14.661, 14.632, 14.612, 14.602, 14.596])
T_0_8 = np.array([13.723, 13.706, 13.694, 13.686, 13.673])
T_1_0 = np.array([12.896, 12.886, 12.871, 12.857, 12.838])

write('build/c_tabelle.tex', make_table([T_0_2, T_0_4, T_0_6, T_0_8, T_1_0], [3,3,3,3,3]))
np.savetxt('build/c.txt', np.column_stack([T_0_2, T_0_4, T_0_6, T_0_8, T_1_0]), header="Unter 0,2 A [s], Unter 0,4 A [s], Unter 0,6 A [s], Unter 0,8 A [s], Unter 1,0 A [s]")

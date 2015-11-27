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

#b)
R_a = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
I   = np.array([100, 86, 59, 50, 42, 36, 32, 29, 26, 23, 21])
U_k = np.array([160, 560, 870, 950, 1100, 1150, 1175, 1200, 1250, 1275, 1300])
U_k = U_k*10**(-3)
write('build/monotabelle.tex', make_table([R_a, I, U_k], [0, 3, 3]))
I = I*10**(-3)
np.savetxt('mono.txt', np.column_stack([R_a, I, U_k]), header="R_a [dings], I [A], U_k [V]")

#c)
R_a = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
I   = np.array([220, 160, 125, 76, 72, 59, 52, 47, 42, 39, 36])
U_k = np.array([3400, 2850, 2500, 2300, 2200, 2150, 2100, 2000, 1930, 1930, 1900])
U_k = U_k*10**(-3)
write('build/gegentabelle.tex', make_table([R_a, I, U_k], [0, 0, 2]))
I = I*10**(-3)
np.savetxt('gegen.txt', np.column_stack([R_a, I, U_k]), header="R_a [dings], I [A], U_k [V]")

#d)
R_a = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
I   = np.array([6.8, 6, 4.7, 3.7, 3.1, 2.75, 2.2, 2.1, 1.9, 1.7, 1.6])
U_k = np.array([180, 230, 300, 350, 390, 410, 440, 440, 450, 460, 470])
U_k = U_k*10**(-3)
write('build/rechtecktabelle.tex', make_table([R_a, I, U_k], [0, 2, 2]))
I = I*10**(-3)
np.savetxt('rechteck.txt', np.column_stack([R_a, I, U_k]), header="R_a [dings], I [A], U_k [V]")

R_a = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
I   = np.array([2.1, 1.85, 1.25, 0.92, 0.72, 0.62, 0.53, 0.45, 0.38, 0.34, 0.31])
U_k = np.array([570, 730, 1150, 1350, 1500, 1550, 1650, 1700, 1750, 1770, 1800])
U_k = U_k*10**(-3)
write('build/sinustabelle.tex', make_table([R_a, I, U_k], [0, 2, 2]))
I = I*10**(-3)
np.savetxt('sinus.txt', np.column_stack([R_a, I, U_k]), header="R_a [dings], I [A], U_k [V]")

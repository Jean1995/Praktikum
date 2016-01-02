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

# A

x = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
v = np.array([29.8, 32.8, 36.3, 40.2, 44.3, 48.9, 53.6, 58.6, 64.6, 70.5, 77.3, 85.4])

write('build/atabelle.tex', make_table([x, v], [1,1]))
np.savetxt('a.txt', np.column_stack([x, v]), header="Abstand auf x-Achse [cm], Frequenz [kHz]")

# B

f = np.array([8262, 16265, 24520, 31820, 37850, 43640, 49118, 68199, 74000, 77260, 80560])
np.savetxt('b.txt', np.column_stack([f]), header="Frequenz [H]")

# C

f = np.array([7806, 15540, 24274, 30000, 36180, 42265, 49500, 68129, 73242])
np.savetxt('c.txt', np.column_stack([f]), header="Frequenz [H]")

# D

d1_num = np.arange(0, 15)
f1 = np.array([1.99, 1.85, 1.68, 1.4, 1.1, 0.7, 0.32, 0.079, 0.48, 0.82, 1.25, 1.6, 1.9, 2.05, 2.2])
np.savetxt('d1.txt', np.column_stack([f1]), header="Spannung [V]")

f2 = np.array([1.7, 1.35, 0.8, 0.048, 0.7, 1.3, 1.7, 1.75, 1.6, 1.05, 0.56, 0.042, 0.63, 1.05, 1.35])
np.savetxt('d2.txt', np.column_stack([f2]), header="Spannung [V]")

write('build/dtabelle.tex', make_table([d1_num, f1, f2], [0,2,2]))

d3_num = np.arange(0, 15)
f3 = np.array([300, 370, 440, 500, 540, 560, 570, 540, 500, 440, 390, 340, 295, 260, 250])
np.savetxt('d3.txt', np.column_stack([f3]), header="Spannung [V]")

write('build/dtabelle2.tex', make_table([d3_num, f3], [0,0]))

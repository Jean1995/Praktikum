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

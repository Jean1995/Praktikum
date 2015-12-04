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

#a)
C_k = np.array([9.99, 8, 6.47, 5.02, 4, 3, 2.03, 1.01])
m   = np.array([14, 11, 9, 8, 6, 5, 3, 2])
write('build/atabelle.tex', make_table([C_k, m], [2,0]))
C_k = C_k*10**(-9)
np.savetxt('a.txt', np.column_stack([C_k, m]), header="C_k [F], maxima []")

#b)
C_k = np.array([9.99, 8, 6.47, 5.02, 4, 3, 2.03, 1.01])
f   = np.array([32.3, 33.33, 34.25, 34.72, 35.71, 37.31, 40.32, 46.73])
write('build/atabelle.tex', make_table([C_k, f], [2,2]))
C_k = C_k*10**(-9)
f   = f*10**(3)
np.savetxt('b.txt', np.column_stack([C_k, m]), header="C_k [F], frequenz [Hz]")

#c)
t1 = 360*10**(-3)
t2 = 552*10**(-3)
ef = 84.39*10**(3)
sf = 14.75
np.savetxt('c.txt', np.column_stack([t1, t2, ef, sf]), header="t1 [s], t2 [s], ef [Hz], sf [Hz]")

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
C_k_err = C_k * 0.005
ex   = np.array([27, 21, 17, 14, 11, 9, 6, 3])
ex_err = ex*0+1 # Kein Fehler beim ablesen? Oder etwa doch?
write('build/atabelle.tex', make_table([C_k, C_k_err, ex, ex_err], [3,3, 3, 3]))
C_k = C_k*10**(-9)
C_k_err = C_k_err*10**(-9)

np.savetxt('a.txt', np.column_stack([C_k, C_k_err, ex, ex_err]), header="C_k [F], C_k_err, ex, ex_err")

#b)
C_k = np.array([9.99, 8, 6.47, 5.02, 4, 3, 2.03, 1.01])
f   = np.array([32.3, 33.33, 34.25, 34.72, 35.71, 37.31, 40.32, 46.73])
write('build/btabelle.tex', make_table([C_k, f], [2,2])) # Hier stand vorher, dass er atabelle.tex Überschreiben soll -> I hate you for that! I just hate you!
C_k = C_k*10**(-9)
f   = f*10**(3)
np.savetxt('b.txt', np.column_stack([C_k, f]), header="C_k [F], frequenz [Hz]")

#c)
t1 = 360*10**(-3)
t2 = 552*10**(-3)
ef = 84.39*10**(3)
sf = 14.75
np.savetxt('c.txt', np.column_stack([t1, t2, ef, sf]), header="t1 [s], t2 [s], ef [Hz], sf [Hz]")

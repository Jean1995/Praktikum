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

#b und c)
f   = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 75000, 100000])
a   = np.array([900, 800, 720, 660, 600, 560, 504, 448, 416, 380, 352, 332, 312, 296, 284, 268, 256, 244, 234, 224, 204, 192, 178, 164, 154, 144, 138, 130, 124, 118, 96.8, 80, 69.6, 61.6, 53.6, 49.2, 41.2, 35.2, 31.2, 27.4, 24.4, 19.8, 16.2, 12.2, 9.76, 8.24, 6.96, 6.08, 5.36, 4.8, 3.24, 2.5])
b   = f*0 + 1/f*10**(6)
U_C = np.array([18400, 17200, 15200, 13600, 12200, 10800, 9360, 8560, 7760, 7040, 6480, 6000, 5600, 5200, 4920, 4640, 4360, 4120, 3920, 3760, 3440, 3180, 2880, 2700, 2540, 2380, 2220, 2100, 2020, 1900, 1540, 1270, 1100, 960, 856, 768, 628, 544, 476, 424, 380, 300, 256, 192, 154, 128, 110, 97.6, 86.4, 77.6, 52, 39.2])
U_C = U_C/2
f = f*2*np.pi
write('build/bctabelle.tex', make_table([f/(2*np.pi), U_C, a, b], [0,0,0,0]))


ftab1, ftab2 = np.array_split(f/(2*np.pi), 2)
atab1, atab2 = np.array_split(a, 2)
btab1, btab2 = np.array_split(b, 2)
Utab1, Utab2 = np.array_split(U_C, 2)

write('build/frequenztabelle1.tex', make_table([ftab1, atab1, btab1, Utab1], [0, 0, 0 ,0]))
write('build/frequenztabelle2.tex', make_table([ftab2, atab2, btab2, Utab2], [0, 0, 0 ,0]))


U_C = U_C * 10**(-3)
a   = a * 10**(-6)
b   = b * 10**(-6)
np.savetxt('bc.txt', np.column_stack([f, U_C, a, b]), header="f [Hz], U_c [V], a [s], b [s]")



r = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 1000, 1500, 2500, 7000, 40000, 100000])
s = np.array([900, 800, 720, 660, 600, 560, 448, 380, 332, 224, 154, 96.8, 35.2, 6.08, 2.5])
i = r*0 +1/r*10**6
k = np.array([18400, 17200, 15200, 13600, 12200, 10800, 8560, 7040, 6000, 3760, 2540, 1540, 544, 97.6, 39.2])
k = k * 10**(-3)
s = s * 10**(-6)
i = i * 10**(-6)
r = r*2*np.pi
np.savetxt('beispielwerted.txt', np.column_stack([r, k, s, i]), header="f [Hz], U_c [V], a [s], b [s]")

#52 Messwerte

#t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12])
#t = (t)*200*10**(-6)
#U = np.array([0.95, 1.65, 2.1, 2.5, 2.8, 3.1, 3.3, 3.45, 3.5, 3.55, 3.65, 3.7]) # In Sekunden
#U = U*5 # Volt

t = np.array([0.4, 0.6, 1, 2, 3, 4, 5, 6, 8])
t = t * 500 * 10**(-6)
U = np.array([5, 6.5, 9.5, 14, 17, 18.5, 19, 19.3, 19.45])
np.savetxt('a.txt', np.column_stack([t, U]), header="t [s], U [V]")
write('build/atabelle.tex', make_table([t*10**3, U], [1,1]))

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

# v) Geschwindigkeiten bestimmen; t wegen Zeit

t6  = np.array([85647, 85844, 85975, 85958, 86013])
t12 = np.array([42900, 42940, 42816, 42857, 42862])
t18 = np.array([28579, 28625, 29066, 28578, 28595])
t24 = np.array([21393, 21391, 21410, 21421, 21441])
t30 = np.array([17130, 17131, 17109, 17173, 17182])
t36 = np.array([14251, 14265, 14275, 14245, 14269])
t42 = np.array([12185, 12239, 12198, 12229, 12209])
t48 = np.array([10652, 10689, 10644, 10644, 10662])
t54 = np.array([9488, 9446, 9469, 9464, 9481])
t60 = np.array([8517, 8500, 8498, 8523, 8543])

t6  = t6  * 10**(-1)
t12 = t12 * 10**(-1)
t18 = t18 * 10**(-1)
t24 = t24 * 10**(-1)
t30 = t30 * 10**(-1)
t36 = t36 * 10**(-1)
t42 = t42 * 10**(-1)
t48 = t48 * 10**(-1)
t54 = t54 * 10**(-1)
t60 = t60 * 10**(-1)

write('build/ttabelle.tex', make_table([t6, t12, t18, t24, t30, t36, t42, t48, t54, t60], [1,1,1,1,1,1,1,1,1,1])) # in ms

t6  = t6  * 10**(-3)
t12 = t12 * 10**(-3)
t18 = t18 * 10**(-3)
t24 = t24 * 10**(-3)
t30 = t30 * 10**(-3)
t36 = t36 * 10**(-3)
t42 = t42 * 10**(-3)
t48 = t48 * 10**(-3)
t54 = t54 * 10**(-3)
t60 = t60 * 10**(-3)

np.savetxt('build/t.txt', np.column_stack([t6, t12, t18, t24, t30, t36, t42, t48, t54, t60]), header="t6 [s], t12 [s], t18 [s], t24 [s], t30 [s], t36 [s], t42 [s], t48 [s], t54 [s], t60 [s],")

# a) s wegen still, r wegen rückwärts, v wegen vorwärts
# schon Frequenzen

s = np.array([207414, 207414, 207413, 207415, 207414])
s *=10**(-1)

write('build/stabelle.tex', make_table([s], [1])) # in 1/s
np.savetxt('build/s.txt', np.column_stack([s]), header="s [1/s]")

r6  = np.array([2073.9, 2073.9, 2073.8, 2073.9, 2073.9])
r12 = np.array([2073.6, 2073.6, 2073.5, 2073.6, 2073.5])
r18 = np.array([2073.3, 2073.3, 2073.3, 2073.2, 2073.3])
r24 = np.array([2073.0, 2072.9, 2073.0, 2072.9, 2073.0])
r30 = np.array([2072.7, 2072.7, 2072.6, 2072.6, 2072.6])
r36 = np.array([2072.4, 2072.3, 2072.4, 2072.4, 2072.4])
r42 = np.array([2072.1, 2072.0, 2071.3, 2072.1, 2072.0])

write('build/rtabelle.tex', make_table([r6, r12, r18, r24, r30, r36, r42], [1,1,1,1,1,1,1])) # in 1/s
np.savetxt('build/r.txt', np.column_stack([r6, r12, r18, r24, r30, r36, r42]), header="r6 [1/s], r12 [1/s], r18 [1/s], r24 [1/s], r30 [1/s], r36 [1/s], r42 [1/s]")

v6  = np.array([2074.5, 2074.5, 2074.5, 2074.5, 2074.5])
v12 = np.array([2074.8, 2074.8, 2074.8, 2074.8, 2074.8])
v18 = np.array([2075.1, 2075.2, 2075.1, 2075.1, 2075.1])
v24 = np.array([2075.4, 2075.4, 2075.4, 2075.4, 2075.4])
v30 = np.array([2075.8, 2075.7, 2075.7, 2075.7, 2075.7])
v36 = np.array([2076.1, 2076.0, 2076.0, 2076.1, 2076.0])

write('build/vtabelle.tex', make_table([v6, v12, v18, v24, v30, v36], [1,1,1,1,1,1])) # in 1/s
np.savetxt('build/v.txt', np.column_stack([v6, v12, v18, v24, v30, v36]), header="v6 [1/s], v12 [1/s], v18 [1/s], v24 [1/s], v30 [1/s], v36 [1/s]")


# b) d wegen distanz

d = np.array([0.034, 0.909, 1.858, 2.694, 3.528, 4.393]) # in cm, beginnend mitvon links oben nach rechts unten alternierend

write('build/dtabelle.tex', make_table([d], [3])) # in cm
d *=10**(-2)
np.savetxt('build/d.txt', np.column_stack([d]), header="s [m]")


# c) i für increment, keine Ahnung

i6  = np.array([0.6, 0.6, 0.6])
i12 = np.array([1.2, 1.2, 1.2])
i18 = np.array([1.9, 1.9, 1.9])
i24 = np.array([2.5, 2.5, 2.5])
i30 = np.array([2.9, 2.8, 2.9])

write('build/dtabelle.tex', make_table([i6, i12, i18, i24, i30], [1,1,1,1,1])) # in 1/s
np.savetxt('build/d.txt', np.column_stack([i6, i12, i18, i24, i30]), header="i6 [1/s], i12 [1/s], i18 [1/s], i24 [1/s], i30 [1/s],")

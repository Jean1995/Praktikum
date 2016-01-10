import matplotlib.pyplot as plt
import numpy as np
print("Marco....")
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

t6, t12, t18, t24, t30, t36, t42, t48, t54, t60 = np.genfromtxt('build/t.txt', unpack=True)
t = unp.uarray([np.mean(t6),np.mean(t12),np.mean(t18),np.mean(t24),np.mean(t30),np.mean(t36),np.mean(t42),np.mean(t48),np.mean(t54),np.mean(t60)],[np.std(t6),np.std(t12),np.std(t18),np.std(t24),np.std(t30),np.std(t36),np.std(t42),np.std(t48),np.std(t54),np.std(t60)])
a = 43.8*10**(-2)
b = 0.1*10**(-2)
s = unp.uarray([a],[b])

dv6 = np.sqrt((np.std(t6)/np.mean(t6))**2+(b/a)**2)*(a/np.mean(t6))
rv6 = (a)/np.mean(t6)
v6 = unp.uarray(rv6, dv6)

dv12 = np.sqrt((np.std(t12)/np.mean(t12))**2+(b/a)**2)*(a/np.mean(t12))
rv12 = (a)/np.mean(t12)
v12 = unp.uarray(rv12, dv12)

dv18 = np.sqrt((np.std(t18)/np.mean(t18))**2+(b/a)**2)*(a/np.mean(t18))
rv18 = (a)/np.mean(t18)
v18 = unp.uarray(rv18, dv18)

dv24 = np.sqrt((np.std(t24)/np.mean(t24))**2+(b/a)**2)*(a/np.mean(t24))
rv24 = (a)/np.mean(t24)
v24 = unp.uarray(rv24, dv24)

dv30 = np.sqrt((np.std(t30)/np.mean(t30))**2+(b/a)**2)*(a/np.mean(t30))
rv30 = (a)/np.mean(t30)
v30 = unp.uarray(rv30, dv30)

dv36 = np.sqrt((np.std(t36)/np.mean(t36))**2+(b/a)**2)*(a/np.mean(t36))
rv36 = (a)/np.mean(t36)
v36 = unp.uarray(rv36, dv36)

dv42 = np.sqrt((np.std(t42)/np.mean(t42))**2+(b/a)**2)*(a/np.mean(t42))
rv42 = (a)/np.mean(t42)
v42 = unp.uarray(rv42, dv42)

dv48 = np.sqrt((np.std(t48)/np.mean(t48))**2+(b/a)**2)*(a/np.mean(t48))
rv48 = (a)/np.mean(t48)
v48 = unp.uarray(rv48, dv48)

dv54 = np.sqrt((np.std(t54)/np.mean(t54))**2+(b/a)**2)*(a/np.mean(t54))
rv54 = (a)/np.mean(t54)
v54 = unp.uarray(rv54, dv54)

dv60 = np.sqrt((np.std(t60)/np.mean(t60))**2+(b/a)**2)*(a/np.mean(t60))
rv60 = (a)/np.mean(t60)
v60 = unp.uarray(rv60, dv60)

rv = np.array([rv6, rv12, rv18, rv24, rv30, rv36, rv42, rv48, rv54, rv60])
dv = np.array([dv6, dv12, dv18, dv24, dv30, dv36, dv42, dv48, dv54, dv60])

v = unp.uarray(rv, dv)

rv = rv * 10**(2)
dv = dv * 10**(2)
write('build/vtabelle.tex', make_table([rv, dv], [2, 2])) # cm/s
rv = rv * 10**(-2)
dv = dv * 10**(-2)
np.savetxt('build/v.txt', np.column_stack([rv, dv]), header="v [m/s], Fehler [m/s]")





#write('build/rv6.tex', make_SI(rv6, r'\metre\per\second', figures=5))
#write('build/dv6.tex', make_SI(dv6, r'\metre\per\second', figures=5))
#write('build/v6.tex', make_SI(v6, r'\metre\per\second', figures=5))


## Beispielplot
#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
#
#
## Beispieltabelle
#a = np.linspace(1, 10, 10)
#b = np.linspace(11, 20, 10)
#write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen
#
#
## Beispielwerte
#
#
#c = ufloat(0, 0)
#write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))
print("....Polo")

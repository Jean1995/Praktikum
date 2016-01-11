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
gang = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
write('build/geschwtabelle.tex', make_table([gang, rv, dv], [1, 2, 2])) # cm/s
rv = rv * 10**(-2)
dv = dv * 10**(-2)
np.savetxt('build/geschw.txt', np.column_stack([rv, dv]), header="v [m/s], Fehler [m/s]")


s = np.genfromtxt('build/s.txt', unpack=True)
rf_0 = np.mean(s)
write('build/rf_0.tex', make_SI(rf_0, r'\per\second', figures=5))
df_0 = np.std(s)
write('build/df_0.tex', make_SI(df_0, r'\per\second', figures=5))
f_0 = ufloat(rf_0, df_0)

write('build/f_0.tex', make_SI(f_0, r'\per\second', figures=5))

d = np.genfromtxt('build/d.txt', unpack=True)
wl = np.array([d[2]-d[0], d[3]-d[1], d[4]-d[2], d[5]-d[3]])
rwl = np.mean(wl)
dwl = np.std(wl)
wl = ufloat(rwl*10**(3), dwl*10**(3))

dc = np.sqrt((dwl/rwl)**2+(df_0/rf_0)**2)*(rwl*rf_0)
rc = (rwl*rf_0)
c = ufloat(rc, dc)

write('build/wl.tex', make_SI(wl, r'\milli\metre', figures=1))
write('build/einsdurchwl.tex', make_SI(1/wl, r'\per\milli\metre', figures=1))
write('build/c.tex', make_SI(c, r'\metre\per\second', figures=3))

r6, r12, r18, r24, r30, r36, r42 = np.genfromtxt('build/r.txt', unpack=True)
rr6 = np.mean(r6)
rr12 = np.mean(r12)

r = unp.uarray([rr6 = np.mean(r6)],[])










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

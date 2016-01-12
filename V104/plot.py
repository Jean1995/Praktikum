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

write('build/f_0.tex', make_SI(f_0, r'\per\second', figures=1))

d = np.genfromtxt('build/d.txt', unpack=True)
wl = np.array([d[2]-d[0], d[3]-d[1], d[4]-d[2], d[5]-d[3]])
rwl = np.mean(wl)
dwl = np.std(wl)
wl = ufloat(rwl*10**(3), dwl*10**(3))

dc = np.sqrt((dwl/rwl)**2+(df_0/rf_0)**2)*(rwl*rf_0)
rc = (rwl*rf_0)
c = ufloat(rc, dc)

relc = ((c-343.2)/343.2)*100

write('build/wl.tex', make_SI(wl, r'\milli\metre', figures=1))
write('build/einsdurchwl.tex', make_SI(1/wl, r'\per\milli\metre', figures=1))
write('build/c.tex', make_SI(c, r'\metre\per\second', figures=3))
write('build/relc.tex', make_SI(c, r'\percent', figures=3))


r6, r12, r18, r24, r30, r36, r42 = np.genfromtxt('build/r.txt', unpack=True)
v6, v12, v18, v24, v30, v36 = np.genfromtxt('build/v.txt', unpack=True)
rr6 = np.mean(r6)
rr12 = np.mean(r12)

#r = unp.uarray([rr6 = np.mean(r6)],[])


r = unp.uarray([np.mean(v36), np.mean(v30), np.mean(v24), np.mean(v18), np.mean(v12), np.mean(v6), np.mean(r6), np.mean(r12), np.mean(r18), np.mean(r24), np.mean(r30), np.mean(r36), np.mean(r42)],[np.std(v36), np.std(v30), np.std(v24), np.std(v18), np.std(v12), np.std(v6), np.std(r6), np.std(r12), np.std(r18), np.std(r24), np.std(r30), np.std(r36), np.std(r42)])
r = r*10
write('build/test.tex', make_table([r], [2, 2]))
dr = r - f_0
#write('build/test.tex', make_SI(dr, r'\milli\metre', figures=1))
rv = unp.uarray([rv36, rv30, rv24, rv18, rv12, rv6, -rv6, -rv12, -rv18, -rv24, -rv30, -rv36, -rv42], [dv36, dv30, dv24, dv18, dv12, dv6, dv6, dv12, dv18, dv24, dv30, dv36, dv42])
rv = rv* 10**(2)

z1 = np.array([rv36, rv30, rv24, rv18, rv12, rv6, -rv6, -rv12, -rv18, -rv24, -rv30, -rv36, -rv42])
z2 = np.array([dv36, dv30, dv24, dv18, dv12, dv6, dv6, dv12, dv18, dv24, dv30, dv36, dv42]) # Das ist in m/s
z1=z1 * 10**(2)
z2=z2 * 10**(2) #jetzt in cm/s

from scipy.optimize import curve_fit

def h(x, m, b):
    return m*x + b
plt.errorbar(z1, unp.nominal_values(dr), xerr=z2, yerr=unp.std_devs(dr), fmt='r.', label=r'$\text{Messwerte} \; \increment f')
parameter, covariance = curve_fit(h, z1, unp.nominal_values(dr))
x_plot = np.linspace(-36, 30, 10000)

plt.plot(x_plot, h(x_plot, parameter[0], parameter[1]), 'b-', label=r'Ausgleichskurve', linewidth=1)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

m_fit = ufloat(parameter[0], fehler[0])
b_fit = ufloat(parameter[1], fehler[1])

write('build/propfak_1.tex', make_SI(m_fit, r'\metre\tothe{-1}', figures=1))
write('build/bwert1.tex', make_SI(b_fit, r'\per\second', figures=1))
plt.ylabel(r'$\increment f \ /\ s^{-1}$')
plt.xlabel(r'$v \ /\ cm/s$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Diese Zeile bitte in Zukunft nicht vergessen sonst unschön! <--- Du hast sie wieder raus genommen!!! >.<
plt.savefig('build/1plot.pdf')

plt.clf()



write('build/divtabelle_1.tex', make_table([rv, dr], [2, 2 ,2 ,2])) # wird angezeigt in v [cm/s], delta v [cm/s], diff f [1/s], Fehler diff f [1/s]

i6, i12, i18, i24, i30 = np.genfromtxt('build/i.txt', unpack=True)
i = unp.uarray([np.mean(i30), np.mean(i24), np.mean(i18), np.mean(i12), np.mean(i6)], [np.std(i30), np.std(i24), np.std(i18), np.std(i12), np.std(i6)])
rv = unp.uarray([rv30, rv24, rv18, rv12, rv6], [dv30, dv24, dv18, dv12, dv6])
rv = rv* 10**(2)
i = i*5

def h(x, m, b):
    return m*x + b
plt.errorbar(unp.nominal_values(rv), unp.nominal_values(i), xerr=unp.std_devs(rv), yerr=unp.std_devs(i), fmt='r.', label=r'$\text{Messwerte} \; \increment f$')
parameter, covariance = curve_fit(h, unp.nominal_values(rv), unp.nominal_values(i))
x_plot = np.linspace(5, 26, 10000)

plt.plot(x_plot, h(x_plot, parameter[0], parameter[1]), 'b-', label=r'Ausgleichskurve', linewidth=1)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

m_fit = ufloat(parameter[0], fehler[0])
b_fit = ufloat(parameter[1], fehler[1])

write('build/propfak_2.tex', make_SI(m_fit, r'\metre\tothe{-1}', figures=1))
#write('build/fit_1_m.tex', make_SI(m_fit*1000, r'\nothing\tothe{-3}', figures=1))
write('build/bwert2.tex', make_SI(b_fit, r'\per\second', figures=1))
plt.ylabel(r'$\increment f \ /\ s^{-1}$')
plt.xlabel(r'$v \ /\ cm/s$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Diese Zeile bitte in Zukunft nicht vergessen sonst unschön! <--- Du hast sie wieder raus genommen!!! >.<
plt.savefig('build/2plot.pdf')

write('build/divtabelle_2.tex', make_table([rv, i], [2, 2 ,1 ,1])) # wird angezeigt in v [cm/s], delta v [cm/s], diff f [1/s], Fehler diff f [1/s]

#write('build/rv6.tex', make_SI(rv6, r'\metre\per\second', figures=5))
#write('build/dv6.tex', make_SI(dv6, r'\metre\per\second', figures=5))
#write('build/v6.tex', make_SI(v6, r'\metre\per\second', figures=5))


deltanu_1 = unp.nominal_values(f_0)*rv60/343.2
deltanu_2 = unp.nominal_values(f_0)-unp.nominal_values(f_0)/(1-(rv60)/(343.2))
write('build/deltanu.tex', make_SI(np.abs(deltanu_1+deltanu_2), r'\per\second', figures=2))


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

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


# Bestimmung der Winkelrichtgröße d
r, phi, f = np.genfromtxt('build/statisch.txt', unpack=True)
phi = np.array([np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi])
d = f*r/phi
rd = np.mean(d)
dd = np.std(d)
d_0 = ufloat(rd,dd)
write('build/winkelrichtgroesse.tex', make_SI(d_0*10**2, r'\newton\centi\metre', figures=1))



#Bestimmung Eigenträgheitsmoment


from scipy.optimize import curve_fit

a, t = np.genfromtxt('build/dynamisch.txt', unpack=True)

def h(x, m, b):
    return m*x + b
plt.plot(a**2, t**2,'xr', label=r'$\text{Messwerte } T^2$')
parameter, covariance = curve_fit(h, a**2, t**2)
x_plot = np.linspace(0, 0.07, 10000)

plt.plot(x_plot, h(x_plot, parameter[0], parameter[1]), 'b-', label=r'Ausgleichskurve', linewidth=1)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

m_fit = ufloat(parameter[0], fehler[0])
b_fit = ufloat(parameter[1], fehler[1])

write('build/m.tex', make_SI(m_fit, r'\second\tothe{2}\per\metre\tothe{2}', figures=1))
write('build/b.tex', make_SI(b_fit, r'\second\tothe{2}', figures=1))
plt.xlabel(r'$a^2 \ /\ m^2$')
plt.ylabel(r'$T^2 \ /\ s^2$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Ich werde diese Zeile nie wieder vergessen, oh Großmeister Marco, mein Führer!
plt.ylim(unp.nominal_values(b_fit),60)
plt.xlim(0,0.07)
plt.savefig('build/plot.pdf')

plt.clf()

tr_d = b_fit*d_0/(4*np.pi) #HIER WIRD DAS BERECHNET
write('build/eigenträgheit.tex', make_SI(tr_d*10**4, r'\gram\centi\metre\tothe{2}', figures=1))
z = tr_d
# Zylinder

m, d, h, t = np.genfromtxt('build/zylinder.txt', unpack=True)

rm_z  = np.mean(m)
m_z  = ufloat(rm_z , 0)
rd_z = np.mean(d)
dd_z = np.std(d)
d_z  = ufloat(rd_z ,dd_z)
rt_z = np.mean(t)
dt_z = np.std(t)
t_z  = ufloat(rt_z ,dt_z)
write('build/m_zylinder.tex', make_SI(rm_z*10**3, r'\gram', figures=1))
write('build/r_zylinder.tex', make_SI(d_z*10**2, r'\centi\metre', figures=1))
write('build/t_zylinder.tex', make_SI(t_z, r'\seconds', figures=1))
tr_z_theorie = (m_z*d_z**2)/2
write('build/trägheit_zylinder_theorie.tex', make_SI(tr_z_theorie*10**7, r'\gram\centi\metre\thothe{2}', figures=1))
tr_z = (((t_z/(2*np.pi))**2)*d_0)
tr_z1 = tr_z - z*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT
write('build/trägheit_zylinder.tex', make_SI(tr_z1*10**7, r'\gram\centi\metre\thothe{2}', figures=1))


# Kugel

m, r, t = np.genfromtxt('build/kugel.txt', unpack=True)

rm_k = np.mean(m)
dm_k = np.std(m)
m_k  = ufloat(rm_k , 0)
rr_k = np.mean(d)
dr_k = np.std(d)
r_k  = ufloat(rr_k ,dr_k)
rt_k = np.mean(t)
dt_k = np.std(t)
t_k  = ufloat(rt_k ,dt_k)
write('build/m_kugel.tex', make_SI(m_k*10**3, r'\gram', figures=1))
write('build/r_kugel.tex', make_SI(r_k*10**2, r'\centi\metre', figures=1))
write('build/t_kugel.tex', make_SI(t_k, r'\seconds', figures=1))
tr_k_theorie = 2*(m_k*r_k**2)/5
write('build/trägheit_kugel_theorie.tex', make_SI(tr_k_theorie*10**7, r'\gram\centi\metre\thothe{2}', figures=1))
tr_k = (((t_k/(2*np.pi))**2)*d_0)
tr_k1 = tr_k - z*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT
write('build/trägheit_kugel.tex', make_SI(tr_k1*10**7, r'\gram\centi\metre\thothe{2}', figures=1))



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

import matplotlib.pyplot as plt
import numpy as np
import math
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
from scipy.optimize import curve_fit

from scipy import signal







#b)
U_0 = 9.5 #haben vergessen U_0 aufzunehmen xD, laut Bildern aber so ca 9.5, da wir daran ja nicht rumgeschraubt haben
f, U_C, a, b = np.genfromtxt('bc.txt', unpack=True)

U = U_C/U_0
plt.plot(f, U,'xr', label=r'$\text{Messwerte} \; U_C  /\  U_0$')
v=f
def h(x, m):
    return 1/np.sqrt(1+m**2*x**2)

parameter, covariance = curve_fit(h, f, U)
x_plot = np.linspace(10, 10**6, 1000000)

plt.plot(x_plot, h(x_plot, parameter[0]), 'r-', label=r'Ausgleichskurve', linewidth=1)
plt.xscale('log')
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

e = unp.uarray(parameter*(-1)*10**(3), fehler*10**(3))
write('build/wert_rc_b.tex', make_SI(e[0], r'\milli\second', figures=1))
np.savetxt('ausgleichswerte_b.txt', np.column_stack([parameter, fehler]), header="m m-Fehler")
#plt.ylim(0,0.4)
#x_plot = np.linspace(0.01, 100, 1000000)
#plt.plot(x_plot, f(x_plot), 'r-', label=r'\text{Theoriekurve} $U_{Br} \ /\  U_s$', linewidth=0.5)
plt.ylabel(r'$U_C \ /\  U_0$')
plt.xlabel(r'$\omega \ /\ s^{-1}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Diese Zeile bitte in Zukunft nicht vergessen sonst unschön!
plt.savefig('build/bplot.pdf')

plt.clf()

#c)
phi = a/b * 2 * np.pi

plt.plot(v, phi,'xr', label=r'$\text{Messwerte} \; \phi $')
plt.xscale('log')
plt.ylim(0, 1.6)
plt.yticks([0, np.pi/4, np.pi/2],[r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
def f(x, a):
    return np.arctan(a*x)

parameter, covariance = curve_fit(f, v, phi)
x_plot = np.linspace(0, 10**6, 1000000)

plt.plot(x_plot, f(x_plot, parameter[0]), 'r-', label=r'Ausgleichskurve', linewidth=1)

fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

e = unp.uarray(parameter*(1)*10**(3), fehler*10**(3))
write('build/wert_rc_c.tex', make_SI(e[0], r'\milli\second', figures=1))
np.savetxt('ausgleichswerte_cneu.txt', np.column_stack([parameter, fehler]), header="a a-Fehler")




#def g(x, k):
#    return np.arctan(k*x)
#
#parameter, covariance = curve_fit(g, v, phi)
#x_plot = np.linspace(0, 10**5, 1000)
#
#plt.plot(x_plot, g(x_plot, parameter[0]), 'r-', label=r'Ausgleichskurve', linewidth=1)
#
#fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
#plt.xscale('log')
#np.savetxt('ausgleichswerte_c.txt', np.column_stack([parameter, fehler]), header="c c-Fehler")
#plt.ylim(0,0.4)
#x_plot = np.linspace(0.01, 100, 1000000)
#plt.plot(x_plot, f(x_plot), 'r-', label=r'\text{Theoriekurve} $U_{Br} \ /\  U_s$', linewidth=0.5)
plt.ylabel(r'$\phi  \ /\ \pi$')
plt.xlabel(r'$\omega  \ /\ s^{-1}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/cplot.pdf')


plt.clf()



#d)
r, k, s, i = np.genfromtxt('beispielwerted.txt', unpack=True)
U = (k/2)/U_0
phi = s/i * 2 * np.pi
plt.polar(phi, U,'xr', label=r'$\text{Messwerte} \; \phi $')
RC = 5.47644 *10**(-3)
#def q(x , RC):
#    return -((x*RC)/(np.sqrt(1+x**2*(RC)**2)))/(x*RC)
#plt.polar(x_plot, f(x_plot, 5.47 *10**(-3)), 'r-', label=r'Theoriekurve', linewidth=1)
x = np.linspace(0, 50000, 10000000)
phi = np.arcsin(((x*RC)/(np.sqrt(1+x**2*(RC)**2))))
y = 1/(np.sqrt(1+x**2*(RC)**2))
plt.polar(phi,y,'b-')
plt.xticks([0, np.pi/4, np.pi/2,  3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4],[r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",  r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"])
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Diese Zeile bitte in Zukunft nicht vergessen sonst unschön!
plt.savefig('build/dplot.pdf')









#
#
#
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


# Beispieltabelle
#a = np.linspace(1, 10, 10)
#b = np.linspace(11, 20, 10)
#write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen


# Beispielwerte


#c = ufloat(0, 0)
#write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

plt.clf()

Umax = 19.5 # Endwert

t, U = np.genfromtxt('a.txt', unpack=True)
U = (-1)*(U-Umax) # Umformen zu einer Entladekurve #LikeABoss

write('build/atabelle_neu.tex', make_table([t*10**3, U], [1,1]))





def lin(x, m, b):
    return m*x+b

parameter2, covariance2 = curve_fit(lin, t, np.log(U))
fehler_2 = np.sqrt(np.diag(covariance2))
RC = -parameter2[0]
U0 = np.exp(parameter2[1])
RC_err = fehler_2[0]
U0_err = np.exp(fehler_2[1])

np.savetxt('ausgleichswerte_a.txt', np.column_stack([1/RC, 1/RC_err]), header="RC RC-Fehler")
c = ufloat(RC, RC_err)
c = 1/c
write('build/wert_rc_a.tex', make_SI(c*10**3, r'\milli\second' ))

x_plot = np.linspace(t[0], t[8], 10000)
def Uc_back(x, RC, U0):
    return (U0 * np.exp(-x/RC))
plt.plot(x_plot*10**(3), Uc_back(x_plot, 1/RC, U0), 'r-', label=r'Ausgleichskurve', linewidth=1)

plt.plot(t*10**(3), U,'xr', label=r'$\text{Messwerte}  \; U_C \ /\  U_0$')
plt.yscale('log')
plt.ylabel(r'$U_{c} \ /\ V$')
plt.xlabel(r'$t \ /\ ms$')
plt.legend(loc='best')
plt.xlim(0, 4.25)

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/aplot.pdf')

# Integratorshit
plt.clf()

x_plot = np.linspace(-3*np.pi, 3*np.pi, 10000)

plt.plot(x_plot, np.sin(x_plot), 'y-', label=r'sin(x)', linewidth=1)
plt.plot(x_plot, -np.cos(x_plot), 'b-', label=r'-cos(x)', linewidth=1)
plt.ylabel(r'$U \ /\ V$')
plt.xlabel(r'$t \ /\ s$')
plt.legend(loc='best')
plt.ylim(-1.5, 1.5)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/sin_theo.pdf')

plt.clf()

x_plot = np.linspace(-3*np.pi, 3*np.pi, 10000)

t = np.linspace(0, 1, 500, endpoint=False)
plt.plot(t, signal.square(2 * np.pi * 5 * t), label=r'Rechteckspannung', linewidth=1 )
#plt.plot(x_plot, signal.square(x_plot), 'b-', label=r'Rechteckspannung', linewidth=1)
plt.ylabel(r'$U \ /\ V$')
plt.xlabel(r'$t \ /\ s$')
plt.legend(loc='best')
plt.ylim(-1.5, 1.5)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/recht_theo.pdf')

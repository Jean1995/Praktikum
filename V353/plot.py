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










#b)
U_0 = 9.5 #haben vergessen U_0 aufzunehmen xD, laut Bildern aber so ca 9.5, da wir daran ja nicht rumgeschraubt haben
f, U_C, a, b = np.genfromtxt('bc.txt', unpack=True)

U = U_C/U_0
plt.plot(f, U,'xr', label=r'$\text{Messwerte} U_C \ /\  U_0$')
plt.xscale('log')
#plt.ylim(0,0.4)
#x_plot = np.linspace(0.01, 100, 1000000)
#plt.plot(x_plot, f(x_plot), 'r-', label=r'\text{Theoriekurve} $U_{Br} \ /\  U_s$', linewidth=0.5)
plt.savefig('build/bplot.pdf')
plt.ylabel(r'$U_C \ /\  U_0$')
plt.xlabel(r'$f$')
plt.legend(loc='best')


plt.clf()

#c)
phi = a/b * 2 * math.pi
plt.plot(f, phi,'xr', label=r'$\text{Messwerte}  \phi$')
plt.xscale('log')
#plt.ylim(0,0.4)
#x_plot = np.linspace(0.01, 100, 1000000)
#plt.plot(x_plot, f(x_plot), 'r-', label=r'\text{Theoriekurve} $U_{Br} \ /\  U_s$', linewidth=0.5)
plt.savefig('build/cplot.pdf')
plt.ylabel(r'$\phi$')
plt.xlabel(r'$f$')
plt.legend(loc='best')















# Beispielplot
x = np.linspace(0, 10, 1000)
y = x ** np.sin(x)
plt.plot(x, y, label='Kurve')
plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')


# Beispieltabelle
a = np.linspace(1, 10, 10)
b = np.linspace(11, 20, 10)
write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen


# Beispielwerte


c = ufloat(0, 0)
write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

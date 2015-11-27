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
from scipy.optimize import curve_fit
R_a, I, U_k = np.genfromtxt('mono.txt', unpack=True)
plt.plot(I, U_k, 'xr', label=r'$Monozelle$')
plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
plt.legend(loc='best')
def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0.02, 0.105, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade} U_1$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/monoplot.pdf')

plt.clf()

R_a, I, U_k = np.genfromtxt('gegen.txt', unpack=True)
plt.plot(I, U_k, 'xr', label=r'$Gegenspannung$')
plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
plt.legend(loc='best')
def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0.03, 0.25, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade} U_1$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/gegenplot.pdf')

plt.clf()

R_a, I, U_k = np.genfromtxt('rechteck.txt', unpack=True)
plt.plot(I, U_k, 'xr', label=r'$Rechtecksspannung$')
plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
plt.legend(loc='best')
def f(x, m, b):
    return m*x+b
x_plot = np.linspace(0.0069, 0.0015, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade} U_1$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/rechteckplot.pdf')

plt.clf()

R_a, I, U_k = np.genfromtxt('sinus.txt', unpack=True)
plt.plot(I, U_k, 'xr', label=r'$Sinusspannung$')
plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
plt.legend(loc='best')
def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0.0003, 0.0023, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade} U_1$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/sinusplot.pdf')








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

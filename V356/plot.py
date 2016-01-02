import matplotlib.pyplot as plt
import numpy as np
print("Hier könnte ihre Werbung stehen!")
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

from uncertainties import ufloat
from scipy.optimize import curve_fit

# Werte der Kette
L = 1.217*10**(-3)
C1 = 20.13*10**(-9)
C2 = 9.41*10**(-9)

# A: Dämliche Regressionen/Aggressionen
x,v = np.genfromtxt('a.txt', unpack=True)
v_log = np.log(v)

def f(x, m, b):
    return m*x + b

def f_grenz(a, b):
    return np.sqrt(2/(a*b)) / (2*np.pi)

parameter, covariance = curve_fit(f, x, v_log)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

e = unp.uarray(parameter, fehler)
write('build/regres_m.tex', make_SI(e[0], r'\milli\second', figures=1))
write('build/regres_b.tex', make_SI(e[1], r'\milli\second', figures=1))
np.savetxt('regress_ausgleichswerte.txt', np.column_stack([parameter, fehler]), header="m/m-Fehler b/b-Fehler")

# jetzt zurückrechnen
f_g1 = 9 # nach wie vielen cm finden wir die 1. Grenzfrequenz
f_g2 = 16.5 # nach wie vielen cm finden wir die 2. Grenzfrequenz
write('build/g1.tex', make_SI( np.exp(f(f_g1, parameter[0], parameter[1])), r'\kilo\hertz', figures=1))
write('build/g2.tex', make_SI( np.exp(f(f_g2, parameter[0], parameter[1])), r'\kilo\hertz', figures=1))

write('build/g1_t.tex', make_SI( 0.001*f_grenz(L, C1) , r'\kilo\hertz', figures=1))
write('build/g2_t.tex', make_SI( 0.001*f_grenz(L, C2) , r'\kilo\hertz', figures=1))

# B: Dispersionsrelation

v = np.genfromtxt('b.txt', unpack=True)
theta = np.array([np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi, 6*np.pi, 7*np.pi, 9*np.pi, 10*np.pi, 11*np.pi, 12*np.pi])



def disper_opt(a):
    return np.sqrt( 1/L * (1/C1 + 1/C2) + 1/L * np.sqrt( (1/C1 + 1/C2)**2 - (4*np.sin(a)**2)/(C1*C2) ) )

def disper_akt(a):
    return np.sqrt( 1/L * (1/C1 + 1/C2) - 1/L * np.sqrt( (1/C1 + 1/C2)**2 - (4*np.sin(a)**2)/(C1*C2) ) )

x = np.linspace(0, np.pi, 1000)
plt.plot(x, 0.001*disper_opt(x), '-b', label='Theoriekurve: Optischer Ast')
plt.plot(x, 0.001*disper_akt(x), '-g', label='Theoriekurve: Akustischer Ast')
plt.plot(theta/14,0.001*v*2*np.pi, 'xr', label='Abgelesene Werte')
plt.xlabel(r'$\theta \:\: $')
plt.ylabel(r'$\omega \:/\: \si{\kilo\hertz}$')
plt.legend(loc='best')
plt.xlim(0, np.pi)
plt.xticks([0, np.pi/4, np.pi/2,  3*np.pi/4, np.pi],[r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$",  r"$\frac{3\pi}{4}$", r"$\pi$"])
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/dispersionsrelation.pdf')

write('build/btabelle.tex', make_table([v*2*np.pi, theta/np.pi, theta/(14*np.pi)], [0,0, 2]))

plt.clf()

# C: Eigenfrequenzen

c = np.genfromtxt('c.txt', unpack=True)
theta = np.array([np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi, 6*np.pi, 7*np.pi, 9*np.pi, 10*np.pi])
#theta = np.array([1 * 0.5*np.pi, 3 * 0.5*np.pi, 5 * 0.5*np.pi, 7 * 0.5*np.pi, 9 * 0.5*np.pi, 11 * 0.5*np.pi, 13 * 0.5*np.pi , 15 * 0.5*np.pi , 17 * 0.5*np.pi])
theta = theta/14
v_ph = (c*2*np.pi)/theta
write('build/ctabelle.tex', make_table([c*2*np.pi, theta/np.pi, 0.001*v_ph], [0,2,0]))

def vph(x):
    return x/np.arcsin( np.sqrt( -0.25*x**4*L**2*C1*C2+0.5*x**2*L*C1*C2*(1/C1 + 1/C2)  ))
xplot = np.linspace(1000, 50000, 100000)
xplot2 = np.linspace(67000, 80000, 100000)
plt.plot(0.001*xplot, 0.001*vph(2*np.pi*xplot), '-b', label='Theoriekurve: Phasengeschwindigkeit')
plt.plot(0.001*xplot2, 0.001*vph(2*np.pi*xplot), '-b', label='Theoriekurve: Phasengeschwindigkeit')
plt.plot(0.001*c, 0.001*v_ph, 'xr', label='Abgelesene Werte')
plt.xlabel(r'$\nu \:/\: \si{\kilo\hertz} $')
plt.ylabel(r'$v_{\text{Ph}} \:/\: \si{\kilo\metre\per\second}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/eigenfrequenzen.pdf')

plt.clf()

# D: Die (perfekte) stehende Welle

d1 = np.genfromtxt('d1.txt', unpack=True)
d1_num = np.arange(0, 15)
plt.plot(d1_num, d1, 'xr', label='Abgelesene Werte')

xplot = np.linspace(0, 14, 1000)
plt.plot(xplot, np.abs(2.1*np.cos(np.pi*xplot/14)), '-b', label='Theoriekurve: Stehende Welle')

plt.ylabel(r'$U \:/\: \si{\volt} $')
plt.xlabel(r'$n_{\text{Kondensator}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/stehende_welle_1.pdf')


plt.clf()

d2 = np.genfromtxt('d2.txt', unpack=True)
d2_num = np.arange(0, 15)
plt.plot(d2_num, d2, 'xr', label='Abgelesene Werte')

xplot = np.linspace(0, 14, 1000)
plt.plot(xplot, np.abs(1.7*np.cos(2*np.pi*xplot/14)), '-b', label='Theoriekurve: Stehende Welle')

plt.ylabel(r'$U \:/\: \si{\volt} $')
plt.xlabel(r'$n_{\text{Kondensator}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/stehende_welle_2.pdf')


plt.clf()

d3 = np.genfromtxt('d3.txt', unpack=True)
d3_num = np.arange(0, 15)
plt.plot(d3_num, d3, 'xr', label='Abgelesene Werte')

#xplot = np.linspace(0, 14, 1000)
#plt.plot(xplot, 300+np.abs(300*np.sin(np.pi*xplot/14)), '-b', label='Theoriekurve: Stehende Welle')

plt.ylabel(r'$U \:/\: \si{\milli\volt} $')
plt.xlabel(r'$n_{\text{Kondensator}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/stehende_welle_3.pdf')

# Beispieltabelle
a = np.linspace(1, 10, 10)
b = np.linspace(11, 20, 10)
write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen


# Beispielwerte

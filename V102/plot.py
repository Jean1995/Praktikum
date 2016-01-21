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


# Apparaturdaten
m_k = ufloat(512*10**(-3),512*10**(-3)*0.0004)
R_k = ufloat(50.75*10**(-3),50.75*10**(-3)*0.00007)
R_k = R_k/2
I_h = 22.5 * 10**(-7)
I_k = 2/5 * m_k * R_k**2
I_g = I_h + I_k

r      = np.genfromtxt('build/radius.txt', unpack=True)
l1, l2 = np.genfromtxt('build/laengen.txt', unpack=True)
R = ufloat(np.mean(r),np.std(r))
L1 = ufloat(np.mean(l1),np.std(l1))
L2 = ufloat(np.mean(l2),np.std(l2))
L = L1 + L2
t1     = np.genfromtxt('build/a.txt', unpack=True)
T1 = ufloat(np.mean(t1),np.std(t1))

G = 8 * (I_g) * np.pi * (L)/((T1**2) * (R**4))
write('build/schubmodul.tex', make_SI(G*10**(-9), r'\giga\pascal', figures=1)) #etwa um 10³ zu groß









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
#

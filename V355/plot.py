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

C_k_nom, C_k_err, ex_nom, ex_err = np.genfromtxt('a.txt', unpack=True)
C_k  = unp.uarray(C_k_nom, C_k_err)
ex = unp.uarray(ex_nom, ex_err) # Extrema, entspricht wr/ws
L = 32.351*10**(-3)
C = 0.8015*10**(-9)
Cs = 0.037*10**(-9)
R = 48

w2 = unp.sqrt((1/C + 2/C_k)/L)
w1 = w2*0 + 1/np.sqrt(L*C) # Ich benutze das Array w2 um ein leeres Array w2 zu bekommen was die richtige Anzahl an Elemente hat #ThugLife
ws = (abs(w1 - w2))/2 #Schwebungsfrequenzen
wr = (w1 + w2)/2 # Resonanzfrequenzen
verh = wr/ws

rel_fehler = unp.nominal_values((abs(ex-verh))/verh)


write('build/wr_ws_verhaeltnis.tex', make_table([ex, C_k*10**(9), wr*10**(-3), ws*10**(-3),verh, rel_fehler], [1,1,1,1,1,1,1,1,1, 1]))



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

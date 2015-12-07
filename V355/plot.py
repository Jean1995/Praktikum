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

#a)

C_k_nom, C_k_err, ex_nom, ex_err = np.genfromtxt('a.txt', unpack=True)
C_k  = unp.uarray(C_k_nom, C_k_err)
ex = unp.uarray(ex_nom, ex_err) # Extrema, entspricht wr/ws
L = 32.351*10**(-3)
C = 0.8015*10**(-9)
Cs = 0.037*10**(-9)
R = 48


w2 = unp.sqrt((1/C + 2/C_k)/L) #kk die angegebene Formel war falsch, BESTE  w+
w1 = w2*0 + 1/np.sqrt(L*C) # Ich benutze das Array w2 um ein leeres Array w2 zu bekommen was die richtige Anzahl an Elemente hat #ThugLife  w-
ws = (abs(w2 - w1))/2 #Schwebungsfrequenzen
wr = (w1 + w2)/2 # Resonanzfrequenzen
verh = wr/ws


np.savetxt('daten.txt', np.column_stack([unp.nominal_values(w1), unp.nominal_values(w2)]), header="w1, w2")
np.savetxt('daten2.txt', np.column_stack([unp.nominal_values(w1/(2*math.pi)), unp.nominal_values(w2/(2*math.pi))]), header="v1, v2")
rel_fehler = unp.nominal_values((abs(ex-verh))/verh*100)


#write('build/wr_ws_verhaeltnis.tex', make_table([ex, C_k*10**(9), wr*10**(-3), ws*10**(-3),verh, rel_fehler], [1,1,1,1,1,1,1,1,1, 2]))

w2_neu = unp.sqrt(( (1/(C+Cs) + 2/C_k))/L ) #da C_alt ersetzt wurde durch C_ges = C_alt + Cs
w1_neu = w2_neu*0 + 1/unp.sqrt(L* (C+Cs))
ws_neu = (abs(w1_neu - w2_neu))/2 #Schwebungsfrequenzen
wr_neu = (w1_neu + w2_neu)/2 # Resonanzfrequenzen
verh_neu = wr_neu/ws_neu

rel_fehler_neu = unp.nominal_values((abs(ex-verh_neu))/verh_neu*100)

#write('build/wr_ws_verhaeltnis_neu.tex', make_table([ex, C_k*10**(9), wr_neu*10**(-3), ws_neu*10**(-3),verh_neu, rel_fehler_neu], [1,1,1,1,1,1,1,1,1, 2]))
np.savetxt('daten3.txt', np.column_stack([unp.nominal_values(w1_neu), unp.nominal_values(w2_neu)]), header="w1neu, w2neu")
np.savetxt('daten4.txt', np.column_stack([unp.nominal_values(w1_neu/(2*math.pi)), unp.nominal_values(w2_neu/(2*math.pi))]), header="v1neu, v2neu")


#b)

C_k, f1, f2 = np.genfromtxt('b.txt', unpack=True)
C_k = C_k*10**(-9)
f1   = f1*10**(-3)
f2   = f2*10**(-3)

rel_fehler_1 = ((abs(f1-w1_neu*10**(-3)/(2*math.pi)))/(w1_neu*10**(-3)/(2*math.pi))*100)
rel_fehler_2 = unp.nominal_values((abs(f2-w2_neu*10**(-3)/(2*math.pi)))/(w2_neu*10**(-3)/(2*math.pi))*100)

#write('build/vergleichdirekt.tex', make_table([f1, w1_neu*10**(-3)/(2*math.pi),rel_fehler_1, f2, w2_neu*10**(-3)/(2*math.pi), rel_fehler_2], [2,2,2,2,2,2]))

# U=30V
vpeak1 = 30395.15
vpeak2 = 46589.89

f = open("vpeak1.txt", "w")
f.seek (10)
f.write(str(vpeak1))
f.close ()

f = open("vpeak2.txt", "w")
f.seek (10)
f.write(str(vpeak2))
f.close ()

def I2(w, K, R, U, C, L):
    return U * 1/np.sqrt( 4*w**2 * K**2 * R**2 * Z(w, C, L)**2 + (1/(w*K) - w*K*Z(w, C, L)**2 + w*R**2*K )**2 )

def Z(w, C, L):
    return w*L - (1/w) * (1/C + 1/K)

K = 1.01*10**(-9)
R = 48
U = 30
C = 0.8015**10**(-9)
L = 32.351*10**(-3)


I1_amp = I2(vpeak1, K, R, U, C, L)
I2_amp = I2(vpeak2, K, R, U, C, L)

f = open("I_vpeak1.txt", "w")
f.seek (10)
f.write(str(I1_amp))
f.close ()

f = open("I_vpeak2.txt", "w")
f.seek (10)
f.write(str(I2_amp))
f.close ()
#write('build/I2_amp.tex', make_SI(Z(blabla), r'\volt' ))














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

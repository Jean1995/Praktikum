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
m_k = ufloat((512*10**(-3)),(512*10**(-3)*0.0004))
R_k = ufloat((50.75*10**(-3)),(50.75*10**(-3)*0.00007))
R_k = R_k/2
I_h = 22.5 * 10**(-7)
I_k = 2/5 * m_k * R_k**2
I_g = I_h + I_k

r      = np.genfromtxt('build/radius.txt', unpack=True)
l1, l2 = np.genfromtxt('build/laengen.txt', unpack=True)
R = ufloat(np.mean(r),np.std(r))
write('build/r.tex', make_SI(R*10**6, r'\micro\metre', figures=1))
L1 = ufloat(np.mean(l1),np.std(l1))
write('build/l1.tex', make_SI(L1*100, r'\centi\metre', figures=1))
L2 = ufloat(np.mean(l2),np.std(l2))
write('build/l2.tex', make_SI(L2*100, r'\centi\metre', figures=1))
L = L1 + L2

# Periodendauern

t1     = np.genfromtxt('build/a.txt', unpack=True)
T1 = ufloat(np.mean(t1),np.std(t1))
t2     = np.genfromtxt('build/b.txt', unpack=True)
T2 = ufloat(np.mean(t2),np.std(t2))
t_0_2, t_0_4, t_0_6, t_0_8, t_1_0 = np.genfromtxt('build/c.txt', unpack=True)
T_0_2 = ufloat(np.mean(t_0_2),np.std(t_0_2))
T_0_4 = ufloat(np.mean(t_0_4),np.std(t_0_4))
T_0_6 = ufloat(np.mean(t_0_6),np.std(t_0_6))
T_0_8 = ufloat(np.mean(t_0_8),np.std(t_0_8))
T_1_0 = ufloat(np.mean(t_1_0),np.std(t_1_0))

Tm = unp.uarray([np.mean(t_0_2), np.mean(t_0_4), np.mean(t_0_6), np.mean(t_0_8), np.mean(t_1_0)],[np.std(t_0_2), np.std(t_0_4), np.std(t_0_6), np.std(t_0_8), np.std(t_1_0)])
np.savetxt('build/zeiten.txt', np.column_stack([unp.nominal_values(1/T_0_2**2), unp.nominal_values(1/T_0_4**2), unp.nominal_values(1/T_0_6**2), unp.nominal_values(1/T_0_8**2), unp.nominal_values(1/T_1_0**2)]), header="1/Tm^2 [1/s^2]")
# Magnetfeld Spulen

n    = 390
R_s  = 78 * 10**(-3)
mu_0 = 4 * np.pi * 10**(-7)
B1   = mu_0 * 8 / np.sqrt(125) * n * 0.2 / R_s
B2   = mu_0 * 8 / np.sqrt(125) * n * 0.4 / R_s
B3   = mu_0 * 8 / np.sqrt(125) * n * 0.6 / R_s
B4   = mu_0 * 8 / np.sqrt(125) * n * 0.8 / R_s
B5   = mu_0 * 8 / np.sqrt(125) * n * 1 / R_s
B = np.array([B1, B2, B3, B4, B5])
np.savetxt('build/b-felder.txt', np.column_stack([B]), header="B-Felder [T]")

# der Schubmodul G

G = (L * np.pi * I_g * 8)/(T1**2 * R**4)
write('build/schubmodul.tex', make_SI(G*10**(-9), r'\giga\pascal', figures=1)) #etwa um 10³ zu groß
D = (np.pi * R**4 * G)/(2 * L)
D_echt = (np.pi * R**4 * 8.2*10**10)/(2 * L)
write('build/D.tex', make_SI(D, r'\micro\tesla', figures=1))
write('build/D_echt.tex', make_SI(D_echt, r'\micro\tesla', figures=1))

# der Elastizitätsmodul

E = 21*10**10
write('build/elastizitaetsmodul.tex', make_SI(E*10**(-9), r'\giga\newton\per\metre\tothe{2}', figures=1))

# Poissonsche Querkontraktionszahl

mu = E/(2* G) - 1
write('build/querkontraktionszahl.tex', make_SI(mu, r'\nothing', figures=1) )

# der Kompressionsmodul Q

Q = E/(3*(1-2*mu))
write('build/kompressionsmodul.tex', make_SI(Q*10**(-9), r'\giga\pascal', figures=1))

# m durch Spulenstrom

#m1 = (4 * np.pi**2 * I_g / T_0_2**2 - D) / B1
#m2 = (4 * np.pi**2 * I_g / T_0_4**2 - D) / B2
#m3 = (4 * np.pi**2 * I_g / T_0_6**2 - D) / B3
#m4 = (4 * np.pi**2 * I_g / T_0_8**2 - D) / B4
#m5 = (4 * np.pi**2 * I_g / T_1_0**2 - D) / B5
#m_0 = np.array([m1, m2, m3, m4, m5])
#m = ufloat(np.mean(m_0), np.std(m_0))
#write('build/magnetisches_moment.tex', make_SI(m, r'\ampere\metre\tothe{2}', figures=1))

from scipy.optimize import curve_fit

def h(x, m, b):
    return m*x + b
plt.errorbar(1000/unp.nominal_values(Tm)**2, B*10**3, xerr=0, yerr=0, fmt='r.', label=r'$\text{Messwerte}') #1/unp.std_devs(Tm)**2
parameter, covariance = curve_fit(h, 1000/unp.nominal_values(Tm)**2,B*10**3)
x_plot = np.linspace(3, 7, 10000)

plt.plot(x_plot, h(x_plot, parameter[0], parameter[1]), 'b-', label=r'Ausgleichskurve', linewidth=1)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

m_fit = ufloat(parameter[0], fehler[0])
b_fit = ufloat(parameter[1], fehler[1])

write('build/propfak_1.tex', make_SI(m_fit, r'\kilo\gram\per\ampere', figures=1))
write('build/bfak_1.tex', make_SI(b_fit, r'\tesla', figures=1))


plt.ylabel(r'$B \ /\ \si{\milli\tesla}$')
plt.xlabel(r'$\frac{1}{T_m²} \ /\ 10^3s^{-2}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Diese Zeile bitte in Zukunft nicht vergessen sonst unschön! <--- Du hast sie wieder raus genommen!!! >.<
plt.savefig('build/1plot.pdf')


m = 4*np.pi**2 * I_g / m_fit #jo
write('build/magnetisches_moment1.tex', make_SI(m, r'\ampere\metre\tothe{2}', figures=1))
m_alternativ = -D/b_fit #jo
write('build/magnetisches_moment2.tex', make_SI(m_alternativ/1000, r'\ampere\metre\tothe{2}', figures=1))


# B-Feld der Erde

B_erde = (4 * np.pi**2 * I_g / T2**2 - D) / m #jo
write('build/magnetfeld_erde.tex', make_SI(B_erde*10**6, r'\micro\tesla', figures=1))
B_erde_alternativ = (4 * np.pi**2 * I_g / T2**2 - D_echt) / m
write('build/magnetfeld_erde_alternativ.tex', make_SI(B_erde_alternativ*10**6, r'\micro\tesla', figures=1))
write('build/test.tex', make_SI((4 * np.pi**2 * I_g / T2**2), r'\micro\tesla', figures=1))

# Diskussionskack

G_echt = 8.2*10**10
G_abweichung = abs(G-G_echt)/(G_echt)
mu_abweichung = abs(mu - (E/(2* G_echt) - 1))/((E/(2* G_echt) - 1))
Q_abweichung = abs(Q-E/(3*(1-2*(E/(2* G_echt) - 1))))/((3*(1-2*(E/(2* G_echt) - 1))))

write('build/g_abweichung.tex', make_SI(G_abweichung*100, r'\percent', figures=1))
write('build/mu_abweichung.tex', make_SI(mu_abweichung*100, r'\percent', figures=1))
write('build/Q_abweichung.tex', make_SI(Q_abweichung*100, r'\percent', figures=1))

write('build/schubmodul_echt.tex', make_SI(G_echt*10**(-9), r'\giga\pascal', figures=0))
write('build/querkontraktionszahl_echt.tex', make_SI((E/(2* G_echt) - 1), r'\nothing', figures=1))
write('build/kompressionsmodul_echt.tex', make_SI((E/(3*(1-2*(E/(2* G_echt) - 1))))*10**(-9), r'\giga\pascal', figures=1))



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

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
#plt.plot(I, U_k, 'xr', label=r'$Monozelle$')
plt.errorbar(I, U_k, xerr=I*0.03, yerr=U_k*0.015, fmt='b.', label=r'$\text{Monozelle}$')

plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')

def f(x, m, b):
    return m*x+b


x_plot = np.linspace(0, 0.105, 1000000)

params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade } U_k$' )
plt.xlim(0, 0.105)
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/monoplot.pdf')


fehler1 = np.sqrt(np.diag(error1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
U_0 = ufloat(params1[1], fehler1[1])
R_i = ufloat(params1[0], fehler1[0])
write('build/mono_u0.tex', make_SI(U_0, r'\volt', figures=1  ))
write('build/mono_ri.tex', make_SI(-R_i, r'\ohm', figures=1  ))

R_i_mono_save = -R_i
U_0_mono_save = U_0

plt.clf()

R_a, I, U_k = np.genfromtxt('gegen.txt', unpack=True)
#plt.plot(I, U_k, 'xr', label=r'$Gegenspannung$')
plt.errorbar(I, U_k, xerr=I*0.03, yerr=U_k*0.015, fmt='b.', label=r'$\text{Gegenspannung}$')
plt.xlabel(r'$I \:/\: \si{\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0, 0.25, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade } U_k$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.xlim(0, 0.25)
plt.legend(loc='best')
plt.savefig('build/gegenplot.pdf')

fehler1 = np.sqrt(np.diag(error1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
U_0 = ufloat(params1[1], fehler1[1])
R_i = ufloat(params1[0], fehler1[0])
write('build/gegen_u0.tex', make_SI(U_0, r'\volt', figures=1 ))
write('build/gegen_ri.tex', make_SI(R_i, r'\ohm', figures=1 ))



plt.clf()

R_a, I, U_k = np.genfromtxt('rechteck.txt', unpack=True)
I = I*1000
#plt.plot(I, U_k, 'xr', label=r'$Rechtecksspannung$')
plt.errorbar(I, U_k, xerr=I*0.03, yerr=U_k*0.015, fmt='b.', label=r'$\text{Rechteckspannung}$')
plt.xlabel(r'$I \:/\: \si{\milli\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
def f(x, m, b):
    return m*x+b
x_plot = np.linspace(0, max(I)+0.5, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade } U_k$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.xlim(0, max(I)+0.5)
plt.legend(loc='best')
plt.savefig('build/rechteckplot.pdf')

fehler1 = np.sqrt(np.diag(error1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
U_0 = ufloat(params1[1], fehler1[1])
R_i = ufloat(params1[0], fehler1[0])
write('build/rechteck_u0.tex', make_SI(U_0, r'\volt', figures=1  ))
write('build/rechteck_ri.tex', make_SI(-R_i*1000, r'\ohm', figures=1  ))

plt.clf()

R_a, I, U_k = np.genfromtxt('sinus.txt', unpack=True)
I=I*1000
#plt.plot(I, U_k, 'xr', label=r'$Sinusspannung$')
plt.errorbar(I, U_k, xerr=I*0.03, yerr=U_k*0.015, fmt='b.', label=r'$\text{Sinusspannung}$')

plt.xlabel(r'$I \:/\: \si{\milli\ampere}$')
plt.ylabel(r'$U_k \:/\: \si{\volt}$')
def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0, 0.0023*1000, 1000000)
params1, error1 = curve_fit(f, I, U_k)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Ausgleichsgerade } U_k$' )
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.xlim(0, 0.0023*1000)
plt.legend(loc='best')
plt.savefig('build/sinusplot.pdf')

fehler1 = np.sqrt(np.diag(error1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
U_0 = ufloat(params1[1], fehler1[1])
R_i = ufloat(params1[0], fehler1[0])
write('build/sinus_u0.tex', make_SI(U_0, r'\volt', figures=1 ))
write('build/sinus_ri.tex', make_SI(-R_i*1000, r'\ohm', figures=1 ))

# Systematischer Fehler
Ri = R_i_mono_save
Rv = 10*10**6
Uk = 1.55

err_U0 = (Ri/Rv)*Uk
write('build/error_U0.tex', make_SI(err_U0*10**6, r'\micro\volt'))


# Leistung und co
plt.clf()
R_a, I, U_k = np.genfromtxt('mono.txt', unpack=True)

#plt.plot(U_k/I, U_k*I, 'xr', label=r'$\text{Leistung}$')

plt.ylabel(r'$P \:/\: \si{\watt}$')
plt.xlabel(r'$R_a \:/\: \si{\ohm}$')


def P(x):
    U_0 = U_0_mono_save.nominal_value
    R_i = R_i_mono_save.nominal_value
    return U_0**2 * x * 1/(R_i+x)**2

x_plot = np.linspace(0, max(U_k/I), 1000000)
plt.plot(x_plot, P(x_plot), '-r', label=r'$\text{Theoriekurve}$')

I = unp.uarray(I, I*0.03)
U_k = unp.uarray(U_k, U_k*0.015)
P = U_k*I
R_a = U_k/I
write('build/P_tabelle.tex', make_table([R_a, P], [1,1]))

plt.errorbar(unp.nominal_values(R_a), unp.nominal_values(P), xerr=unp.std_devs(R_a), yerr=unp.std_devs(P), fmt='b.', label=r'$\text{Leistung}$')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.legend(loc='best')
plt.savefig('build/leistung.pdf')


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

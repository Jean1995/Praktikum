import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

T, U1, U2, U3, U4 = np.genfromtxt('daten.txt', unpack=True)


Uoffset = ((0.0075/1000) + (0.0088/1000))/2
U1 = U1-Uoffset

def f(x, m, b):
    return m*x+b

Tnull = 24.4+273.2

## Unsere Ausgleichsrechnung:
x = T**4 - Tnull**4
x = x-Tnull**4
y = U1
xy = x * y
xy_mittelwert = (1/12) * (np.sum(xy))
x_mittelwert = (1/12) * (np.sum(x))
y_mittelwert = (1/12) * (np.sum(y))

x2_mittelwert = (1/12) * (np.sum(x * x))

b = (x2_mittelwert * y_mittelwert - x_mittelwert * xy_mittelwert) / (x2_mittelwert - x_mittelwert**2)
m = (xy_mittelwert - x_mittelwert * y_mittelwert) / (x2_mittelwert - x_mittelwert**2)
bla = np.array([m, b])
np.savetxt('meine_ausgleichwerte', np.column_stack([bla]), header = "m b" )

#x_plot = np.linspace(0, 10**10, 1000000)

#plt.plot(x_plot, f(x_plot, m, b-Uoffset*10), '-k', label=r'' )

#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$T^4 - T_0^4 \:/\: \si{\kelvin\tothe{4}}$')
#plt.ylabel(r'$U \:/\: \si{\volt}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')

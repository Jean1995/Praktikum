import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

#Daten einlesen
T, U1, U2, U3, U4 = np.genfromtxt('daten.txt', unpack=True)
d, Uabs = np.genfromtxt('daten2.txt', unpack=True)

Uoffset = ((0.0075/1000) + (0.0088/1000))/2
U1 = U1-Uoffset
U2 = U2-Uoffset
U3 = U3-Uoffset
U4 = U4-Uoffset
Uabs = Uabs-Uoffset


Tnull = 24.4+273.2

plt.plot(T**4-Tnull**4, U1, 'xr', label=r'$U_1$')
plt.plot(T**4-Tnull**4, U2, 'xb', label=r'$U_2$')
plt.plot(T**4-Tnull**4, U3, 'xg', label=r'$U_3$')
plt.plot(T**4-Tnull**4, U4, 'xk', label=r'$U_4$')


#fitten
from scipy.optimize import curve_fit

def f(x, m, b):
    return m*x+b

x_plot = np.linspace(0, 10**10, 1000000)

params1, error1 = curve_fit(f, T**4-Tnull**4, U1)
plt.plot(x_plot, f(x_plot, params1[0], params1[1]), '-r', label=r'$\text{Theoriekurve} U_1$' )
fehler1 = np.sqrt(np.diag(error1))

params2, error2 = curve_fit(f, T**4-Tnull**4, U2)
plt.plot(x_plot, f(x_plot, params2[0], params2[1]), '-b', label=r'$\text{Theoriekurve} U_2$' )
fehler2 = np.sqrt(np.diag(error2))

params3, error3 = curve_fit(f, T**4-Tnull**4, U3)
plt.plot(x_plot, f(x_plot, params3[0], params3[1]), '-g', label=r'$\text{Theoriekurve} U_3$' )
fehler3 = np.sqrt(np.diag(error3))

params4, error4 = curve_fit(f, T**4-Tnull**4, U4)
plt.plot(x_plot, f(x_plot, params4[0], params4[1]), '-k', label=r'$\text{Theoriekurve} U_4$' )
fehler4 = np.sqrt(np.diag(error4))



np.savetxt('ausgleichswerte1.txt', np.column_stack([params1, fehler1]), header="params1, error" )

np.savetxt('ausgleichswerte2.txt', np.column_stack([params2, fehler2]), header="params2, error" )

np.savetxt('ausgleichswerte3.txt', np.column_stack([params3, fehler3]), header="params3, error" )

np.savetxt('ausgleichswerte4.txt', np.column_stack([params4, fehler4]), header="params4, error" )



## Unsere Ausgleichsrechnung:
x = T**4 - Tnull**4
#x = x-Tnull**4
y = U1
xy = x * y
xy_mittelwert = (1/12) * (np.sum(xy))
x_mittelwert = (1/12) * (np.sum(x))
y_mittelwert = (1/12) * (np.sum(y))

x2_mittelwert = (1/12) * (np.sum(x * x))

b = (x2_mittelwert * y_mittelwert - x_mittelwert * xy_mittelwert) / (x2_mittelwert - x_mittelwert**2)
m = (xy_mittelwert - x_mittelwert * y_mittelwert) / (x2_mittelwert - x_mittelwert**2)
bla = np.array([m, b])

fstr = (y-y_mittelwert)**2
sigma = np.sqrt( (1/12) * sum(fstr) )

error_m =np.sqrt( (sigma**2)/(12*(x2_mittelwert-x_mittelwert**2)) )
error_b =np.sqrt( (sigma**2 * x2_mittelwert)/(12*(x2_mittelwert-x_mittelwert**2)) )

#error_b = np.sqrt( np.sum(x**2)/(12**2*np.sum(x**2)- np.sum(x)**2) * np.std(y)**2 )

bla2 = np.array([error_m, error_b])

np.savetxt('meine_ausgleichwerte', np.column_stack([bla, bla2]), header = "m b" )

x_plot = np.linspace(0, 10**10, 1000000)

plt.plot(x_plot, f(x_plot, m, b), '-k', label=r'$\text{Theoriekurve manuel} U_1$' )




#plt.plot(x, y, label='Kurve')
plt.xlabel(r'$T^4 - T_0^4 \:/\: \si{\kelvin\tothe{4}}$')
plt.ylabel(r'$U \:/\: \si{\volt}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

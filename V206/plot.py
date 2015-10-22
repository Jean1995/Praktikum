import matplotlib.pyplot as plt
import numpy as np

t, T1, Pb, T2, Pa, P = np.genfromtxt('daten.txt', unpack=True)

T1=T1+273.2
T2=T2+273.2
plt.plot(t, T1,'xr', label=r'$T_1$')
plt.plot(t, T2,'xb', label=r'$T_2$')


# fitten
from scipy.optimize import curve_fit

def f(x, a, b, c):
    return a*x*x + b*x + c

parameter1, covariance1 = curve_fit(f, t, T1)
parameter2, covariance2 = curve_fit(f, t, T2)
x_plot = np.linspace(0, 30, 1000)

plt.plot(x_plot, f(x_plot, parameter1[0], parameter1[1], parameter1[2]), 'r-', label=r'Theoriekurve $T_1$', linewidth=1)
plt.plot(x_plot, f(x_plot, parameter2[0], parameter2[1], parameter2[2]), 'b-', label=r'Theoriekurve $T_2$', linewidth=1)

np.savetxt('ausgleichswerte.txt', np.column_stack([parameter1, parameter2]), header="T1 T2")



plt.xlabel(r'$t \:/\: \si{\minute}$')
plt.ylabel(r'$T \:/\: \si{\kelvin}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

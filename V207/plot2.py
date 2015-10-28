import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp


d, Uabs = np.genfromtxt('daten2.txt', unpack=True)

plt.plot(d, Uabs,'xr', label=r'$U_d$')


plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x**(-2) + b

par1, e1 = curve_fit(f, d, Uabs)

x_plot = np.linspace(0.05, 0.25, 100000)
plt.plot(x_plot, f(x_plot, par1[0], par1[1]), 'b-',label=r'Theoriekurve $U_d$', linewidth=1)

fehler = np.sqrt(np.diag(e1))

np.savetxt('ausgleichswerte_d.txt', np.column_stack([par1, e1]), header= "d U_d")

plt.xlabel(r'$d \:/\: \si{\metre}$')
plt.ylabel(r'$U_d \:/\: \si{\volt}$')
plt.legend(loc='best')

plt.savefig('build/plot2.pdf')

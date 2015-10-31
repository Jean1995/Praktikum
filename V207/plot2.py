import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp


d, Uabs = np.genfromtxt('daten2.txt', unpack=True)
Uoffset = ((0.0075/1000) + (0.0088/1000))/2
Uabs = Uabs-Uoffset

plt.plot(d, Uabs,'xr', label=r'$U_d$')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x**(-2) + b

par1, e1 = curve_fit(f, d, Uabs)

x_plot = np.linspace(0.05, 0.25, 100000)
plt.plot(x_plot, f(x_plot, par1[0], par1[1]), 'b-',label=r'Theoriekurve $U_d$', linewidth=1)

fehler = np.sqrt(np.diag(e1))

np.savetxt('ausgleichswerte_d.txt', np.column_stack([par1, fehler]), header= "d U_d")



## Unsere Ausgleichsrechnung:

y = Uabs
x = 1/d**2

b = ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) ) / ( 11 * np.sum(x**2) - np.sum(x)**2 )
m = ( 11 * np.sum(x*y) - np.sum(x) * np.sum(y) ) / ( 11 * np.sum(x**2) - np.sum(x)**2 )

delta_y = y-b-m*x
sy_quadrat = ( np.sum(delta_y**2) ) / (11 - 2)

sigma_b_quadrat = sy_quadrat * ( np.sum(x**2) ) / ( 11 * np.sum(x**2) - np.sum(x)**2 )
sigma_m_quadrat = sy_quadrat * ( 11 ) / ( 11 * np.sum(x**2) - np.sum(x)**2 )

mb = np.array([m, b])
mb_sigma = np.array([np.sqrt(sigma_m_quadrat), np.sqrt(sigma_b_quadrat)])


np.savetxt('Uabs_Ausgleichswerte', np.column_stack([mb, mb_sigma]), header="m/b m_err/b_err" )

##

plt.xlabel(r'$d \:/\: \si{\metre}$')
plt.ylabel(r'$U_d \:/\: \si{\volt}$')
plt.legend(loc='best')

plt.savefig('build/plot2.pdf')

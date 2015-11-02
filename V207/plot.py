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


Tnull = 24.6+273.2

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

y = U1
x = T**4 - Tnull**4

b = ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
m = ( 12 * np.sum(x*y) - np.sum(x) * np.sum(y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

delta_y = y-b-m*x
sy_quadrat = ( np.sum(delta_y**2) ) / (12 - 2)

sigma_b_quadrat = sy_quadrat * ( np.sum(x**2) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
sigma_m_quadrat = sy_quadrat * ( 12 ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

mb = np.array([m, b])
mb_sigma = np.array([np.sqrt(sigma_m_quadrat), np.sqrt(sigma_b_quadrat)])

m1 = m
m1_error=np.sqrt(sigma_m_quadrat)

np.savetxt('U1_ausgleichswerte.txt', np.column_stack([mb, mb_sigma]), header="m/b m_err/b_err" )

## Unsere Ausgleichsrechnung U2:

y = U2
x = T**4 - Tnull**4

b = ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
m = ( 12 * np.sum(x*y) - np.sum(x) * np.sum(y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

delta_y = y-b-m*x
sy_quadrat = ( np.sum(delta_y**2) ) / (12 - 2)

sigma_b_quadrat = sy_quadrat * ( np.sum(x**2) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
sigma_m_quadrat = sy_quadrat * ( 12 ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

mb = np.array([m, b])
mb_sigma = np.array([np.sqrt(sigma_m_quadrat), np.sqrt(sigma_b_quadrat)])

m2 = m
m2_error=np.sqrt(sigma_m_quadrat)

np.savetxt('U2_ausgleichswerte.txt', np.column_stack([mb, mb_sigma]), header="m/b m_err/b_err" )

## Unsere Ausgleichsrechnung U3:

y = U3
x = T**4 - Tnull**4

b = ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
m = ( 12 * np.sum(x*y) - np.sum(x) * np.sum(y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

delta_y = y-b-m*x
sy_quadrat = ( np.sum(delta_y**2) ) / (12 - 2)

sigma_b_quadrat = sy_quadrat * ( np.sum(x**2) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
sigma_m_quadrat = sy_quadrat * ( 12 ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

mb = np.array([m, b])
mb_sigma = np.array([np.sqrt(sigma_m_quadrat), np.sqrt(sigma_b_quadrat)])

m3 = m
m3_error=np.sqrt(sigma_m_quadrat)

np.savetxt('U3_ausgleichswerte.txt', np.column_stack([mb, mb_sigma]), header="m/b m_err/b_err" )

## Unsere Ausgleichsrechnung U4:

y = U4
x = T**4 - Tnull**4

b = ( np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x*y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
m = ( 12 * np.sum(x*y) - np.sum(x) * np.sum(y) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

delta_y = y-b-m*x
sy_quadrat = ( np.sum(delta_y**2) ) / (12 - 2)

sigma_b_quadrat = sy_quadrat * ( np.sum(x**2) ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )
sigma_m_quadrat = sy_quadrat * ( 12 ) / ( 12 * np.sum(x**2) - np.sum(x)**2 )

mb = np.array([m, b])
mb_sigma = np.array([np.sqrt(sigma_m_quadrat), np.sqrt(sigma_b_quadrat)])

m4 = m
m4_error=np.sqrt(sigma_m_quadrat)

np.savetxt('U4_ausgleichswerte.txt', np.column_stack([mb, mb_sigma]), header="m/b m_err/b_err" )



#plt.plot(x, y, label='Kurve')
plt.xlabel(r'$T^4 - T_0^4 \:/\: \si{\kelvin\tothe{4}}$')
plt.ylabel(r'$U \:/\: \si{\volt}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

# Bestimmung der Steigungen und somit der Emissionsvermögen

epsilon1 = m1/m2
epsilon2 = m2/m2
epsilon3 = m3/m2
epsilon4 = m4/m2
epsilon = np.array([epsilon1, epsilon2, epsilon3, epsilon4])

error_epsilon1 = np.sqrt( ((m1_error**2) / (m2**2)) + ((m1**2) / (m2**4)) * m2_error**2)
error_epsilon2 = np.sqrt( ((m2_error**2) / (m2**2)) + ((m2**2) / (m2**4)) * m2_error**2)
error_epsilon3 = np.sqrt( ((m3_error**2) / (m2**2)) + ((m3**2) / (m2**4)) * m2_error**2)
error_epsilon4 = np.sqrt( ((m4_error**2) / (m2**2)) + ((m4**2) / (m2**4)) * m2_error**2)
error_epsilon = np.array([error_epsilon1, error_epsilon2, error_epsilon3, error_epsilon4])

np.savetxt('Emissionsvermoegen.txt', np.column_stack([epsilon, error_epsilon]), header = "epsilon epsilon_error")

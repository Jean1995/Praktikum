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

def f_ab(x, a, b):
    return 2*a*x + b

parameter1, covariance1 = curve_fit(f, t, T1)
parameter2, covariance2 = curve_fit(f, t, T2)
x_plot = np.linspace(0, 31, 1000)

plt.plot(x_plot, f(x_plot, parameter1[0], parameter1[1], parameter1[2]), 'r-', label=r'Theoriekurve $T_1$', linewidth=1)
plt.plot(x_plot, f(x_plot, parameter2[0], parameter2[1], parameter2[2]), 'b-', label=r'Theoriekurve $T_2$', linewidth=1)

fehler1 = np.sqrt(np.diag(covariance1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
fehler2 = np.sqrt(np.diag(covariance2))

np.savetxt('ausgleichswerte.txt', np.column_stack([parameter1, parameter2, fehler1, fehler2]), header="T1 T2 T1_Sigma, T2_Sigma")



plt.xlabel(r'$t \:/\: \si{\minute}$')
plt.ylabel(r'$T \:/\: \si{\kelvin}$')
plt.legend(loc='best')
plt.xlim(0, 31)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

# Differentialquotienten
# wähle 7, 14, 21, 28 Minuten

d7_1 = f_ab(7, parameter1[0], parameter1[1])
d7_2 = f_ab(7, parameter2[0], parameter2[1])

d14_1 = f_ab(14, parameter1[0], parameter1[1])
d14_2 = f_ab(14, parameter2[0], parameter2[1])

d21_1 = f_ab(21, parameter1[0], parameter1[1])
d21_2 = f_ab(21, parameter2[0], parameter2[1])

d28_1 = f_ab(28, parameter1[0], parameter1[1])
d28_2 = f_ab(28, parameter2[0], parameter2[1])

d1 = np.array([d7_1, d14_1, d21_1, d28_1])
d2 = np.array([d7_2, d14_2, d21_2, d28_2])

np.savetxt('diffquotienten.txt', np.column_stack([d1, d2]), header="d1 d2")

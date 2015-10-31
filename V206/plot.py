import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

t, T1, Pb, T2, Pa, P , T1_error, Pb_error, T2_error, Pa_error, P_error = np.genfromtxt('daten.txt', unpack=True)

T1=T1+273.2
T2=T2+273.2
t=t*60
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
x_plot = np.linspace(0, 31*60, 1000)

plt.plot(x_plot, f(x_plot, parameter1[0], parameter1[1], parameter1[2]), 'r-', label=r'Ausgleichspolynom $T_1$', linewidth=1)
plt.plot(x_plot, f(x_plot, parameter2[0], parameter2[1], parameter2[2]), 'b-', label=r'Ausgleichspolynom $T_2$', linewidth=1)

fehler1 = np.sqrt(np.diag(covariance1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar
fehler2 = np.sqrt(np.diag(covariance2))

np.savetxt('ausgleichswerte.txt', np.column_stack([parameter1, parameter2, fehler1, fehler2]), header="T1 T2 T1_Sigma, T2_Sigma")



plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$T \:/\: \si{\kelvin}$')
plt.legend(loc='best')
plt.xlim(0, 31*60)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

# Differentialquotienten
# wähle 7, 14, 21, 28 Minuten

parameter1 = unp.uarray(parameter1, fehler1)
parameter2 = unp.uarray(parameter2, fehler2)

d7_1 = f_ab(7*60, parameter1[0], parameter1[1])
d7_2 = f_ab(7*60, parameter2[0], parameter2[1])

d14_1 = f_ab(14*60, parameter1[0], parameter1[1])
d14_2 = f_ab(14*60, parameter2[0], parameter2[1])

d21_1 = f_ab(21*60, parameter1[0], parameter1[1])
d21_2 = f_ab(21*60, parameter2[0], parameter2[1])

d28_1 = f_ab(28*60, parameter1[0], parameter1[1])
d28_2 = f_ab(28*60, parameter2[0], parameter2[1])

d1 = np.array([d7_1, d14_1, d21_1, d28_1])
d2 = np.array([d7_2, d14_2, d21_2, d28_2])

np.savetxt('diffquotienten.txt', np.column_stack([unp.nominal_values(d1), unp.nominal_values(d2), unp.std_devs(d1), unp.std_devs(d2)]), header="d1 d2 err_d1 err_d2")
# Güteziffer bestimmen

# N = Leistungsaufnahme Kompressor
# m1 = Masse Wasser
# c1 = spez. Wärmekapazität Wasser
# c2 = m2*c2 = Wärmekapazität Schlange
# d = Differenzenquotienten
def v(N, m1, c1, c2, d):
    return (1/N)*(m1*c1+c2) * d

N = ufloat(np.mean(P[1:31]), np.std(P[1:31]))
m1 = 3.992
c1 = 4184
c2 = 750

# Für T1
v7_1 = v(N, m1, c1, c2, d7_1)
v14_1 = v(N, m1, c1, c2, d14_1)
v21_1 = v(N, m1, c1, c2, d21_1)
v28_1 = v(N, m1, c1, c2, d28_1)

# Für T2
v7_2 = v(N, m1, c1, c2, d7_2)
v14_2 = v(N, m1, c1, c2, d14_2)
v21_2 = v(N, m1, c1, c2, d21_2)
v28_2 = v(N, m1, c1, c2, d28_2)

v1 = np.array([v7_1, v14_1, v21_1, v28_1])
v2 = np.array([v7_2, v14_2, v21_2, v28_2])

np.savetxt('gueteziffern.txt', np.column_stack([unp.nominal_values(v1), unp.nominal_values(v2), unp.std_devs(v1), unp.std_devs(v2)]), header="v T1, v T2, err_v T1, err_v T2")

# v real

T1 = unp.uarray(T1, T1_error)
T2 = unp.uarray(T2, T2_error)

v_ideal_7 = (T1[7])/(T1[7]-T2[7])
v_ideal_14 = (T1[14])/(T1[14]-T2[14])
v_ideal_21 = (T1[21])/(T1[21]-T2[21])
v_ideal_28 = (T1[28])/(T1[28]-T2[28])

v_ideal = np.array([v_ideal_7, v_ideal_14, v_ideal_21, v_ideal_28])
np.savetxt('ideale_gueteziffern.txt', np.column_stack([unp.nominal_values(v_ideal),  unp.std_devs(v_ideal)]), header="v_ideal v_ideal_error")

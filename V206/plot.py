import matplotlib.pyplot as plt
import numpy as np

t, T1, Pb, T2, Pa, P = np.genfromtxt('daten.txt', unpack=True)

T1=T1+273.2
T2=T2+273.2
plt.plot(t, T1,'xr', label=r'$T_1$')
plt.plot(t, T2,'xb', label=r'$T_2$')
plt.xlabel(r'$t \:/\: \si{\minute}$')
plt.ylabel(r'$T \:/\: \si{\kelvin}$')
plt.legend(loc='best')



# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')

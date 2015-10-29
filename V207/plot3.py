import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


def f(x, T):

        return ((2 * const.pi *  const.c ** 2 * const.h) / (x**5 * 10**(-6))) * np.exp( ((const.c * const.h) / (const.k * x*10**(-6) * T))-1 )**(-1)

x = np.linspace(1, 10, 100000)
plt.plot(x, f(x,1000), '-r', label=r'$T=1000 K$')
plt.plot(x, f(x,750), '-b', label=r'$T=750 K$')
plt.plot(x, f(x,500), '-g', label=r'$T=500 K$')

#plt.plot(x, y, label='Kurve')
plt.xlabel(r'$Wellenlänge \:/\: \si{\nano\metre}$')
plt.ylabel(r'$ \frac{dP}{d\lambda} \si{\watt\per\metre} $')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot3.pdf')

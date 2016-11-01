import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

mp = const.physical_constants["proton mass"][0]
mn = const.physical_constants["neutron mass"][0]
ev = const.eV
c = const.c
av = 15.5 * 10**6 * ev * 1/c**2
ao = 16.8 * 10**6 * ev * 1/c**2
ac = 0.715 * 10**6 * ev * 1/c**2
aa = 23 * 10**6 * ev * 1/c**2
ap = 11.3 * 10**6 * ev * 1/c**2

def M(A, Z):
    m1 = Z * mp
    m2 = (A-Z)*mn
    m3 = - av * A
    m4 = ao * A**(2/3)
    m5 = ac * (Z**2)/(A**(1/3))
    m6 = aa * (2*Z-A)**2 * 1/A
    m7 = 0
    if ((A-Z)%2 == 0 and Z%2 ==0):
        m7 = (-1)*ap*A**(-1/2)
        #gg kern
    if ((A-Z)%2 == 1 and Z%2 ==1):
        m7 = (1)*ap*A**(-1/2)
        #uu kern
    mass = m1 + m2 + m3 + m4 + m5 + m6 + m7
    return mass

def M_ungerade(A, Z):
    m1 = Z * mp
    m2 = (A-Z)*mn
    m3 = - av * A
    m4 = ao * A**(2/3)
    m5 = ac * (Z**2)/(A**(1/3))
    m6 = aa * (2*Z-A)**2 * 1/A
    m7 = (1)*ap*A**(-1/2)
    mass = m1 + m2 + m3 + m4 + m5 + m6 + m7
    return mass

def M_gerade(A,Z):
    m1 = Z * mp
    m2 = (A-Z)*mn
    m3 = - av * A
    m4 = ao * A**(2/3)
    m5 = ac * (Z**2)/(A**(1/3))
    m6 = aa * (2*Z-A)**2 * 1/A
    m7 = (-1)*ap*A**(-1/2)
    mass = m1 + m2 + m3 + m4 + m5 + m6 + m7
    return mass

for Z in range(90, 99):
    plt.plot(Z, M(239, Z)*c**2 * 10**(-6) * 1/ev, 'go')
plt.xlabel(r'$Z$')
plt.ylabel(r'$M(Z) / \frac{MeV}{c^2}$')
plt.title('Aufgabe 1b - A=239')
plt.savefig('1b_1.pdf')

plt.clf()

for Z in range(42, 53):
    plt.plot(Z, M(109, Z)*c**2 * 10**(-6) * 1/ev, 'go')
plt.xlabel(r'$Z$')
plt.ylabel(r'$M(Z) / \frac{MeV}{c^2}$')
plt.title('Aufgabe 1b - A=109')
plt.savefig('1b_2.pdf')


plt.clf()


x1 = np.linspace(43,55, 1000)
plt.plot(x1, M_gerade(116, x1)*c**2 * 10**(-6) * 1/ev, 'r')

x2 = np.linspace(43,55, 1000)
plt.plot(x2, M_ungerade(116, x2)*c**2 * 10**(-6) * 1/ev, 'b')

for Z in range(43, 55):
    plt.plot(Z, M(116, Z)*c**2 * 10**(-6) * 1/ev, 'go')
plt.xlabel(r'$Z$')
plt.ylabel(r'$M(Z) / \frac{MeV}{c^2}$')
plt.title('Aufgabe 1b - A=116')



plt.savefig('1b_3.pdf')

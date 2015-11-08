import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from table import (
    make_table,
    make_SI,
    write,
)
from uncertainties import ufloat

#################

# Werte und Tabelle
m_w, m_k, U_w, T_w, U_k, T_k, U_m, T_m = np.genfromtxt('daten.txt', unpack=True)
write('build/tabelle.tex', make_table([m_w, m_k, U_w, T_w, U_k, T_k, U_m, T_m], [2, 2, 2, 2, 2, 2, 2, 2]))

cg_mg = 224.84 # Joule pro Kelvin
cw = 4.18 # Joule pro Kelvin pro Gramm

ck_0 = ( ((cw * m_w[0]) + cg_mg) * (T_m[0] - T_w[0]) ) / ( m_k[0] * (T_k[0] - T_m[0]) )
ck_1 = ( ((cw * m_w[1]) + cg_mg) * (T_m[1] - T_w[1]) ) / ( m_k[1] * (T_k[1] - T_m[1]) )
ck_2 = ( ((cw * m_w[2]) + cg_mg) * (T_m[2] - T_w[2]) ) / ( m_k[2] * (T_k[2] - T_m[2]) )
ck_3 = ( ((cw * m_w[3]) + cg_mg) * (T_m[3] - T_w[3]) ) / ( m_k[3] * (T_k[3] - T_m[3]) )
ck_4 = ( ((cw * m_w[4]) + cg_mg) * (T_m[4] - T_w[4]) ) / ( m_k[4] * (T_k[4] - T_m[4]) )

write('build/ck_blei.tex', str("%.2f" % ck_0))
write('build/ck_zinn1.tex', str("%.2f" % ck_1))
write('build/ck_zinn2.tex', str("%.2f" % ck_2))
write('build/ck_zinn3.tex', str("%.2f" % ck_3))
write('build/ck_graphit.tex', str("%.2f" % ck_4))
#Mittel spezifisch
ck_zinnm = (ck_1+ck_2+ck_3)/3
write('build/ck_zinnm.tex', str("%.2f" % ck_zinnm))

ck_zinnqm = (ck_1**2 + ck_2**2 + ck_3**2)/3

ck_zinna = np.sqrt( ck_zinnqm - ck_zinnm**2 )
write('build/ck_zinna.tex', str("%.2f" % ck_zinna))


# Beispielwerte
ck_zinngut = (ck_2+ck_3)/2
write('build/ck_zinngut.tex', str("%.2f" % ck_zinngut))

c = ufloat(24601, 42)

# Molwärmen
write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))
M = np.array([207.2,118.7,12])
a = np.array([29,27,8])
a = a*10**(-6)
k = np.array([42,55,33])
k = k*10**9
V = np.array([M[0]/11.35 ,M[1]/7.28 ,M[2]/2.25])
V = V * 10**(-6)

C_blei = ck_0*M[0] - 9*a[0]**2 * k[0] * V[0]*T_m[0]
write('build/C_blei.tex', str("%.2f" % C_blei))

C_graphit = ck_4*M[2] - 9*a[2]**2 * k[2] * V[2]*T_m[4]
write('build/C_graphit.tex', str("%.2f" % C_graphit))

C_zinn1 = ck_1*M[1] - 9*a[1]**2 * k[1] * V[1]*T_m[1]
write('build/C_zinn1.tex', str("%.2f" % C_zinn1))

C_zinn2 = ck_2*M[1] - 9*a[1]**2 * k[1] * V[1]*T_m[2]
write('build/C_zinn2.tex', str("%.2f" % C_zinn2))

C_zinn3 = ck_3*M[1] - 9*a[1]**2 * k[1] * V[1]*T_m[3]
write('build/C_zinn3.tex', str("%.2f" % C_zinn3))

#Mittel molar
C_zinnm = (C_zinn1+C_zinn2+C_zinn3)/3
C_zinnqm = (C_zinn1**2+C_zinn2**2+C_zinn3**2)/3
C_zinna = np.sqrt(C_zinnqm-C_zinnm**2)

write('build/C_zinnm.tex', str("%.2f" % C_zinnm))
write('build/C_zinna.tex', str("%.2f" % C_zinna))

C_zinngut = (C_zinn2+C_zinn3)/2
write('build/C_zinngut.tex', str("%.2f" % C_zinngut))

C_zinnguta = np.sqrt((C_zinn2**2+C_zinn3**2)/2-C_zinngut**2)
write('build/C_zinnguta.tex', str("%.2f" % C_zinnguta))

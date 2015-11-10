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

#a) Wheatstonebrücke
#Widerstand Wert 14
R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('wheat.wert14.txt', unpack=True)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

Rx = np.array([Rx1, Rx2, Rx3])
write('build/wheat/wheat.R14_1.tex', str(Rx1))
write('build/wheat/wheat.R14_2.tex', str(Rx2))
write('build/wheat/wheat.R14_3.tex', str(Rx3))

#Widerstand Wert 11
R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('wheat.wert11.txt', unpack=True)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

Rx = np.array([Rx1, Rx2, Rx3])
write('build/wheat/wheat.R11_1.tex', str(Rx1))
write('build/wheat/wheat.R11_2.tex', str(Rx2))
write('build/wheat/wheat.R11_3.tex', str(Rx3))


#b) Kapazitätsbrücke
#R/C-Kombination Wert 8
C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('kapa.kombiwert8.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Cx1 = C2[0]*R34[0]
Cx2 = C2[1]*R34[1]
Cx3 = C2[2]*R34[2]

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

write('build/kapa/kapa.C8_1.tex', str(Cx1))
write('build/kapa/kapa.C8_2.tex', str(Cx2))
write('build/kapa/kapa.C8_3.tex', str(Cx3))
write('build/kapa/kapa.R8_1.tex', str(Rx1))
write('build/kapa/kapa.R8_2.tex', str(Rx2))
write('build/kapa/kapa.R8_3.tex', str(Rx3))

#Kondensator Wert 3
C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('kapa.wert3.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Cx1 = C2[0]*R34[0]
Cx2 = C2[1]*R34[1]
Cx3 = C2[2]*R34[2]

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

write('build/kapa/kapa.C3_1.tex', str(Cx1))
write('build/kapa/kapa.C3_2.tex', str(Cx2))
write('build/kapa/kapa.C3_3.tex', str(Cx3))
write('build/kapa/kapa.R3_1.tex', str(Rx1))
write('build/kapa/kapa.R3_2.tex', str(Rx2))
write('build/kapa/kapa.R3_3.tex', str(Rx3))

#Kondensator Wert 1
C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('kapa.wert1.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Cx1 = C2[0]*R34[0]
Cx2 = C2[1]*R34[1]
Cx3 = C2[2]*R34[2]

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

write('build/kapa/kapa.C1_1.tex', str(Cx1))
write('build/kapa/kapa.C1_2.tex', str(Cx2))
write('build/kapa/kapa.C1_3.tex', str(Cx3))
write('build/kapa/kapa.R1_1.tex', str(Rx1))
write('build/kapa/kapa.R1_2.tex', str(Rx2))
write('build/kapa/kapa.R1_3.tex', str(Rx3))


#c) Induktivitätsbrücke
#L/R-Kombination Wert 19
L_2, L_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('indu.kombiwert19.txt', unpack=True)
L_2 = L_2*10**(-3)
L_2er = L_2er*10**(-3)
L2  = unp.uarray(L_2, L_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)

Lx1 = L2[0]*R34[0]
Lx2 = L2[1]*R34[1]

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]

write('build/indu/indu.L19_1.tex', str(Lx1))
write('build/indu/indu.L19_2.tex', str(Lx2))
write('build/indu/indu.R19_1.tex', str(Rx1))
write('build/indu/indu.R19_2.tex', str(Rx2))


#d) Maxwell-Brücke
#L/R-Kombination Wert 19
C_4, C_4er, R_2, R_2er, R_3, R_3er, R_4, R_34, R_34er = np.genfromtxt('max.kombiwert19.txt', unpack=True)
C_4 = C_4*10**(-9)
C_4er = C_4er*10**(-9)
C4  = unp.uarray(C_4, C_4er)
R2  = unp.uarray(R_2, R_2er)
R3  = unp.uarray(R_3, R_3er)
R34 = unp.uarray(R_34, R_34er)

Lx1 = C4[0]*R3[0]*R2[0]
Lx2 = C4[1]*R3[1]*R2[1]
Lx3 = C4[2]*R3[2]*R2[2]

Rx1 = R2[0]*R34[0]
Rx2 = R2[1]*R34[1]
Rx3 = R2[2]*R34[2]

write('build/max/max.L19_1.tex', str(Lx1))
write('build/max/max.L19_2.tex', str(Lx2))
write('build/max/max.L19_3.tex', str(Lx3))
write('build/max/max.R19_1.tex', str(Rx1))
write('build/max/max.R19_2.tex', str(Rx2))
write('build/max/max.R19_3.tex', str(Rx3))

## Beispielplot
#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
#
#
## Beispieltabelle
#a = np.linspace(1, 10, 10)
#b = np.linspace(11, 20, 10)
#write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen
#
#
## Beispielwerte
#
#
#c = ufloat(24601, 42)
#write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

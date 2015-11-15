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
write('build/wheat1tabelle.tex', make_table([R2, R34], [1,1 , 1, 1]))
Rx = R2*R34
Rx_mean=np.mean(Rx)


write('build/wheat/wheat.R14_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/wheat/wheat.R14_2.tex', make_SI(Rx[1], r'\ohm', figures=1 ))
write('build/wheat/wheat.R14_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/wheat/wheat.R14m.tex', make_SI(Rx_mean, r'\ohm', figures=1))
#Widerstand Wert 11
R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('wheat.wert11.txt', unpack=True)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)
write('build/wheat2tabelle.tex', make_table([R2, R34], [1, 1, 1, 1]))
Rx = R2 * R34
Rx_mean=np.mean(Rx)

write('build/wheat/wheat.R11_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/wheat/wheat.R11_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/wheat/wheat.R11_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/wheat/wheat.R11m.tex', make_SI(Rx_mean, r'\ohm', figures=1))

#b) Kapazitätsbrücke
#R/C-Kombination Wert 8
C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('kapa.kombiwert8.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)
write('build/kapa1tabelle.tex', make_table([C2*10**(9), R2, 1/R34], [1, 1, 1, 1, 1, 1]))
Cx = C2*1/R34
Cx_mean=np.mean(Cx)

Rx = R2*R34
Rx_mean=np.mean(Rx)

write('build/kapa/kapa.C8_1.tex', make_SI(Cx[0]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C8_2.tex', make_SI(Cx[1]*10**9., r'\nano\farad', figures=1))
write('build/kapa/kapa.C8_3.tex', make_SI(Cx[2]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C8m.tex', make_SI(Cx_mean*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.R8_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/kapa/kapa.R8_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/kapa/kapa.R8_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/kapa/kapa.R8m.tex', make_SI(Rx_mean, r'\ohm', figures=1))

#Kondensator Wert 3
C_2, C_2er, R_2, R_2er, R_3, R_4, baz, bar = np.genfromtxt('kapa.wert3.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)
#Hab den Mist aufgegeben: Diese sch**** Tabelle wird jetzt von Hand erstellt. Punkt!
#write('build/kapa2tabelle.tex', make_table([C2*10**(9), R2, baz, bar], [1, 1, 1, 1, 1, 1]))
Cx = C2*1/R34
Cx_mean=np.mean(Cx)

Rx = R2*R34
Rx_mean = np.mean(Rx)

write('build/kapa/kapa.C3_1.tex', make_SI(Cx[0]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C3_2.tex', make_SI(Cx[1]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C3_3.tex', make_SI(Cx[2]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C3m.tex', make_SI(Cx_mean*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.R3_1.tex', make_SI(Rx[0], r'\ohm', figures=1)) # siunitx scheint Probleme mit 0.0 +- 0 zu haben? -> Gelöst mit dauernden workaround in table.py!
write('build/kapa/kapa.R3_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/kapa/kapa.R3_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/kapa/kapa.R3m.tex', make_SI(Rx_mean, r'\ohm', figures=1))

#Kondensator Wert 1
C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('kapa.wert1.txt', unpack=True)
C_2 = C_2*10**(-9)
C_2er = C_2er*10**(-9)
C2  = unp.uarray(C_2, C_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)
write('build/kapa3tabelle.tex', make_table([C2*10**(9), R2, 1/R34], [1, 1, 1, 1, 1, 1]))
Cx = C2*(1/R34)
Cx_mean=np.mean(Cx)

Rx = R2*R34
Rx_mean=np.mean(Rx)

write('build/kapa/kapa.C1_1.tex', make_SI(Cx[0]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C1_2.tex', make_SI(Cx[1]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C1_3.tex', make_SI(Cx[2]*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.C1m.tex', make_SI(Cx_mean*10**9, r'\nano\farad', figures=1))
write('build/kapa/kapa.R1_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/kapa/kapa.R1_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/kapa/kapa.R1_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/kapa/kapa.R1m.tex', make_SI(Rx_mean, r'\ohm', figures=1))


#c) Induktivitätsbrücke
#L/R-Kombination Wert 19
L_2, L_2er, R_2, R_2er, R_3, R_4, R_34, R_34er = np.genfromtxt('indu.kombiwert19.txt', unpack=True)
L_2 = L_2*10**(-3)
L_2er = L_2er*10**(-3)
L2  = unp.uarray(L_2, L_2er)
R2  = unp.uarray(R_2, R_2er)
R34 = unp.uarray(R_34, R_34er)
write('build/indu1tabelle.tex', make_table([L2*10**(3), R2, R34], [1, 1, 1, 1, 1, 1]))
Lx = L2*R34
Lx_mean = np.mean(Lx)

Rx = R2*R34
Rx_mean = np.mean(Rx)

write('build/indu/indu.L19_1.tex', make_SI(Lx[0]*10**3 , r'\milli\henry', figures=1))
write('build/indu/indu.L19_2.tex', make_SI(Lx[1]*10**3 , r'\milli\henry', figures=1))
write('build/indu/indu.L19m.tex', make_SI(Lx_mean*10**3 , r'\milli\henry', figures=1))
write('build/indu/indu.R19_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/indu/indu.R19_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/indu/indu.R19m.tex', make_SI(Rx_mean, r'\ohm', figures=1))

#d) Maxwell-Brücke
#L/R-Kombination Wert 19
C_4, C_4er, R_2, R_2er, R_3, R_3er, R_4, R_4er = np.genfromtxt('max.kombiwert19.txt', unpack=True)
C_4 = C_4*10**(-9)
C_4er = C_4er*10**(-9)
C4  = unp.uarray(C_4, C_4er)
R2  = unp.uarray(R_2, R_2er)
R3  = unp.uarray(R_3, R_3er)
R4  = unp.uarray(R_4, R_4er)
R34 = R3/R4
write('build/max1tabelle.tex', make_table([C4*10**(9), R4, R3, R2, R34], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
Lx = C4*R3*R2
Lx_mean = np.mean(Lx)

Rx = R2*R34
Rx_mean = np.mean(Rx)

write('build/max/max.L19_1.tex', make_SI(Lx[0]*10**3 , r'\milli\henry', figures=1))
write('build/max/max.L19_2.tex', make_SI(Lx[1]*10**3 , r'\milli\henry', figures=1))
write('build/max/max.L19_3.tex', make_SI(Lx[2]*10**3 , r'\milli\henry', figures=1))
write('build/max/max.L19m.tex', make_SI(Lx_mean*10**3 , r'\milli\henry', figures=1))
write('build/max/max.R19_1.tex', make_SI(Rx[0], r'\ohm', figures=1))
write('build/max/max.R19_2.tex', make_SI(Rx[1], r'\ohm', figures=1))
write('build/max/max.R19_3.tex', make_SI(Rx[2], r'\ohm', figures=1))
write('build/max/max.R19m.tex', make_SI(Rx_mean, r'\ohm', figures=1))

#e)
v, U = np.genfromtxt('frequenzmessung.txt', unpack=True)

vtab1, vtab2 = np.array_split(v, 2)
Utab1, Utab2 = np.array_split(U, 2)
write('build/frequenztabelle.tex', make_table([vtab1, Utab1, vtab2, Utab2], [0, 4, 0, 4]))

Us = 8.16
v0 = 1125
U = U/Us
v = v/v0
plt.plot(v, U,'xr', label=r'$\text{Messwerte} U_{Br} \ /\  U_s$')
plt.xscale('log')
plt.ylim(0,0.4)

v= v*v0
v0 = 1498.07 #errechneter Wert mit C_3 und R
v = v/v0
# fitten
from scipy.optimize import curve_fit

def f(v):
    return np.sqrt((1/9)*(v**2-1)**2/((1-v**2)**2+9*v**2))
x_plot = np.linspace(0.01, 100, 1000000)
plt.plot(x_plot, f(x_plot), 'r-', label=r'\text{Theoriekurve} $U_{Br} \ /\  U_s$', linewidth=0.5)
plt.savefig('build/plot.pdf')
plt.ylabel(r'$U_{Br} \ /\  U_S$')
plt.xlabel(r'$\nu \ /\  \nu_0$')
plt.legend(loc='best')
#f)

U_br = U[11] #Brückenspannung bei Minimum
Br_Bs = np.sqrt(f(2)) #Br/Bs Faktor für omega = 2, da omega = w/w0 = v/v0 = 2*v0/v0 = 2
U2 = U_br / Br_Bs
k = U2/Us
write('build/klirrfaktor.tex', str(k))












## Beispielplot
#x = np.linspace(0, 10, 1000)
#y = x ** np.sin(x)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
#
#
## Beispieltabelle
a = np.linspace(1, 10, 10)
b = np.linspace(11, 20, 10)
write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen
#
#
## Beispielwerte
#
#
#c = ufloat(24601, 42)
#write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

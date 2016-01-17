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


# Bestimmung der Winkelrichtgröße d
r, phi, f = np.genfromtxt('build/statisch.txt', unpack=True)
phi = np.array([np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi])
d = f*r/phi
rd = np.mean(d)
dd = np.std(d)
d_0 = ufloat(rd,dd)
write('build/winkelrichtgroesse.tex', make_SI(d_0*10**2, r'\newton\centi\metre', figures=1))



#Bestimmung Eigenträgheitsmoment


from scipy.optimize import curve_fit

a, t = np.genfromtxt('build/dynamisch.txt', unpack=True)

def h(x, m, b):
    return m*x + b
plt.plot(a**2, t**2,'xr', label=r'$\text{Messwerte } T^2$')
parameter, covariance = curve_fit(h, a**2, t**2)
x_plot = np.linspace(0, 0.07, 10000)

plt.plot(x_plot, h(x_plot, parameter[0], parameter[1]), 'b-', label=r'Ausgleichskurve', linewidth=1)
fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

m_fit = ufloat(parameter[0], fehler[0])
b_fit = ufloat(parameter[1], fehler[1])

write('build/m.tex', make_SI(m_fit, r'\second\tothe{2}\per\metre\tothe{2}', figures=1))
write('build/b.tex', make_SI(b_fit, r'\second\tothe{2}', figures=1))
plt.xlabel(r'$a^2 \ /\ m^2$')
plt.ylabel(r'$T^2 \ /\ s^2$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08) # Ich werde diese Zeile nie wieder vergessen, oh Großmeister Marco, mein Führer! <- So soll es sein!m
plt.ylim(unp.nominal_values(b_fit),60)
plt.xlim(0,0.07)
plt.savefig('build/plot.pdf')

plt.clf()

tr_d = b_fit*d_0/(4*np.pi**2) #HIER WIRD DAS BERECHNET
write('build/eigentraegheit.tex', make_SI(tr_d*10**3, r'\gram\metre\tothe{2}', figures=1)) # Oh je: Das hier ist jetzt in Gramm MeterQuadrat
z = tr_d
# Zylinder

m, d, h, t = np.genfromtxt('build/zylinder.txt', unpack=True)

rm_z  = np.mean(m)
m_z  = ufloat(rm_z , 0)
rd_z = np.mean(d)
dd_z = np.std(d)
d_z  = ufloat(rd_z ,dd_z)
rt_z = np.mean(t)
dt_z = np.std(t)
t_z  = ufloat(rt_z ,dt_z)
write('build/m_zylinder.tex', make_SI(rm_z*10**3, r'\gram', figures=1))
write('build/r_zylinder.tex', make_SI(d_z*10**2, r'\centi\metre', figures=1))
write('build/t_zylinder.tex', make_SI(t_z, r'\second', figures=1))
tr_z_theorie = (m_z*d_z**2)/2
write('build/traegheit_zylinder_theorie.tex', make_SI(tr_z_theorie*10**3, r'\gram\metre\tothe{2}', figures=1)) # hier ist Einheit jut
tr_z = (((t_z/(2*np.pi))**2)*d_0)
tr_z1 = tr_z #- z#*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT -> lulz
write('build/traegheit_zylinder.tex', make_SI(tr_z1*10**3, r'\gram\metre\tothe{2}', figures=1))

write('build/abweichung_zylinder.tex', str((("%.1f" % unp.nominal_values(abs((tr_z1-tr_z_theorie))/tr_z_theorie*100))))) # good job


# Kugel

m, r, t = np.genfromtxt('build/kugel.txt', unpack=True)

rm_k = np.mean(m)
dm_k = np.std(m)
m_k  = ufloat(rm_k , 0)
rr_k = np.mean(r)
dr_k = np.std(r)
r_k  = ufloat(rr_k ,dr_k)
rt_k = np.mean(t)
dt_k = np.std(t)
t_k  = ufloat(rt_k ,dt_k)
write('build/m_kugel.tex', make_SI(rm_k*10**3, r'\gram', figures=1))
write('build/r_kugel.tex', make_SI(r_k*10**2, r'\centi\metre', figures=1))
write('build/t_kugel.tex', make_SI(t_k, r'\second', figures=1))
tr_k_theorie = 2*(m_k*r_k**2)/5
write('build/traegheit_kugel_theorie.tex', make_SI(tr_k_theorie*10**7, r'\gram\centi\metre\tothe{2}', figures=1))
tr_k = (((t_k/(2*np.pi))**2)*d_0)
tr_k1 = tr_k - z*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT
write('build/traegheit_kugel.tex', make_SI(tr_k1*10**7, r'\gram\centi\metre\tothe{2}', figures=1))

write('build/abweichung_kugel.tex', str(("%.2f" % unp.nominal_values(abs((tr_k1-tr_k_theorie))/tr_k_theorie*100))))

# Mensch

m_m, h_bein_1, h_bein_2, d_beine, a_beine, h_kopf, d_kopf, h_torso, d_torso, l_arme, d_arme, a_arme, t2, t1 = np.genfromtxt('build/mensch.txt', unpack=True)


rm_m      = np.mean(m_m)
rh_bein_1 = np.mean(h_bein_1)
rh_bein_2 = np.mean(h_bein_2)
rd_beine  = np.mean(d_beine)
ra_beine  = np.mean(a_beine)
rh_kopf   = np.mean(h_kopf)
rd_kopf   = np.mean(d_kopf)
rh_torso  = np.mean(h_torso)
rd_torso  = np.mean(d_torso)
rl_arme   = np.mean(l_arme)
rd_arme   = np.mean(d_arme)
ra_arme   = np.mean(a_arme)
rt1       = np.mean(t1)
rt2       = np.mean(t2)

dm_m      = np.std(m_m)
dh_bein_1 = np.std(h_bein_1)
dh_bein_2 = np.std(h_bein_2)
dd_beine  = np.std(d_beine)
da_beine  = np.std(a_beine)
dh_kopf   = np.std(h_kopf)
dd_kopf   = np.std(d_kopf)
dh_torso  = np.std(h_torso)
dd_torso  = np.std(d_torso)
dl_arme   = np.std(l_arme)
dd_arme   = np.std(d_arme)
da_arme   = np.std(a_arme)
dt1       = np.std(t1)
dt2       = np.std(t2)

m_m      = ufloat(rm_m, dm_m)
h_bein_1 = ufloat(rh_bein_1, dh_bein_1)
h_bein_2 = ufloat(rh_bein_2, dh_bein_2)
d_beine  = ufloat(rd_beine, dd_beine)
a_beine  = ufloat(ra_beine, da_beine)
h_kopf   = ufloat(rh_kopf, dh_kopf)
d_kopf   = ufloat(rd_kopf, dd_kopf)
h_torso  = ufloat(rh_torso, dh_torso)
d_torso  = ufloat(rd_torso, dd_torso)
l_arme   = ufloat(rl_arme, dl_arme)
d_arme   = ufloat(rd_arme, dd_arme)
a_arme   = ufloat(ra_arme, da_arme)
t1       = ufloat(rt1, dt1)
t2       = ufloat(rt2, dt2)

write('build/laenge_b_1.tex', make_SI((h_bein_1)*10**2, r'\centi\metre', figures=1))
write('build/radius_b.tex', make_SI((d_beine)*10**2, r'\centi\metre', figures=1))
write('build/laenge_b_2.tex', make_SI((h_bein_2)*10**2, r'\centi\metre', figures=1))
write('build/abstand_b.tex', make_SI((a_beine)*10**2, r'\centi\metre', figures=1))
write('build/laenge_t.tex', make_SI((h_torso)*10**2, r'\centi\metre', figures=1))
write('build/radius_t.tex', make_SI((d_torso)*10**2, r'\centi\metre', figures=1))
write('build/laenge_k.tex', make_SI((h_kopf)*10**2, r'\centi\metre', figures=1))
write('build/radius_k.tex', make_SI((d_kopf)*10**2, r'\centi\metre', figures=1))
write('build/laenge_a.tex', make_SI((l_arme)*10**2, r'\centi\metre', figures=1))
write('build/radius_a.tex', make_SI((d_arme)*10**2, r'\centi\metre', figures=1))
write('build/abstand_a.tex', make_SI((a_arme)*10**2, r'\centi\metre', figures=1))
write('build/masse.tex', make_SI((m_m)*10**3, r'\gram', figures=1))
write('build/t1.tex', make_SI((t1), r'\second', figures=1))
write('build/t2.tex', make_SI((t2), r'\second', figures=1))









V_arm_1  = np.pi * l_arme * d_arme**2
V_arm_2  = np.pi * l_arme * d_arme**2
V_bein_1 = np.pi * h_bein_1 * d_beine**2
V_bein_2 = np.pi * h_bein_2 * d_beine**2
V_kopf   = np.pi * h_kopf * d_kopf**2
V_torso  = np.pi * h_torso * d_torso**2


V = V_arm_1 + V_arm_2 + V_bein_1 + V_bein_2 + V_kopf + V_torso
write('build/volumen_mensch.tex', make_SI(V*10**6, r'\centi\metre\tothe{3}', figures=1))
dichte = m_m/V
write('build/dichte_mensch.tex', make_SI(dichte, r'\kilo\gram\per\metre\tothe{3}', figures=1))

m_arm_1  = V_arm_1 * dichte
m_arm_2  = V_arm_2 * dichte
m_bein_1 = V_bein_1 * dichte
m_bein_2 = V_bein_2 * dichte
m_kopf   = V_kopf * dichte
m_torso  = V_torso * dichte

I_pose_1_theorie = (1/2 * m_bein_1 * d_beine**2 + (a_beine + d_beine)**2 * m_bein_1) + (1/2 * m_bein_2 * d_beine**2 + (a_beine + d_beine)**2 * m_bein_2) + (1/2 * m_torso * d_torso**2) + (1/2 * m_kopf * d_kopf**2) + (m_arm_1 * (1/4 * d_arme**2 + 1/12 * l_arme**2) + (a_arme + l_arme/2)**2 * m_arm_1) + (m_arm_2 * (1/4 * d_arme**2 + 1/12 * l_arme**2) + (a_arme + l_arme/2)**2 * m_arm_2)
write('build/traegheit_mensch_pose_1_theorie.tex', make_SI(I_pose_1_theorie*10**7, r'\gram\centi\metre\tothe{2}', figures=1))

I_pose_1 = (((t1/(2*np.pi))**2)*d_0)
I_pose_1 = I_pose_1 - z*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT
write('build/traegheit_mensch_pose_1.tex', make_SI((I_pose_1)*10**7, r'\gram\centi\metre\tothe{2}', figures=1))

write('build/abweichung_pose_1.tex', str(("%.2f" % unp.nominal_values(abs((I_pose_1-I_pose_1_theorie))/I_pose_1_theorie*100))))

I_pose_2_theorie = (1/2 * m_bein_1 * d_beine**2 + (a_beine + d_beine)**2 * m_bein_1) + (1/2 * m_bein_2 * d_beine**2 + (a_beine + d_beine)**2 * m_bein_2) + (1/2 * m_torso * d_torso**2) + (1/2 * m_kopf * d_kopf**2) + (1/2 * (m_arm_1 * d_arme**2) + (a_arme + d_arme/2)**2 * m_arm_1) + (1/2 * (m_arm_2 * d_arme**2) + (a_arme + d_arme/2)**2 * m_arm_2)
write('build/traegheit_mensch_pose_2_theorie.tex', make_SI(I_pose_2_theorie*10**7, r'\gram\centi\metre\tothe{2}', figures=1))

I_pose_2 = (((t2/(2*np.pi))**2)*d_0)
I_pose_2 = I_pose_2 - z*10**(-3)    # WIESO ER DAS AUCH IMMER MIT 1000 MULTIPLIZIERT HAT
write('build/traegheit_mensch_pose_2.tex', make_SI((I_pose_2)*10**7, r'\gram\centi\metre\tothe{2}', figures=1))

write('build/abweichung_pose_2.tex', str(("%.2f" % unp.nominal_values(abs((I_pose_2-I_pose_2_theorie))/I_pose_2_theorie*100))))


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
#c = ufloat(0, 0)
#write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

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

# a) statische Methode

r   = np.array([5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11])
phi = np.array([90, 180, 270, 90, 180, 270, 90, 180, 270, 90, 180, 270])
phi1 = np.array([np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi])
f   = np.array([0.74, 1.4, 2.1, 0.52, 1, 1.5, 0.42, 0.78, 1.18, 0.32, 0.64, 0.96])

phi_b = np.array([np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi, np.pi/2, np.pi, 3/2*np.pi])

r = r*10**(-2)
d = f*r/phi_b

write('build/statischtabelle.tex', make_table([r*10**2, phi, f, d*10**2], [2, 0, 2, 2])) #Einheiten ok: Hier wird d in Ncm berechnet und angegeben <- jetzt nimmer



np.savetxt('build/statisch.txt', np.column_stack([r, phi, f]), header="r [m], phi [grad], F [N]")

# b) dynamische Methode

#m1 = (221.78+-0.01)g
#m1 = (223.46+-0.01)g

d = np.array([6, 7.7, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5])
t = np.array([13.63, 15.61, 19.64, 22.12, 24.41, 27.09, 29.58, 32.23, 34.86, 37.55])
t = t/5

write('build/dynamischtabelle.tex', make_table([d, t], [2, 2]))

d = d*10**(-2)

np.savetxt('build/dynamisch.txt', np.column_stack([d, t]), header="d [m], T [s]")

# c)
# Zylinder

m_1 = np.array([1005.8, 1005.8, 1005.8, 1005.8, 1005.8])
d_1 = np.array([8, 8.01, 8.015, 8.01, 8.01])
h_1 = np.array([13.995, 14.005, 14.005, 14.01, 13.99])
t_1 = np.array([5.97, 6.01, 5.96, 5.96, 6])
t_1 = t_1/5



m_1 = m_1*10**(-3)
h_1 = h_1*10**(-2)
d_1 = d_1*10**(-2)/2 #huh

write('build/zylindertabelle.tex', make_table([m_1, d_1*10**2, h_1*10, t_1], [4, 3, 4, 3]))
np.savetxt('build/zylinder.txt', np.column_stack([m_1, d_1, h_1, t_1]), header="m [kg], r [m], h [m], T [s]")

# Kugel

m_2 = np.array([812.5, 812.5, 812.5, 812.5, 812.5])
d_2 = np.array([13.755, 13.76, 13.775, 13.74, 13.74])
t_2 = np.array([8.35, 8.56, 8.49, 8.40, 8.59])
t_2 = t_2/5

write('build/kugeltabelle.tex', make_table([m_2*10, d_2*10, t_2], [1, 2, 2]))

m_2 = m_2*10**(-3)
d_2 = d_2*10**(-2)/2

write('build/kugeltabelle.tex', make_table([m_2*10, d_2*2*10, t_2], [3, 4, 2]))
np.savetxt('build/kugel.txt', np.column_stack([m_2, d_2, t_2]), header="m [kg], r [m], T [s]")

# d) Mensch
m        = np.array([162.05, 162.02, 162.03, 162.04, 162.05])
h_bein_1 = np.array([15.105, 14.905, 14.83, 14.865, 14.925])
h_bein_2 = np.array([15.585, 15.49, 15.505, 15.48, 15.475])
d_beine  = np.array([1.705, 2.12, 1.09, 1.165, 1.72])
a_beine  = np.array([2.185, 2.365, 2.365, 2.385, 2.33])
h_kopf   = np.array([5.55, 5.55, 5.66, 5.835, 5.83])
d_kopf   = np.array([1.585, 3.085, 3.08, 2.64, 2.135])
h_torso  = np.array([9.805, 9.765, 9.82, 9.775, 9.885])
d_torso  = np.array([4.18, 3.95, 2.59, 4.57, 3.68])
l_arme   = np.array([14.08, 13.81, 13.81, 13.9, 13.895])
d_arme   = np.array([1.665, 1.62, 1.43, 1.55, 1.04])
a_arme   = np.array([5.87, 5.835, 5.75, 5.855, 5.925])
t1       = np.array([2.23, 2.16, 2.18, 2.13, 2.16])
t2       = np.array([3.32, 3.38, 3.27, 3.32, 3.24])

t1 = t1/5
t2 = t2/5

#write('build/menschtabelle1.tex', make_table([m*10, h_bein_1*10, h_bein_2*10, d_beine*10**2, a_beine*10**2, h_kopf*10**2], [1, 2, 2, 1, 2, 2]))
#
#write('build/menschtabelle2.tex', make_table([d_kopf*10**2, h_torso*10**2, d_torso*10**2, l_arme*10, d_arme*10**2, a_arme*10**2], [2, 1, 1, 2, 1, 1]))
#
#write('build/menschtabelle3.tex', make_table([t1*10, t2*10], [1, 1]))


m        = m*10**(-3)
h_bein_1 = h_bein_1*10**(-2)
h_bein_2 = h_bein_2*10**(-2)
d_beine  = d_beine*10**(-2)/2
a_beine  = a_beine*10**(-2)/2
h_kopf   = h_kopf*10**(-2)
d_kopf   = d_kopf*10**(-2)/2
h_torso  = h_torso*10**(-2)
d_torso  = d_torso*10**(-2)/2
l_arme   = l_arme*10**(-2)
d_arme   = d_arme*10**(-2)/2
a_arme   = a_arme*10**(-2)/2

write('build/menschtabelle1.tex', make_table([m*10, h_bein_1*10, h_bein_2*10, d_beine*2*10**2, a_beine*10**2], [4, 4, 4, 3, 3]))

write('build/menschtabelle2.tex', make_table([h_kopf*10**2, d_kopf*2*10**2, h_torso*10**2, d_torso*10*2**2, l_arme*10], [3, 3, 3, 2, 4]))

write('build/menschtabelle3.tex', make_table([d_arme*2*10**3, a_arme*2*10**2, t1*10, t2*10], [2, 3, 1, 1]))

#write('build/menschtabelle1.tex', make_table([m, h_bein_1, h_bein_2, d_beine*2, a_beine*2, h_kopf, d_kopf*2], [5, 5, 5, 5, 5, 5, 5]))
#
#write('build/menschtabelle2.tex', make_table([h_torso, d_torso*2, l_arme, d_arme*2, a_arme*2, t1, t2], [5, 4, 5, 5, 5, 2, 2]))

np.savetxt('build/mensch.txt', np.column_stack([m, h_bein_1, h_bein_2, d_beine, a_beine, h_kopf, d_kopf, h_torso, d_torso, l_arme, d_arme, a_arme, t1, t2]), header="alles in SI-Einheiten")

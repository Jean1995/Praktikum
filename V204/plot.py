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

### DATEN HIER ###

#Temperaturen in °C nach 700 Sekunden für T1, T4, T5, T8
T_600  = np.array([35.70, 34.73, 37.26, 31.14])




#Temperaturen in °C T1, T2 nach 10 s, 50 s, 200s, 300s, 500s
T1  = np.array([21.09, 23.12, 29.05, 31.11, 33.81])
T2  = np.array([21.40, 26.05, 31.24, 32.98, 35.47])
t_T = np.array([10, 50, 200, 300, 500])

#Maxima: Für T1 und T2: A1/A2 Amplitude am i-ten Maximum, t1/t2 Zeit an dem i-tes Maximum erreicht wird
A1 = np.array([35.7, 39.8, 43.6, 46.8, 49.5, 51.6, 53.6, 55.6, 56.7, 58.3])
A1_min = np.array([27.3, 33.0 , 35.3, 38.9, 41.8, 44.5, 46.7, 48.5, 50.0, 51.3])
A2 = np.array([31.3, 34.9, 38.3, 41.3, 44.0, 46.3, 48.0, 49.9, 51.3, 52.7])
A2_min = np.array([27.3, 31.0, 34.6, 37.8, 40.8, 43.1, 45.3, 47.1, 49.0, 50.1])
t1 = np.array([44, 123, 205, 238, 326, 444, 523, 605, 683, 739])
t2 = np.array([62, 146, 223, 297, 383, 460, 539, 618,  697, 778])
write('build/dyn1.tex', make_table([A1-A1_min, A2-A2_min, t1, t2], [1, 1, 0, 0]))   # [4,2] = Nachkommastellen


#Maxima: Für T5 und T6: A5/A6 Amplitude am i-ten Maximum, t5/t6 Zeit an dem i-tes Maximum erreicht wird
A5 = np.array([35.5, 40.1, 43.8, 47.1, 49.6, 51.9, 53.5, 60.4, 66.6, 68.0 ])
A5_min = np.array([ 27.3, 32.8, 36.2, 39.7, 42.8, 45.1, 47.3, 49.3, 50.4, 51.8 ])
A6 = np.array([33.5, 38.2, 41.9, 45.3, 47.7, 50.0 ,51.9, 53.5, 54.9, 58.0 ])
A6_min = np.array([ 27.3, 33.2, 36.9, 40.4, 43.7, 45.9, 48.3, 49.7, 51.3, 52.8 ])
t5 = np.array([41, 123, 202, 285, 362, 441, 518, 604, 683, 762 ])
t6 = np.array([55,132, 212, 292, 374, 451, 530, 609, 689, 769 ])
write('build/dyn2.tex', make_table([A5-A5_min, A6-A6_min, t5, t6], [1, 1, 0, 0]))   # [4,2] = Nachkommastellen


#Maxima: Für T7 und T8: A7/A8 Amplitude am i-ten Maximum, t7/t8 Zeit an dem i-tes Maximum erreicht wird
A7 = np.array([ 42.0, 46.5, 50.8,  54.2, 56.9, 59.6, 60.6   ])
A7_min = np.array( [ 30.0, 37.1, 41.1, 44.3, 45.5, 50.0, 51.4   ]  )
A8 = np.array([31.7,35.3,  38.4, 41.1, 43.4 , 45.4, 46.9  ])
A8_min = np.array( [ 31.4, 35.2,  38.2, 40.8, 43.2, 45.1, 46.4 ] )
t7 = np.array([112, 312, 512,  716, 916, 1120, 1320 ])
t8 = np.array([200, 400,596, 788, 984, 1180,1380 ])
write('build/dyn3.tex', make_table([A7-A7_min, A8-A8_min, t7, t8], [1, 1, 0, 0]))   # [4,2] = Nachkommastellen


### DATEN ENDE ###


# Wärmestom

k_messing = 120 # Quelle Wikipedia, such mal ne bessere.. ._.
A = 0.012 * 0.004 # ist das das richtige Messing?

delta_T = T2-T1
delta_x = 0.03
W = -k_messing * A * delta_T/delta_x # nach Gl(1) Skript
write('build/wstrom.tex', make_table([t_T, W], [0, 2]))
write('build/a_ws.tex', str("%.5f" % A))
write('build/x_ws.tex', str("%.2f" % delta_x))
write('build/k_ws.tex', str("%.0f" % k_messing))

# Fucking Angström

#Messing

deltaA_1 = (A1-A1_min)/(A2-A2_min) # um die Amplituden zu bekommen BAH
deltaT_1 = t2-t1
rho_1 = 8520
c_1 = 385

kappa_1_a = (rho_1 * c_1 * delta_x**2  ) / (2* deltaT_1 * np.log(deltaA_1)  )
write('build/kappa_1_tab.tex', make_table([kappa_1_a], [2]))
kappa_1 = ufloat(np.mean(kappa_1_a), np.std(kappa_1_a))
write('build/kappa_1.tex', make_SI(kappa_1, r'\watt\per\metre\per\kelvin' ))


test_deltaA_1 = ufloat(  np.mean( (A1-A1_min)/(A2-A2_min) ) , np.std( (A1-A1_min)/(A2-A2_min) )   )
test_deltaT_1 = ufloat( np.mean( t2-t1 ), np.std( t2-t1) )
kappa_1_a_test = (rho_1 * c_1 * delta_x**2  ) / (2* test_deltaT_1 * unp.log(test_deltaA_1)  )
write('build/test_kappa_1.tex', make_SI(kappa_1_a_test, r'\watt\per\metre\per\kelvin' ))

#Alu

deltaA_2 = (A5-A5_min)/(A6-A6_min) # um die Amplituden zu bekommen BAH
deltaT_2 = t6-t5
rho_2 = 2800
c_2 = 830

kappa_2_a = (rho_2 * c_2 * delta_x**2  ) / (2* deltaT_2 * np.log(deltaA_2)  )
write('build/kappa_2_tab.tex', make_table([kappa_2_a], [2]))
kappa_2 = ufloat(np.mean(kappa_2_a), np.std(kappa_2_a))
write('build/kappa_2.tex', make_SI(kappa_2, r'\watt\per\metre\per\kelvin' ))

#Edelstrahl

deltaA_3 = (A7-A7_min)/(A8-A8_min) # um die Amplituden zu bekommen BAH
deltaT_3 = t8-t7
rho_3 = 8000
c_3 = 400

kappa_3_a = (rho_3 * c_3 * delta_x**2  ) / (2* deltaT_3 * np.log(deltaA_3)  )
write('build/kappa_3_tab.tex', make_table([kappa_3_a], [2]))
kappa_3 = ufloat(np.mean(kappa_3_a), np.std(kappa_3_a))
write('build/kappa_3.tex', make_SI(kappa_3, r'\watt\per\metre\per\kelvin' ))


# Beispielplot
x = np.linspace(0, 10, 1000)
y = x ** np.sin(x)
plt.plot(x, y, label='Kurve')
plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')


# Beispieltabelle
a = np.linspace(1, 10, 10)
b = np.linspace(11, 20, 10)
write('build/tabelle.tex', make_table([a, b], [4, 2]))   # [4,2] = Nachkommastellen


# Beispielwerte


c = ufloat(0, 0)
write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

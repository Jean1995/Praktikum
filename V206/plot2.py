import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp

p = np.array([5.5, 7, 8.2 , 9.5, 10.5, 11.5, 12.2])
T = np.array([17.5, 27.0, 35.0, 40.0, 44.0, 47.0, 50.0])

T_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
p_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


T=T+273.2 # celsius in kelvin
p=p*100000 # bar in pascal

plt.plot(1/T, np.log(p),'xb', label=r'Drucktemperaturen')


# fitten
from scipy.optimize import curve_fit

def f(x, m, b):
    return m * x + b

parameter, covariance = curve_fit(f, 1/T, np.log(p))
x_plot = np.linspace(0.0030, 0.0035, 1000)

plt.plot(x_plot, f(x_plot, parameter[0], parameter[1]), 'r-', label=r'Theoriedampfdruckkurve', linewidth=1)

fehler = np.sqrt(np.diag(covariance)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

np.savetxt('ausgleichswerte_dampfdruckkurve.txt', np.column_stack([parameter, fehler]), header="m/b m/b-Fehler")





plt.xlabel(r'$1/T \:/\: [\si{\per \kelvin}]$')
plt.ylabel(r'$ln(p_b) \:/\: [ln(\si{\pascal})]$')

plt.legend(loc='best')
#plt.xlim(0.0030, 0.0035)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot2.pdf')


#Bestimmung Massendurchsatz

R = 8.314 # allgemeine Gaskonstante
m = ufloat(parameter[0], fehler[0])
L = m*R # Verdampfungswärme

N = ufloat(124.77, 1)

v_t1, v_t2, err_v_t1, err_v_t2 = np.genfromtxt('gueteziffern.txt', unpack=True) # Gueteziffer * N ergibt wieder dQ2 / dt

diff = unp.uarray(v_t2, err_v_t2)
diff = diff * N

md = diff/L
np.savetxt('massendurchsatz.txt', np.column_stack([unp.nominal_values(md),  unp.std_devs(md)]), header="md md_error")


#Bestimmung Kompressorleistung

t, T1, Pb, T2, Pa, P , T1_error, Pb_error, T2_error, Pa_error, P_error = np.genfromtxt('daten.txt', unpack=True)

T1=T1+273.2 #celsius in kelvin
T2=T2+273.2
kappa = 1.14 #Adiabatenkoeffizient
R_s = 76.513 #spez. Gaskonstante
Mas = 120.91

Pa = unp.uarray(Pa, Pa_error)
Pb = unp.uarray(Pb, Pb_error)
T2 = unp.uarray(T2, T2_error)
Pa=Pa*100000 #Bar in Pascal
Pb=Pb*100000 #Bar in Pascal
rho7 = (Pa[7])/(T2[7]*R_s)
rho14 = (Pa[14])/(T2[14]*R_s)
rho21 = (Pa[21])/(T2[21]*R_s)
rho28 = (Pa[28])/(T2[28]*R_s)
dichte = np.array([rho7, rho14, rho21, rho28])
np.savetxt('dichten.txt', np.column_stack([unp.nominal_values(dichte),  unp.std_devs(dichte)]), header="dichte dichte_error")

#richtige Dichte yo
R_0 = 5.51
TR_0 = 273.2
PR = 100000

R_7  = (Pa[7] /PR)*(TR_0/T2[7] )*R_0
R_14 = (Pa[14]/PR)*(TR_0/T2[14])*R_0
R_21 = (Pa[21]/PR)*(TR_0/T2[21])*R_0
R_28 = (Pa[28]/PR)*(TR_0/T2[28])*R_0

rdichte = np.array([R_7, R_14, R_21, R_28])
np.savetxt('rdichten.txt', np.column_stack([unp.nominal_values(rdichte),  unp.std_devs(rdichte)]), header="rdichte rdichte_error")

rP_mech_7 = (1/(kappa-1)) * (Pb[7] * (Pa[7]/Pb[7])**(1/kappa)-Pa[7]) * (1/rdichte[0]) * md[0] * Mas * 0.001
rP_mech_14 = (1/(kappa-1)) * (Pb[14] * (Pa[14]/Pb[14])**(1/kappa)-Pa[14]) * (1/rdichte[1]) * md[1] * Mas * 0.001
rP_mech_21 = (1/(kappa-1)) * (Pb[21] * (Pa[21]/Pb[21])**(1/kappa)-Pa[21]) * (1/rdichte[2]) * md[2] * Mas * 0.001
rP_mech_28 = (1/(kappa-1)) * (Pb[28] * (Pa[28]/Pb[28])**(1/kappa)-Pa[28]) * (1/rdichte[3]) * md[3] * Mas * 0.001

rP_mech = np.array([rP_mech_7, rP_mech_14, rP_mech_21, rP_mech_28])

np.savetxt('rp_mech.txt', np.column_stack([unp.nominal_values(rP_mech),  unp.std_devs(rP_mech)]), header="rP_mech rP_mech_error")



P_mech_7 = (1/(kappa-1)) * (Pb[7] * (Pa[7]/Pb[7])**(1/kappa)-Pa[7]) * (1/dichte[0]) * md[0] * Mas * 0.001
P_mech_14 = (1/(kappa-1)) * (Pb[14] * (Pa[14]/Pb[14])**(1/kappa)-Pa[14]) * (1/dichte[1]) * md[1] * Mas * 0.001
P_mech_21 = (1/(kappa-1)) * (Pb[21] * (Pa[21]/Pb[21])**(1/kappa)-Pa[21]) * (1/dichte[2]) * md[2] * Mas * 0.001
P_mech_28 = (1/(kappa-1)) * (Pb[28] * (Pa[28]/Pb[28])**(1/kappa)-Pa[28]) * (1/dichte[3]) * md[3] * Mas * 0.001

P_mech = np.array([P_mech_7, P_mech_14, P_mech_21, P_mech_28])

np.savetxt('p_mech.txt', np.column_stack([unp.nominal_values(P_mech),  unp.std_devs(P_mech)]), header="P_mech P_mech_error")

##################################################### Import system libraries ######################################################
import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update(mpl.rc_params_from_file('meine-matplotlibrc'))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
################################################ Finish importing system libraries #################################################

################################################ Adding subfolder to system's path #################################################
import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
############################################# Finish adding subfolder to system's path #############################################

##################################################### Import custom libraries ######################################################
from curve_fit import ucurve_fit
from table import (
    make_table,
    make_full_table,
    make_composed_table,
    make_SI,
    write,
)
from regression import (
    reg_linear,
    reg_quadratic,
    reg_cubic
)
from error_calculation import(
    MeanError
)
################################################ Finish importing custom libraries #################################################







################################ FREQUENTLY USED CODE ################################
#
########## IMPORT ##########
# t, U, U_err = np.genfromtxt('data.txt', unpack=True)
# t *= 1e-3


########## ERRORS ##########
# R_unc = ufloat(R[0],R[2])
# U = 1e3 * unp.uarray(U, U_err)
# Rx_mean = np.mean(Rx)                 # Mittelwert und syst. Fehler
# Rx_mean_err = MeanError(noms(Rx))     # Fehler des Mittelwertes
#
## Relative Fehler zum späteren Vergleich in der Diskussion
# RelFehler_G = (G_mess - G_lit) / G_lit
# RelFehler_B = (B_mess - B_lit) / B_lit
# write('build/RelFehler_G.tex', make_SI(RelFehler_G*100, r'\percent', figures=1))
# write('build/RelFehler_B.tex', make_SI(RelFehler_B*100, r'\percent', figures=1))


########## CURVE FIT ##########
# def f(t, a, b, c, d):
#     return a * np.sin(b * t + c) + d
#
# params = ucurve_fit(f, t, U, p0=[1, 1e3, 0, 0])   # p0 bezeichnet die Startwerte der zu fittenden Parameter
# params = ucurve_fit(reg_linear, x, y)             # linearer Fit
# params = ucurve_fit(reg_quadratic, x, y)          # quadratischer Fit
# params = ucurve_fit(reg_cubic, x, y)              # kubischer Fit
# a, b = params
# write('build/parameter_a.tex', make_SI(a * 1e-3, r'\kilo\volt', figures=1))       # type in Anz. signifikanter Stellen
# write('build/parameter_b.tex', make_SI(b * 1e-3, r'\kilo\hertz', figures=2))      # type in Anz. signifikanter Stellen


########## PLOTTING ##########
# plt.clf                   # clear actual plot before generating a new one
#
## automatically choosing limits with existing array T1
# t_plot = np.linspace(np.amin(T1), np.amax(T1), 100)
# plt.xlim(t_plot[0]-1/np.size(T1)*(t_plot[-1]-t_plot[0]), t_plot[-1]+1/np.size(T1)*(t_plot[-1]-t_plot[0]))
#
## hard coded limits
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\second}$')
# plt.ylabel(r'$U \:/\: \si{\kilo\volt}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/aufgabenteil_a_plot.pdf')


########## WRITING TABLES ##########
### IF THERE IS ONLY ONE COLUMN IN A TABLE (workaround):
## a=np.array([Wert_d[0]])
## b=np.array([Rx_mean])
## c=np.array([Rx_mean_err])
## d=np.array([Lx_mean*1e3])
## e=np.array([Lx_mean_err*1e3])
#
# write('build/Tabelle_b.tex', make_table([a,b,c,d,e],[0, 1, 0, 1, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
# write('build/Tabelle_b_texformat.tex', make_full_table(
#     'Messdaten Kapazitätsmessbrücke.',
#     'table:A2',
#     'build/Tabelle_b.tex',
#     [1,2,3,4,5],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                               # die Multicolumns sein sollen
#     ['Wert',
#     r'$C_2 \:/\: \si{\nano\farad}$',
#     r'$R_2 \:/\: \si{\ohm}$',
#     r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
#     r'$C_x \:/\: \si{\nano\farad}$']))
#
## Aufsplitten von Tabellen, falls sie zu lang sind
# t1, t2 = np.array_split(t * 1e3, 2)
# U1, U2 = np.array_split(U * 1e-3, 2)
# write('build/loesung-table.tex', make_table([t1, U1, t2, U2], [3, None, 3, None]))  # type in Nachkommastellen
#
## Verschmelzen von Tabellen (nur Rohdaten, Anzahl der Zeilen muss gleich sein)
# write('build/Tabelle_b_composed.tex', make_composed_table(['build/Tabelle_b_teil1.tex','build/Tabelle_b_teil2.tex']))


########## ARRAY FUNCTIONS ##########
# np.arange(2,10)                   # Erzeugt aufwärts zählendes Array von 2 bis 10
# np.zeros(15)                      # Erzeugt Array mit 15 Nullen
# np.ones(15)                       # Erzeugt Array mit 15 Einsen
#
# np.amin(array)                    # Liefert den kleinsten Wert innerhalb eines Arrays
# np.argmin(array)                  # Gibt mir den Index des Minimums eines Arrays zurück
# np.amax(array)                    # Liefert den größten Wert innerhalb eines Arrays
# np.argmax(array)                  # Gibt mir den Index des Maximums eines Arrays zurück
#
# a1,a2 = np.array_split(array, 2)  # Array in zwei Hälften teilen
# np.size(array)                    # Anzahl der Elemente eines Arrays ermitteln


########## ARRAY INDEXING ##########
# y[n - 1::n]                       # liefert aus einem Array jeden n-ten Wert als Array


########## DIFFERENT STUFF ##########
# R = const.physical_constants["molar gas constant"]      # Array of value, unit, error

##############Vorarbeit###################

h = 6.62607004*10**(-34)
c = 299792458
d = 201.4*10**(-12)
const = (h*c)/(2*d)
e = 1.6021766208*10**(-19)
R = 13.6*e
Rd = 10973731
alpha = 7.2973525664*10**(-3)
def Energie(Theta):
    """
        Args:
            Winkel: Theta [degree]
        Returns:
            Energie: E [eV]
    """
    Theta = Theta/360 * 2*np.pi
    E = const/np.sin(Theta)/e
    return E

def Sigma(z, E, n, j):
    """
        Args:
            z: Kernladungszahl
            E: Kantenenergie [Joule]
            n: Ankunfsschale
            j: Herkunftsschale
        Returns:
            Sigma
    """
    return z-np.sqrt( E / ( R*(1/(n**2) - (1/j**2)) ) )

def Sigma_L(Z, E):
    """
        Args:
            Bullcrap
        Returns:
            Bullshit
    """
    return Z - np.sqrt( 4/alpha * np.sqrt(E/R)- 5*E/R ) * np.sqrt( 1 + 19/32*alpha**2 * E/R )

E_max_roehre = 35

write('build/E_max_t.tex', make_SI(E_max_roehre, r'\kilo\electronvolt', figures=1))
lambda_min_t = c*h/(E_max_roehre*e)
write('build/lambda_min_t.tex', make_SI(lambda_min_t*10**12, r'\pico\metre', figures=1))










#############1#####################

theta = np.genfromtxt('messdaten/mess_1_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_1_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
plt.ylim(np.amin(I)-10, 270)
plt.axvline(14, color='g', linestyle='--')

plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse$')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_1.pdf')

#############2##############

theta = np.genfromtxt('messdaten/mess_2_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_2_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
k_kante_b = 19.875
k_kante_a = 22.2
plt.axvline(k_kante_a, color='b', linestyle='--')
plt.axvline(k_kante_b, color='g', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse$')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_2.pdf')

z = 29
E_k_kante_a = Energie(k_kante_a)*e
E_k_kante_b = Energie(k_kante_b)*e
sigma_1 = z - np.sqrt(E_k_kante_b/R)                      #wieso auch immer diese Reihenfolge Idk
sigma_2 = z - 2*np.sqrt((R*(z-sigma_1)**2-E_k_kante_a)/R) #still dont know....

# Meine Ansätze, Sigma zu bestimmen - Ich denke, so ist es richtig (Altprotokoll macht komische Sachen)
sigma_2_nach_jay = z-np.sqrt( E_k_kante_b / ( R*(1 - (1/9)) ) )
write('build/sigma_2_nach_jay.tex', make_SI(sigma_2_nach_jay, r' ', figures=2))
sigma_1_nach_jay = z-np.sqrt( E_k_kante_a / ( R*(1 - (1/4)) ) )
write('build/sigma_1_nach_jay.tex', make_SI(sigma_1_nach_jay, r' ', figures=2))

write('build/Theta_k_kante_a.tex', make_SI(k_kante_a, r'\degree', figures=1))
write('build/Theta_k_kante_b.tex', make_SI(k_kante_b, r'\degree', figures=1))
write('build/E_k_kante_a_cu.tex', make_SI(E_k_kante_a*10**(-3)/e, r'\kilo\electronvolt', figures=2))
write('build/E_k_kante_b_cu.tex', make_SI(E_k_kante_b*10**(-3)/e, r'\kilo\electronvolt', figures=2))
write('build/sigma_1_cu.tex', make_SI(sigma_1, r' ', figures=2))
write('build/sigma_2_cu.tex', make_SI(sigma_2, r' ', figures=2))

E_k_kante_a_lit = 8.04699993*10**3*e
E_k_kante_b_lit = 8.90400028*10**3*e

sigma_1_lit     = z - np.sqrt(E_k_kante_b_lit/R)                      #wieso auch immer diese Reihenfolge Idk
sigma_2_lit     = z - 2*np.sqrt((R*(z-sigma_1_lit)**2-E_k_kante_a_lit)/R) #still dont know....again


write('build/E_k_kante_a_lit_cu.tex', make_SI(E_k_kante_a_lit/e*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/E_k_kante_b_lit_cu.tex', make_SI(E_k_kante_b_lit/e*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/sigma_1_lit_cu.tex', make_SI(sigma_1_lit, r' ', figures=2))
write('build/sigma_2_lit_cu.tex', make_SI(sigma_2_lit, r' ', figures=2))
E_k_kante_a_rel = abs(E_k_kante_a_lit - E_k_kante_a)/E_k_kante_a_lit * 100
E_k_kante_b_rel = abs(E_k_kante_b_lit - E_k_kante_b)/E_k_kante_b_lit * 100
sigma_1_rel = abs(sigma_1 - sigma_1_lit)/sigma_1_lit * 100
sigma_2_rel = abs(sigma_2 - sigma_2_lit)/sigma_2_lit * 100
write('build/E_k_kante_a_rel_cu.tex', make_SI(E_k_kante_a_rel, r'\percent', figures=1))
write('build/E_k_kante_b_rel_cu.tex', make_SI(E_k_kante_b_rel, r'\percent', figures=1))
write('build/sigma_1_rel_cu.tex', make_SI(sigma_1_rel, r'\percent', figures=1))
write('build/sigma_2_rel_cu.tex', make_SI(sigma_2_rel, r'\percent', figures=1))

#############Auflösungsvermögen#############
#Nehme lineare Steigung zwischen Peaks an
theta_1_start = 19.4
theta_1_ende = 20.4
theta_1_peak = 19.9
delta_theta_1_l = theta_1_peak - theta_1_start
delta_theta_1_r = theta_1_ende - theta_1_peak

theta_1_halb_l = theta_1_start + 0.5*delta_theta_1_l
theta_1_halb_r = theta_1_peak + 0.5*delta_theta_1_r

theta_halbwert = theta_1_halb_r - theta_1_halb_l
energie_halbwert = Energie(theta_1_halb_l) - Energie(theta_1_halb_r)

write('build/delta_Energie_1.tex', make_SI(energie_halbwert*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/delta_theta_1.tex', make_SI(theta_halbwert, r'\degree', figures=2))


theta_2_start = 21.6
theta_2_ende = 22.8
theta_2_peak = 22.2
delta_theta_2_l = theta_2_peak - theta_2_start
delta_theta_2_r = theta_2_ende - theta_2_peak

theta_2_halb_l = theta_2_start + 0.5*delta_theta_2_l
theta_2_halb_r = theta_2_peak + 0.5*delta_theta_2_r

theta_halbwert_2 = theta_2_halb_r - theta_2_halb_l
energie_halbwert_2 = Energie(theta_2_halb_l) - Energie(theta_2_halb_r)

write('build/delta_Energie_2.tex', make_SI(energie_halbwert_2*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/delta_theta_2.tex', make_SI(theta_halbwert_2, r'\degree', figures=2))
#############3################

theta = np.genfromtxt('messdaten/mess_3_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_3_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
theta_min = 4.7
plt.axvline(theta_min, color='b', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse$')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_3.pdf')

E_max      = Energie(theta_min)
lambda_min = c*h/(E_max*e)
write('build/Theta_min.tex', make_SI(theta_min, r'\degree', figures=1))
write('build/E_max.tex', make_SI(E_max*10**(-3), r'\kilo\electronvolt', figures=1))
write('build/lambda_min.tex', make_SI(lambda_min*10**12, r'\pico\metre', figures=1))
E_max_lit  = 35*10**3

E_max_rel = abs(E_max - E_max_lit)/E_max_lit * 100
write('build/E_max_rel.tex', make_SI(E_max_rel, r'\percent', figures=1))


################Germanium################

theta = np.genfromtxt('messdaten/mess_ge_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_ge_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
plt.ylim(np.amin(I), 50)
plt.axvline(16.1, color='k', linestyle='--')
plt.axvline(15.4, color='k', linestyle='--')
kante = (16.1-15.4)/2 + 15.4
plt.axvline(kante, color='b', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_ge.pdf')

E_ge = Energie(kante)
sigma_ge = Sigma(32,E_ge*e,1,3 )
write('build/E_ge.tex', make_SI(E_ge*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/sigma_ge.tex', make_SI(sigma_ge, r' ', figures=2))


##################Zirkonium##############

theta = np.genfromtxt('messdaten/mess_zr_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_zr_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
plt.ylim(np.amin(I), 300)
plt.axvline(9.2, color='k', linestyle='--')
plt.axvline(10, color='k', linestyle='--')
kante = (10-9.2)/2 + 9.2
plt.axvline(kante, color='b', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_zr.pdf')

E_zr = Energie(kante)
sigma_zr = Sigma(40,E_zr*e,1,3 )
write('build/E_zr.tex', make_SI(E_zr*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/sigma_zr.tex', make_SI(sigma_zr, r' ', figures=2))

################Strontium################

theta = np.genfromtxt('messdaten/mess_sr_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_sr_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
plt.ylim(np.amin(I), 180)
plt.axvline(10.3, color='k', linestyle='--')
plt.axvline(11, color='k', linestyle='--')
kante = (11-10.3)/2 + 10.3
plt.axvline(kante, color='b', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Anzahl gemessener Impulse')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_sr.pdf')

E_sr = Energie(kante)
sigma_sr = Sigma(38,E_sr*e,1,3 )
write('build/E_sr.tex', make_SI(E_sr*10**(-3), r'\kilo\electronvolt', figures=2))
write('build/sigma_sr.tex', make_SI(sigma_sr, r' ', figures=2))

##############Wismut#############

theta = np.genfromtxt('messdaten/mess_wi_winkel.txt', unpack=True)
I     = np.genfromtxt('messdaten/mess_wi_rate.txt', unpack=True)
theta = theta/2

plt.clf()                   # clear actual plot before generating a new one
t_plot = np.linspace(np.amin(theta)-0.1, np.amax(theta)+0.1 , 100)
plt.xlim(np.amin(theta)-0.1, np.amax(theta)+0.1)
plt.ylim(np.amin(I), 160)
plt.axvline(11.2, color='k', linestyle='--')
plt.axvline(13.2, color='k', linestyle='--')
#kante = 13.2-11.2
#plt.axvline(kante, color='b', linestyle='--')


plt.plot(theta, I, 'r.', label=r'Absorbtionsspektrum von Wismut (L-Kanten)$')
plt.xlabel(r'$\Theta \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\: \text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_wi.pdf')

write('build/theta_wi_1.tex', make_SI(11.2, r'\degree', figures=1))
write('build/theta_wi_2.tex', make_SI(13.2, r'\degree', figures=1))


delta_E_wi = Energie(11.2) - Energie(13.2)
write('build/wi_delta_E.tex', make_SI(delta_E_wi, r'\electronvolt', figures=1))
delta_E_wi = delta_E_wi * e

sigma_wi = Sigma_L(83, delta_E_wi)
write('build/sigma_wi.tex', make_SI(sigma_wi, r' ', figures=2))


############RYDBERG#########
#Z_array = ([32-1,38-1,40-1])
Z_array = ([32-sigma_ge,38-sigma_sr,40-sigma_zr])
E_array = ([E_ge, E_sr, E_zr])
E_array = np.sqrt(E_array)

plt.clf()                   # clear actual plot before generating a new one

plt.plot(Z_array, E_array, 'r.', label=r'Messwerte$')
plt.xlabel(r'$Z$')
plt.ylabel(r'$ \sqrt{E \:/\: \si{\joule} }  $')

params = ucurve_fit(reg_linear, Z_array, E_array)             # linearer Fit
a, b = params

t_plot = np.linspace(30, 42, 1000)
plt.plot(t_plot, a.n*t_plot+b.n, 'b-', label=r'Linearer Fit$')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_ryd.pdf')
write('build/parameter_a.tex', make_SI(a, r'\kilo\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b.tex', make_SI(b, r'\kilo\hertz', figures=2))      # type in Anz. signifikanter Stellen
ebberg = 4/3 *e* a**2/(h*c)
write('build/ebberg_konstante.tex', make_SI(4/3 * a**2, r'\electronvolt', figures=1))       # type in Anz. signifikanter Stellen
write('build/ebberg_konstante_ry.tex', make_SI(4/3 *e* a**2/(h*c), r'\per\metre', figures=1))

write('build/ebberg_konstante_lit.tex', make_SI(Rd, r'\per\metre', figures=1))

ebberg_rel = abs(ebberg - Rd)/Rd * 100
write('build/ebbergs_rel.tex', make_SI(ebberg_rel.n, r'\percent', figures=1))

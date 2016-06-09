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
#e],[0, 1, 0, 1, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
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

omega_l, omega_r, l = np.genfromtxt('messdaten/messung.txt', unpack=True)
omega_l_deg= omega_l
omega_r_deg = omega_r
omega_l = omega_l/180 * np.pi # deg -> rad
omega_r = omega_r/180 * np.pi # deg -> rad

###### a #####

phi_l = 345.9
phi_r = 237.5
phi_gemessen = 0.5 * (phi_r - phi_l)
write('build/phi.tex', make_SI(phi_gemessen, r'\degree', figures=1)) # Rechne aber nicht mit dem Phi sondern mit 60
phi = 60/180 * np.pi # 60 grad

eta = np.pi - (omega_r - omega_l)
n = ( np.sin( 0.5* (abs(eta)+phi) ) ) / ( np.sin(phi/2) )

write('build/tab1.tex', make_table([l, omega_l_deg, omega_r_deg, eta, n],[1, 0, 0, 3, 3]))
write('build/tab1_tex.tex', make_full_table(
    'Messdaten zur Bestimmung der Brechungsindices.',
    'tab1',
    'build/tab1.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                            # die Multicolumns sein sollen
    [
    r'$\lambda \:/\: \si{\nano\metre}$',
    r'$\Omega_l \:/\: \si{\degree}$',
    r'$\Omega_r \:/\: \si{\second}$',
    r'$\eta \:/\: \si{\radian}$',
    r'$n \:/\: \si{\radian}$']))


############ b und c ###################

from scipy.optimize import curve_fit

def f(x, a, b, c):
    return a + b/x**2 + c/x**4

parameter1, covariance1 = curve_fit(f, l, n**2)
t_plot = np.linspace(np.amin(l), np.amax(l), 1000000)

plt.plot(t_plot, np.sqrt(f(t_plot, parameter1[0], parameter1[1], parameter1[2])), 'b-', label=r'konkaver Fit', linewidth=1)
fehler1 = np.sqrt(np.diag(covariance1)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

a = parameter1[0]  #Hier fehlen die Fehler der Parameter, kA wieso covariance da 'inf' ausgibt...
b = parameter1[1]  #Hier fehlen die Fehler der Parameter, kA wieso covariance da 'inf' ausgibt...
c = parameter1[2]  #Hier fehlen die Fehler der Parameter, kA wieso covariance da 'inf' ausgibt...

write('build/A_0.tex', make_SI(a, r' ', figures=1))
write('build/A_2.tex', make_SI(b, r'\nano\metre\tothe{2}', figures=1))
write('build/A_4.tex', make_SI(c, r'\nano\metre\tothe{4}', figures=1))


s = (n**2 - a -b/l**2)**2
s_qua = 1/(np.size(n)-2)*np.sum(s)
write('build/Varianz_Methode_1.tex', make_SI(s_qua, r' ', figures=7))
np.savetxt('build/ausgleichswerte.txt', np.column_stack([parameter1, fehler1]), header=" ")

def f2(x, a, b, c):
    return a - b*x**2 - c*x**4

parameter2, covariance2 = curve_fit(f2, l, n**2)

#plt.plot(t_plot, f2(t_plot, parameter2[0], parameter2[1], parameter2[2]), 'g-', label=r'Ausgleich_2', linewidth=1) Erste Methode ist besser => nicht plotten
fehler2 = np.sqrt(np.diag(covariance2)) # Diagonalelemente der Kovarianzmatrix stellen Varianzen dar

s = (n**2 - a +b*l**2)**2
s_qua = 1/(np.size(n)-2)*np.sum(s)
write('build/Varianz_Methode_2.tex', make_SI(s_qua, r' ', figures=7))
np.savetxt('build/ausgleichswerte2.txt', np.column_stack([parameter2, fehler2]), header=" ")



t_plot = np.linspace(np.amin(l), np.amax(l), 1000000)
#plt.plot(t_plot, t_plot*o.n+i.n, 'y-', label='Linearer Fit')



plt.plot(l, n, 'rx', label='Messdaten')
#plt.xlim(-0.01, 0.025)
#plt.ylim(-1000, 50000)
plt.xlabel(r'$\lambda \:/\: \si{\nano\metre}$')
plt.ylabel(r'$n$')
plt.legend(loc='best')
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a_1.pdf')


#################### d ##################


v = (f(589,a,b,c) - 1)/(f(486,a,b,c) - f(656,a,b,c))
write('build/abbesche_zahl.tex', make_SI(v, r' ', figures=1))



################ e ###################
lambda_c = 656*10**(-9)
lambda_f = 486*10**(-9)
A_lambda_c = -0.05*(2*b*10**(-18)/(lambda_c**3) + 4*c*10**(-32)/(lambda_c**5))
A_lambda_f = -0.05*(2*b*10**(-18)/(lambda_f**3) + 4*c*10**(-32)/(lambda_f**5))
write('build/A_lambda_c.tex', make_SI(A_lambda_c, r' ', figures=1))
write('build/A_lambda_f.tex', make_SI(A_lambda_f, r' ', figures=1))


################# f ##################

lambda_i = np.sqrt(b*10**(-18)/(a-1))
write('build/lambda_i.tex', make_SI(lambda_i*10**9, r'\nano\metre', figures=1)) # Gültigkeit nicht zu hoch einschätzen, da 1. getaylort und 2. gefittet und 3. Messwerte entsprechen nur einem kleinen Bereich

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



z_1, s_1 = np.genfromtxt('messdaten/1.txt', unpack=True)
s_1 = s_1/5.046*10**(-3)
lambda_laser = 2*s_1/z_1

#oder ohne die beiden Kackwerte

z_x, s_x = np.genfromtxt('messdaten/1_1.txt', unpack=True)
s_x = s_x/5.046*10**(-3)
lambda_laser_x = 2*s_x/z_x
lambda_laser_x = ufloat(np.mean(lambda_laser_x) ,MeanError(noms(lambda_laser_x)))
write('build/lambda_laser_x.tex', make_SI(lambda_laser_x * 1e9, r'\nano\metre', figures=3))


write('build/Tabelle_a.tex', make_table([s_1*10**3, z_1, lambda_laser*10**9],[3, 0, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_a_texformat.tex', make_full_table(
    'Messdaten bezüglich der Wellenlänge.',
    'tab:1',
    'build/Tabelle_a.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                           # die Multicolumns sein sollen
    [
    r'$\Delta l \:/\: \si{\milli\metre}$',
    r'$\text{Maxima}$',
    r'$\lambda \:/\: \si{\nano\metre}$']))

lambda_laser = ufloat(np.mean(lambda_laser) ,MeanError(noms(lambda_laser)))
write('build/lambda_laser.tex', make_SI(lambda_laser * 1e9, r'\nano\metre', figures=3))

z_2, p_2 = np.genfromtxt('messdaten/2.txt', unpack=True)
n_luft = 1 + (z_2*lambda_laser)/(2*0.05)*296.15/273.15*1.0132/p_2  # 1+(z*lambda)/(2*b)*(T)/(T_0)*(p_0)/(Delta_p)

write('build/Tabelle_b.tex', make_table([z_2, p_2, unp.nominal_values(n_luft)],[0, 1, 5]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_b_texformat.tex', make_full_table(
    'Messdaten bezüglich des Brechungsindex von Luft.',
    'tab:2',
    'build/Tabelle_b.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                           # die Multicolumns sein sollen
    [
    r'$\text{Maxima}$',
    r'$\Delta p \:/\: \si{\bar}$',
    r'$n$']))

#n_luft = ufloat(np.mean(n_luft) ,MeanError(noms(n_luft)))
n_luft = np.mean(n_luft)
write('build/n_luft.tex', make_SI(n_luft, r'', figures=2))
write('build/n_luft_lit.tex', make_SI(1.000292, r'', figures=6))   #http://www.chemie.de/lexikon/Brechzahl.html
n_luft_rel = abs(unp.nominal_values(n_luft)-1.000292)/1.000292 *100
write('build/n_luft_rel.tex', make_SI(n_luft_rel, r'\percent', figures=3))

n_luft_neu = 1 + (z_2*lambda_laser_x)/(2*0.05)*296.15/273.15*1.0132/p_2

n_luft_neu = np.mean(n_luft_neu)
write('build/n_luft_neu.tex', make_SI(n_luft_neu, r'', figures=1))
n_luft_rel_neu = abs(unp.nominal_values(n_luft_neu)-1.000292)/1.000292 *100
write('build/n_luft_rel_neu.tex', make_SI(n_luft_rel_neu, r'\percent', figures=3))



z_3, p_3 = np.genfromtxt('messdaten/3.txt', unpack=True)
n_gas = 1 + (z_3*lambda_laser)/(2*0.05)*296.15/273.15*1.0132/p_3  # 1+(z*lambda)/(2*b)*(T)/(T_0)*(p_0)/(Delta_p)

write('build/Tabelle_c.tex', make_table([z_3, p_3, unp.nominal_values(n_gas)],[0, 1, 5]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_c_texformat.tex', make_full_table(
    'Messdaten bezüglich des Brechungsindex von $\ce{C4H8}$.',
    'tab:3',
    'build/Tabelle_c.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                           # die Multicolumns sein sollen
    [
    r'$\text{Maxima}$',
    r'$\Delta p \:/\: \si{\bar}$',
    r'$n$']))

#n_luft = ufloat(np.mean(n_luft) ,MeanError(noms(n_luft)))
n_gas = np.mean(n_gas)
write('build/n_gas.tex', make_SI(n_gas, r'', figures=1))
write('build/n_gas_lit.tex', make_SI(1.3811, r'', figures=4))   #http://www.chemicalbook.com/ChemicalProductProperty_DE_CB4763080.htm
n_gas_rel = abs(unp.nominal_values(n_gas)-1.3811)/1.3811 *100
write('build/n_gas_rel.tex', make_SI(n_gas_rel, r'\percent', figures=3))

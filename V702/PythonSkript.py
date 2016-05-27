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



from matplotlib.ticker import FormatStrFormatter



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
import math

Nullprosec = ((210+191)/2)/900
write('build/nulleffekt.tex', make_SI(Nullprosec, r'\per\second', figures=1))

Indium  = np.genfromtxt('messdaten/Indium.txt', unpack=True)
Indium  = np.log(Indium - Nullprosec*220)

Zeit = np.arange(1,18)
Zeit = Zeit*220
np.savetxt('messdaten/test.txt', np.column_stack([Zeit]), header="Impulse /220s")

params = ucurve_fit(reg_linear, Zeit, Indium)             # linearer Fit
a, b = params
write('build/parameter_a_indium.tex', make_SI(a, r'\per\second', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b_indium.tex', make_SI(b, r'', figures=2))      # type in Anz. signifikanter Stellen
write('build/lambda_indium.tex', make_SI(-a, r'\per\second', figures=2))
write('build/halbzeit_indium.tex', make_SI(np.log(2)/(-a)/60, r'\minute', figures=2))
write('build/halbzeit_indium_lit.tex', make_SI(54.29, r'\minute', figures=2)) #http://www.periodensystem-online.de/index.php?id=isotope&el=49&mz=116&nrg=0.1273&show=nuklid
write('build/halbzeit_indium_rel.tex', make_SI(abs(54.29-np.log(2)/(-a)/60)/54.29*100, r'\percent', figures=2))




t_plot = np.linspace(np.amin(Zeit), np.amax(Zeit), 100)
plt.plot(t_plot, np.exp(t_plot*a.n+b.n), 'b-', label='Linearer Fit')
#plt.plot(Zeit, np.exp(Indium), 'rx', label='logarithmierte Messdaten')
plt.errorbar(Zeit, np.exp(Indium), fmt='rx', yerr=np.sqrt(np.exp(Indium)), label='Messdaten')        # mit Fehlerbalken

ax = plt.gca()
ax.set_yscale('log')
plt.tick_params(axis='y', which='minor')
ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\selinder, 'rx', label='Messdaten')
plt.xlim(t_plot[0], t_plot[-1])
plt.ylim(700, 2500)
plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$\text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ausgleich.pdf')

plt.clf()


Rhodium = np.genfromtxt('messdaten/Rhodium.txt', unpack=True) # <- Alle Messwerte
Rhodium = np.log(Rhodium - Nullprosec*17)

Zeit = np.arange(1,44)
Zeit = Zeit*17

Rhodium1 = np.genfromtxt('messdaten/Rhodium1.txt', unpack=True) # <- Alle Messwerte ab 14
Rhodium1 = np.log(Rhodium1 - Nullprosec*17)

Zeit1 = np.arange(14,44)
Zeit1 = Zeit1*17



params = ucurve_fit(reg_linear, Zeit1, Rhodium1)             # linearer Fit
a, b = params
write('build/parameter_a_rhodium.tex', make_SI(a, r'per\second', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b_rhodium.tex', make_SI(b, r'\nothing', figures=2))      # type in Anz. signifikanter Stellen
write('build/lambda_rhodium.tex', make_SI(-a, r'\per\second', figures=2))
write('build/halbzeit_rhodium.tex', make_SI(np.log(2)/(-a)/60, r'\minute', figures=2))
write('build/halbzeit_rhodium_lit.tex', make_SI(13/3, r'\minute', figures=2)) #http://www.periodensystem-online.de/index.php?id=isotope&el=45&mz=104&nrg=0.129&show=nuklid
write('build/halbzeit_rhodium_rel.tex', make_SI(abs(13/3-np.log(2)/(-a)/60)/(13/3)*100, r'\percent', figures=2))

Rhodium2 = np.genfromtxt('messdaten/Rhodium2.txt', unpack=True) # <- die ersten 14 Messwerte
Zeit2 = np.arange(1,15)
Zeit2 = Zeit2*17

np.savetxt('messdaten/test.txt', np.column_stack([Rhodium2 - Nullprosec*17]), header="Impulse /17s")

#### irgendwas stimmt hier noch nicht, oder so, wahrscheinlich, äh, keine Ahnung >.<

Rhodium2[0]  = np.log(Rhodium2[0]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[0] ))
Rhodium2[1]  = np.log(Rhodium2[1]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[1] ))
Rhodium2[2]  = np.log(Rhodium2[2]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[2] ))
Rhodium2[3]  = np.log(Rhodium2[3]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[3] ))
Rhodium2[4]  = np.log(Rhodium2[4]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[4] ))
Rhodium2[5]  = np.log(Rhodium2[5]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[5] ))
Rhodium2[6]  = np.log(Rhodium2[6]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[6] ))
Rhodium2[7]  = np.log(Rhodium2[7]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[7] ))
Rhodium2[8]  = np.log(Rhodium2[8]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[8] ))
Rhodium2[9]  = np.log(Rhodium2[9]  - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[9] ))
Rhodium2[10] = np.log(Rhodium2[10] - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[10]))
Rhodium2[11] = np.log(Rhodium2[11] - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[11]))
Rhodium2[12] = np.log(Rhodium2[12] - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[12]))
Rhodium2[13] = np.log(Rhodium2[13] - Nullprosec*17 - np.exp(b.n)*np.exp(a.n*Zeit2[13]))
#Rhodium2[14] = np.log(Rhodium2[14] - Nullprosec*17 - np.e**(4.27)*np.e**(-0.00302*Zeit2[14]))
#Rhodium2[15] = np.log(Rhodium2[15] - Nullprosec*17 - np.e**(4.27)*np.e**(-0.00302*Zeit2[15]))
#Rhodium2[16] = np.log(Rhodium2[16] - Nullprosec*17 - np.e**(4.27)*np.e**(-0.00302*Zeit2[16]))

np.savetxt('messdaten/test2.txt', np.column_stack([np.e**(4.27)*(1 - np.e**(-0.00302*17))*np.e**(-0.00302*Zeit2[0] )]), header="Impulse /17s")




params = ucurve_fit(reg_linear, Zeit2, Rhodium2)             # linearer Fit
c, d = params


t_plot = np.linspace(np.amin(Zeit1), np.amax(Zeit1), 100)
plt.plot(t_plot, np.exp(t_plot*a.n+b.n), 'b-', label='Linearer Fit langlebig')
t_plot2 = np.linspace(np.amin(Zeit2), np.amax(Zeit2), 100)
plt.plot(t_plot2, np.exp(t_plot2*c.n+d.n), 'g-', label='Linearer Fit kurzlebig ')

#plt.plot(Zeit2, Rhodium2, 'gx', label='logarithmierte Messdaten kurzlebig')
plt.errorbar(Zeit2, np.exp(Rhodium2), fmt='gx', yerr=np.sqrt(np.exp(Rhodium2)), label='Korrigierte Messdaten für kurzlebiges Isotop')        # mit Fehlerbalken

#plt.plot(Zeit, Rhodium, 'rx', label='logarithmierte Messdaten insgesamt')
plt.errorbar(Zeit, np.exp(Rhodium), fmt='rx', yerr=np.sqrt(np.exp(Rhodium)), label='Gesamte Messdaten')        # mit Fehlerbalken


# t_plot = np.linspace(-0.5, 2 * np.pi + 0.5, 1000) * 1e-3
#
## standard plotting
# plt.plot(t_plot * 1e3, f(t_plot, *noms(params)) * 1e-3, 'b-', label='Fit')
# plt.plot(t * 1e3, U * 1e3, 'rx', label='Messdaten')
## plt.errorbar(B * 1e3, noms(y) * 1e5, fmt='rx', yerr=stds(y) * 1e5, label='Messdaten')        # mit Fehlerbalken
## plt.xscale('log')                                                                            # logarithmische x-Achse
# plt.xlim(t_plot[0] * 1e3, t_plot[-1] * 1e3)
# plt.xlabel(r'$t \:/\: \si{\milli\selinder, 'rx', label='Messdaten')

plt.yscale('log')
#plt.xlim(t_plot[0], t_plot[-1])
#ax = plt.gca()
#ax.set_yscale('log')
#plt.tick_params(axis='y', which='minor')
#ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))

plt.xlabel(r'$t \:/\: \si{\second}$')
plt.ylabel(r'$\text{Impulse}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ausgleich2.pdf')

write('build/parameter_c_rhodium.tex', make_SI(c, r'per\second', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_d_rhodium.tex', make_SI(d, r'\nothing', figures=2))      # type in Anz. signifikanter Stellen
write('build/lambda_rhodium2.tex', make_SI(-c, r'\per\second', figures=2))
write('build/halbzeit_rhodium2.tex', make_SI(np.log(2)/(-c), r'\second', figures=2))
write('build/halbzeit_rhodium2_lit.tex', make_SI(42.3, r'\second', figures=2)) #http://www.internetchemie.info/chemiewiki/index.php?title=Rhodium-Isotope
write('build/halbzeit_rhodium2_rel.tex', make_SI(abs(42.3-np.log(2)/(-c))/(42.3)*100, r'\percent', figures=2))

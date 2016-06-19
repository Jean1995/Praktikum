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

#### Grenzspannungen

## Messung 1 (grün)

I_1, U_1 = np.genfromtxt('messdaten/messung_1.txt', unpack=True) # I[nA] !
params_1 = ucurve_fit(reg_linear, U_1, np.sqrt(I_1))
a_1, b_1 = params_1
write('build/a_1.tex', make_SI(a_1, r'\ampere\tothe{0.5}\per\volt', figures=1))
write('build/b_1.tex', make_SI(b_1, r'\ampere\tothe{0.5}', figures=1))

t_plot_1 = np.linspace(np.amin(U_1), np.amax(U_1)+0.03, 99)
plt.plot(t_plot_1, a_1.n*t_plot_1+b_1.n, 'b-', label='Linearer Fit')
plt.plot(U_1, np.sqrt(I_1), 'rx', label='Messdaten')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$\sqrt{I} \:/\: \sqrt{\si{\nano\ampere}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_1.pdf')

U_g_1 = -b_1/a_1
write('build/U_g_1.tex', make_SI(U_g_1, r'\volt', figures=1))

## Messung 2 (indigo)

plt.clf()
I_2, U_2 = np.genfromtxt('messdaten/messung_2.txt', unpack=True) # I[nA] !
params_2 = ucurve_fit(reg_linear, U_2, np.sqrt(I_2))
a_2, b_2 = params_2
write('build/a_2.tex', make_SI(a_2, r'\ampere\tothe{0.5}\per\volt', figures=1))
write('build/b_2.tex', make_SI(b_2, r'\ampere\tothe{0.5}', figures=1))

t_plot_2 = np.linspace(np.amin(U_2), np.amax(U_2)+0.03, 99)
plt.plot(t_plot_2, a_2.n*t_plot_2+b_2.n, 'b-', label='Linearer Fit')
plt.plot(U_2, np.sqrt(I_2), 'rx', label='Messdaten')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$\sqrt{I} \:/\: \sqrt{\si{\nano\ampere}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_2.pdf')

U_g_2 = -b_2/a_2
write('build/U_g_2.tex', make_SI(U_g_2, r'\volt', figures=1))

## Messung 3 (violett)

plt.clf()
I_3, U_3 = np.genfromtxt('messdaten/messung_3.txt', unpack=True) # I[nA] !
params_3 = ucurve_fit(reg_linear, U_3, np.sqrt(I_3))
a_3, b_3 = params_3
write('build/a_3.tex', make_SI(a_3, r'\ampere\tothe{0.5}\per\volt', figures=1))
write('build/b_3.tex', make_SI(b_3, r'\ampere\tothe{0.5}', figures=1))

t_plot_3 = np.linspace(np.amin(U_3), np.amax(U_3)+0.03, 99)
plt.plot(t_plot_3, a_3.n*t_plot_3+b_3.n, 'b-', label='Linearer Fit')
plt.plot(U_3, np.sqrt(I_3), 'rx', label='Messdaten')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$\sqrt{I} \:/\: \sqrt{\si{\nano\ampere}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_3.pdf')

U_g_3 = -b_3/a_3
write('build/U_g_3.tex', make_SI(U_g_3, r'\volt', figures=1))

## Messung 4 (UV)

plt.clf()
I_4, U_4 = np.genfromtxt('messdaten/messung_4.txt', unpack=True) # I[nA] !
params_4 = ucurve_fit(reg_linear, U_4, np.sqrt(I_4))
a_4, b_4 = params_4
write('build/a_4.tex', make_SI(a_4, r'\ampere\tothe{0.5}\per\volt', figures=1))
write('build/b_4.tex', make_SI(b_4, r'\ampere\tothe{0.5}', figures=1))

t_plot_4 = np.linspace(np.amin(U_4), np.amax(U_4)+0.03, 99)
plt.plot(t_plot_4, a_4.n*t_plot_4+b_4.n, 'b-', label='Linearer Fit')
plt.plot(U_4, np.sqrt(I_4), 'rx', label='Messdaten')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$\sqrt{I} \:/\: \sqrt{\si{\nano\ampere}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_4.pdf')

U_g_4 = -b_4/a_4
write('build/U_g_4.tex', make_SI(U_g_4, r'\volt', figures=1))

# Messung 5 (orange)

plt.clf()
I_5, U_5 = np.genfromtxt('messdaten/messung_5.txt', unpack=True) # I[nA] !
params_5 = ucurve_fit(reg_linear, U_5, np.sqrt(I_5))
a_5, b_5 = params_5
write('build/a_5.tex', make_SI(a_5, r'\ampere\tothe{0.5}\per\volt', figures=1))
write('build/b_5.tex', make_SI(b_5, r'\ampere\tothe{0.5}', figures=1))

t_plot_5 = np.linspace(np.amin(U_5), np.amax(U_5)+0.03, 99)
plt.plot(t_plot_5, a_5.n*t_plot_5+b_5.n, 'b-', label='Linearer Fit')
plt.plot(U_5, np.sqrt(I_5), 'rx', label='Messdaten')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$\sqrt{I} \:/\: \sqrt{\si{\nano\ampere}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_5.pdf')

U_g_5 = -b_5/a_5
write('build/U_g_5.tex', make_SI(U_g_5, r'\volt', figures=1))





### Und jetzt die Bestimmung

#Quelle Wellenlängen: https://de.wikipedia.org/wiki/Quecksilberdampflampe
l = np.array([546.07, 491.6, 435.83, 404.66, 576.96])
U_g = np.array([U_g_1.n, U_g_2.n, U_g_3.n, U_g_4.n, U_g_5.n])
U_g_err = np.array([U_g_1.s, U_g_2.s, U_g_3.s, U_g_4.s, U_g_5.s])
U_g_ges = np.array([U_g_1, U_g_2, U_g_3, U_g_4, U_g_5])
c=299792458
f = c/l

plt.clf()
params_6 = ucurve_fit(reg_linear, f, U_g)
a_6, b_6 = params_6
write('build/a_6.tex', make_SI(a_6*10**(-9)*10**(15), r'\volt\second','e-15', figures=3))
write('build/b_6.tex', make_SI(b_6, r'\volt', figures=1))
t_plot_6 = np.linspace(np.amin(f), np.amax(f), 99)
plt.plot(t_plot_6*10**-3, a_6.n*t_plot_6+b_6.n, 'b-', label='Linearer Fit')
plt.errorbar(f*10**-3, U_g, fmt='rx', yerr=U_g_err, label='Messdaten')
plt.xlabel(r'$f \:/\: \si{\tera\hertz}$')
plt.ylabel(r'$U_g \:/\: \si{volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/messung_6.pdf')

#http://physics.nist.gov/cuu/Constants/index.html

h = 6.626070040*10**(-34)
e = 1.6021766208*10**(-19)

m_abw = (a_6*10**(-9) - (h/e)) / (h/e)
write('build/abw_he.tex', make_SI(m_abw.n*100, r'\percent', figures=1))
write('build/ak.tex', make_SI(b_6, r'\electronvolt', figures=1))
write('build/a_lit.tex', make_SI(h/e*10**15, r'\volt\second','e-15', figures=3))


tab_a = np.array([a_1, a_2, a_3, a_4, a_5])
tab_b = np.array([b_1, b_2, b_3, b_4, b_5])

write('build/Tabelle_0.tex', make_table([l, tab_a, tab_b],[2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_0_texformat.tex', make_full_table(
    'Parameter der linearen Fits.',
    'tab:0',
    'build/Tabelle_0.tex',
    [1, 2],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                           # die Multicolumns sein sollen
    [
    r'$\lambda \:/\: \si{\nano\metre}$',
    r'$m \:/\: \si{\volt\second}$',
    r'$b \:/\: \si{\volt}$']))

write('build/Tabelle_1.tex', make_table([l,U_g_ges],[2, 1]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_1_texformat.tex', make_full_table(
    'Schnittpunkte mit der Spannungsachse.',
    'tab:1',
    'build/Tabelle_1.tex',
    [1],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                           # die Multicolumns sein sollen
    [
    r'$\lambda \:/\: \si{\nano\metre}$',
    r'$U_g \:/\: \si{\volt}$',]))


#### Teil 2 ####

plt.clf()
I, U = np.genfromtxt('messdaten/messung_lang.txt', unpack=True) # I[nA] !
plt.plot(U, I, 'rx', label='Messdaten für das orangene Licht')



plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$I \:/\: \si{\nano\ampere}$')
plt.legend(loc='lower left')
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.axes([0.57, 0.57, 0.35, 0.35])
plt.grid()
plt.plot(U[19:37], I[19:37], 'rx', label=r'Messdaten für das orangene Licht im Bereich um $\SI{0}{\volt}$')

plt.savefig('build/messung_lang.pdf')


plt.clf()
plt.plot(U[19:37], I[19:37], 'rx', label=r'Messdaten für das orangene Licht im Bereich um $\SI{0}{\volt}$')
plt.xlabel(r'$U \:/\: \si{\volt}$')
plt.ylabel(r'$I \:/\: \si{\nano\ampere}$')
plt.legend(loc='best')
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)


plt.savefig('build/messung_lang_2.pdf')


#### BACKUP

#l = np.array([546.07, 491.6, 435.83, 404.66, 576.96])
#U_g = np.array([U_g_1.n, U_g_2.n, U_g_3.n, U_g_4.n, U_g_5.n])
#U_g_err = np.array([U_g_1.s, U_g_2.s, U_g_3.s, U_g_4.s, U_g_5.s])
#c=299792458
#f = l*10**(-9)/c
#
#plt.clf()
#params_6 = ucurve_fit(reg_linear, f, U_g)
#a_6, b_6 = params_6
#write('build/a_6.tex', make_SI(a_6, r'\volt\metre', figures=1))
#write('build/b_6.tex', make_SI(a_6, r'\volt', figures=1))
#t_plot_6 = np.linspace(np.amin(f), np.amax(f), 99)
#plt.plot(t_plot_6, a_6.n*t_plot_6+b_6.n, 'b-', label='Linearer Fit')
#plt.errorbar(f, U_g, fmt='rx', yerr=U_g_err, label='Messdaten')
#plt.xlabel(r'$f \:/\: \si{\hertz}$')
#plt.ylabel(r'$U_g \:/\: \si{volt}$')
#plt.legend(loc='best')
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/messung_6.pdf')

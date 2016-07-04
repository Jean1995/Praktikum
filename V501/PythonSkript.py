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


# E-Feld Teilaufgabe a)

D_lang, U_1, U_2, U_3 = np.genfromtxt('messdaten/messung_E_lang.txt', unpack=True)
D_kurz, U_4, U_5 = np.genfromtxt('messdaten/messung_E_kurz.txt', unpack=True)
D_lang = D_lang * 0.0254
D_kurz = D_kurz * 0.0254 # in meter umrechnen

print("Die folgenden Plots werden Ihnen präsentiert von: Micra Tours!")
# U_b,1 = 200 V
params1 = ucurve_fit(reg_linear, U_1, D_lang)             # linearer Fit
m1, b1 = params1
write('build/parameter_m1.tex', make_SI(m1, r'\metre\per\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b1.tex', make_SI(b1, r'\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot1 = np.linspace(np.amin(U_1)-0.5, np.amax(U_1)+0.5, 100)
plt.plot(t_plot1, (m1.n*t_plot1+b1.n)*100, 'b-', label='Linearer Fit')
plt.plot(U_1, D_lang*100, 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$D \:/\: \si{\centi\metre}$')
plt.xlabel(r'$U_1 \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a1.pdf')

print("In Kooperation mit: Sassi Vacation!")
print("Starring Amba-Lamb-Guy (https://www.youtube.com/watch?v=fLQWKOB5se0)")
# U_b,2 = 250 V
plt.clf()
params2 = ucurve_fit(reg_linear, U_2, D_lang)             # linearer Fit
m2, b2 = params2
write('build/parameter_m2.tex', make_SI(m2, r'\metre\per\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b2.tex', make_SI(b2, r'\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot2 = np.linspace(np.amin(U_2)-0.5, np.amax(U_2)+0.5, 100)
plt.plot(t_plot2, (m2.n*t_plot2+b2.n)*100, 'b-', label='Linearer Fit')
plt.plot(U_2, D_lang*100, 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$D \:/\: \si{\centi\metre}$')
plt.xlabel(r'$U_2 \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a2.pdf')

print("... das maken dauert heute aber echt lang...")
# U_b,3 = 300 V
plt.clf()
params3 = ucurve_fit(reg_linear, U_3, D_lang)             # linearer Fit
m3, b3 = params3
write('build/parameter_m3.tex', make_SI(m3, r'\metre\per\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b3.tex', make_SI(b3, r'\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot3 = np.linspace(np.amin(U_3)-0.5, np.amax(U_3)+0.5, 100)
plt.plot(t_plot3, (m3.n*t_plot3+b3.n)*100, 'b-', label='Linearer Fit')
plt.plot(U_3, D_lang*100, 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$D \:/\: \si{\centi\metre}$')
plt.xlabel(r'$U_3 \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a3.pdf')

print("Wer lässt auch gefühlt 34 Funktionen in einem Protokoll fitten?!")
# U_b,4 = 350 V
plt.clf()
params4 = ucurve_fit(reg_linear, U_4, D_kurz)             # linearer Fit
m4, b4 = params4
write('build/parameter_m4.tex', make_SI(m4, r'\metre\per\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b4.tex', make_SI(b4, r'\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot4 = np.linspace(np.amin(U_4)-0.5, np.amax(U_4)+0.5, 100)
plt.plot(t_plot4, (m4.n*t_plot4+b4.n)*100, 'b-', label='Linearer Fit')
plt.plot(U_4, D_kurz*100, 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$D \:/\: \si{\centi\metre}$')
plt.xlabel(r'$U_4 \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a4.pdf')

print("Der isst auch kleine Kinder...")
# U_b,5 = 400 V
plt.clf()
params5 = ucurve_fit(reg_linear, U_5, D_kurz)             # linearer Fit
m5, b5 = params5
write('build/parameter_m5.tex', make_SI(m5, r'\metre\per\volt', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b5.tex', make_SI(b5, r'\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot5 = np.linspace(np.amin(U_5)-0.5, np.amax(U_5)+0.5, 100)
plt.plot(t_plot5, (m5.n*t_plot5+b5.n)*100, 'b-', label='Linearer Fit')
plt.plot(U_5, D_kurz*100, 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$D \:/\: \si{\centi\metre}$')
plt.xlabel(r'$U_5 \:/\: \si{\volt}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a5.pdf')

# E-Feld a), Teil 2

die_bs = np.array([b1.n, b2.n, b3.n, b4.n, b5.n])
die_bs_err = np.array([b1.s, b2.s, b3.s, b4.s, b5.s])
die_ms = np.array([m1.n, m2.n, m3.n, m4.n, m5.n])
die_ms_err = np.array([m1.s, m2.s, m3.s, m4.s, m5.s])
print("Übrigens: Das ist unser letztes Protkoll aus dem AP!")
U_b = np.array([200, 250, 300, 350, 400])
empf = unp.uarray([m1.n, m2.n, m3.n, m4.n, m5.n], [m1.s, m2.s, m3.s, m4.s, m5.s])
plt.clf()

write('Tabelle_c.tex', make_table([U_b, die_ms*10**4, die_ms_err*10**4, die_bs*10**3, die_bs_err*10**3],[0, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('Tabelle_c_texformat.tex', make_full_table(
    'Fitparameter: Steigung $m$ und y-Achsenabschnitt $b$.',
    'tab:c',
    'Tabelle_c.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$U_b \:/\: \si{\volt}$',
    r'$m \:/\: 10^{-4}\si{\metre\per\volt}$',
    r'$\increment{m} \:/\: 10^{-4}\si{\metre\per\volt}$',
    r'$b \:/\: 10^{-3}\si{\volt}$',
    r'$\increment{b} \:/\: 10^{-3}\si{\volt}$']))




params6 = ucurve_fit(reg_linear, 1/U_b, noms(empf))             # linearer Fit
m6, b6 = params6
write('build/parameter_m6.tex', make_SI(m6, r'\metre', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b6.tex', make_SI(b6, r'\metre\per\volt', figures=2))      # type in Anz. signifikanter Stellen
t_plot6 = np.linspace(np.amin(1/U_b*100)-0.001*20, np.amax(1/U_b*100)+0.001*20, 10)
plt.plot(t_plot6, (m6.n*t_plot6+b6.n*100), 'b-', label='Linearer Fit')
plt.errorbar(1/U_b*100, noms(empf)*100, fmt='rx', yerr=stds(empf)*100, label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{U_d} \:/\: 10^{-2}\si{\metre\per\volt}$')
plt.xlabel(r'$\frac{1}{U_\text{B}} \:/\: 10^{-2}\si{\volt\tothe{-1}}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a6.pdf')

d = 0.38 * 0.01
p = 1.9 * 0.01
L = 14.3*0.01
m6_lit = (p*L)/(2*d)

write('build/parameter_m6_lit.tex', make_SI(m6_lit, r'\metre', figures=3))       # type in Anz. signifikanter Stellen
err_m6 = (m6 - m6_lit) / m6_lit
write('build/parameter_m6_rel.tex', make_SI(err_m6*100, r'\percent', figures=2))

# E-Feld b)

v, A = np.genfromtxt('messdaten/frequenzen.txt', unpack=True)

U_b = 400 #(???)
D_amp = A[1] * 0.0254 # habe nicht ganz verstanden welchen Amplitudenwert wir nehmen sollen
U_amp = 1/m6 * U_b * D_amp # Formel umgestellt und eingesetzt
write('build/U_amp.tex', make_SI(U_amp, r'\volt', figures=2))
write('build/v0.tex', make_SI(v[0], r'\kilo\hertz', figures=1))
write('build/v1.tex', make_SI(v[1], r'\kilo\hertz', figures=1))
write('build/v2.tex', make_SI(v[2], r'\kilo\hertz', figures=1))
write('build/v3.tex', make_SI(v[3], r'\kilo\hertz', figures=1))

write('build/v0_mal_2.tex', make_SI(v[0]*2, r'\kilo\hertz', figures=1))
write('build/v3_durch_2.tex', make_SI(v[3]/2, r'\kilo\hertz', figures=1))





######################################

# B-Feld a)

D_lang, I_1, I_2, I_3 = np.genfromtxt('messdaten/messung_B_lang.txt', unpack=True)
D_kurz, I_4, I_5 = np.genfromtxt('messdaten/messung_B_kurz.txt', unpack=True)
D_lang = D_lang * 0.0254
D_kurz = D_kurz * 0.0254 # in meter umrechnen
U_b_B = np.array([250, 300, 350, 400, 450])

mu_0 = 4*np.pi*10**(-7)
N = 20 #? geraten
R = 0.282 # ? geraten
L = 17.5*0.01
B_1 = mu_0 * 8/np.sqrt(125) * N/R * I_1
B_2 = mu_0 * 8/np.sqrt(125) * N/R * I_2
B_3 = mu_0 * 8/np.sqrt(125) * N/R * I_3
B_4 = mu_0 * 8/np.sqrt(125) * N/R * I_4
B_5 = mu_0 * 8/np.sqrt(125) * N/R * I_5

# U_b,1 = 250 V
plt.clf()
params7 = ucurve_fit(reg_linear, B_1, D_lang/(L**2+D_lang**2))             # linearer Fit
m7, b7 = params7
write('build/parameter_m7.tex', make_SI(m7, r'\per\metre\per\tesla', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b7.tex', make_SI(b7, r'\per\metre', figures=2))      # type in Anz. signifikanter Stellen
t_plot7 = np.linspace(np.amin(B_1), np.amax(B_1), 100)
plt.plot(t_plot7*10**6, (m7.n*t_plot7+b7.n), 'b-', label='Linearer Fit')
plt.plot(B_1*10**(6), D_lang/(L**2+D_lang**2), 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{L^2 + D^2} \:/\: \si{\per\metre} $')
plt.xlabel(r'$B_1 \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a7.pdf')

# U_b,2 = 300 V
plt.clf()
params8 = ucurve_fit(reg_linear, B_2, D_lang/(L**2+D_lang**2))             # linearer Fit
m8, b8 = params8
print("https://www.youtube.com/watch?v=Mdi534Q1Zsg")
write('build/parameter_m8.tex', make_SI(m8, r'\per\metre\per\tesla', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b8.tex', make_SI(b8, r'\per\metre', figures=2))      # type in Anz. signifikanter Stellen
t_plot8 = np.linspace(np.amin(B_2), np.amax(B_2), 100)
plt.plot(t_plot8*10**(6), (m8.n*t_plot8+b8.n), 'b-', label='Linearer Fit')
plt.plot(B_2*10**(6), D_lang/(L**2+D_lang**2), 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{L^2 + D^2} \:/\: \si{\per\metre} $')
plt.xlabel(r'$B_2 \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a8.pdf')

# U_b,2 = 350 V
plt.clf()
params9 = ucurve_fit(reg_linear, B_3, D_lang/(L**2+D_lang**2))             # linearer Fit
m9, b9 = params9
write('build/parameter_m9.tex', make_SI(m9, r'\per\metre\per\tesla', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b9.tex', make_SI(b9, r'\per\metre', figures=2))      # type in Anz. signifikanter Stellen
t_plot9 = np.linspace(np.amin(B_3), np.amax(B_3), 100)
plt.plot(t_plot9*10**(6), (m9.n*t_plot9+b9.n), 'b-', label='Linearer Fit')
plt.plot(B_3*10**(6), D_lang/(L**2+D_lang**2), 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{L^2 + D^2} \:/\: \si{\per\metre} $')
plt.xlabel(r'$B_3 \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a9.pdf')

# U_b,2 = 400 V
plt.clf()
params10 = ucurve_fit(reg_linear, B_4, D_kurz/(L**2+D_kurz**2))             # linearer Fit
m10, b10 = params10
write('build/parameter_m10.tex', make_SI(m10, r'\per\metre\per\tesla', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b10.tex', make_SI(b10, r'\per\metre', figures=2))      # type in Anz. signifikanter Stellen
t_plot10 = np.linspace(np.amin(B_4), np.amax(B_4), 100)
plt.plot(t_plot10*10**(6), (m10.n*t_plot10+b10.n), 'b-', label='Linearer Fit')
plt.plot(B_4*10**(6), D_kurz/(L**2+D_kurz**2), 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{L^2 + D^2} \:/\: \si{\per\metre} $')
plt.xlabel(r'$B_4 \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a10.pdf')

# U_b,2 = 450 V
plt.clf()
params11 = ucurve_fit(reg_linear, B_5, D_kurz/(L**2+D_kurz**2))             # linearer Fit
m11, b11 = params11
write('build/parameter_m11.tex', make_SI(m11, r'\per\metre\per\tesla', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b11.tex', make_SI(b11, r'\per\metre', figures=2))      # type in Anz. signifikanter Stellen
t_plot11 = np.linspace(np.amin(B_5), np.amax(B_5), 100)
plt.plot(t_plot11*10**(6), (m11.n*t_plot11+b11.n), 'b-', label='Linearer Fit')
plt.plot(B_5*10**(6), D_kurz/(L**2+D_kurz**2), 'rx', label='Messdaten')
#plt.xlim(t_plot1[0], t_plot1[-1])
plt.ylabel(r'$\frac{D}{L^2 + D^2} \:/\: \si{\per\metre} $')
plt.xlabel(r'$B_5 \:/\: \si{\micro\tesla}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a11.pdf')


die_bs_b = np.array([b7.n, b8.n, b9.n, b10.n, b11.n])
die_bs_err_b = np.array([b7.s, b8.s, b9.s, b10.s, b11.s])
die_ms_b = np.array([m7.n, m8.n, m9.n, m10.n, m11.n])
die_ms_err_b = np.array([m7.s, m8.s, m9.s, m10.s, m11.s])

write('Tabelle_e.tex', make_table([U_b_B, die_ms_b, die_ms_err_b, die_bs_b, die_bs_err_b],[0, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('Tabelle_e_texformat.tex', make_full_table(
    'Fitparameter: Steigung $m$ und y-Achsenabschnitt $b$.',
    'tab:e',
    'Tabelle_e.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$U_b \:/\: \si{\volt}$',
    r'$m \:/\:\si{\per\metre\per\tesla}$',
    r'$\increment{m} \:/\:\si{\per\metre\per\tesla}$',
    r'$b \:/\: \si{\per\metre}$',
    r'$\increment{b} \:/\:\si{\per\metre}$']))

steigungen = unp.uarray(die_ms_b, die_ms_err_b)
konstante = 8*U_b_B*steigungen**2
write('build/konstante_0.tex', make_SI(konstante[0]*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen
write('build/konstante_1.tex', make_SI(konstante[1]*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen
write('build/konstante_2.tex', make_SI(konstante[2]*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen
write('build/konstante_3.tex', make_SI(konstante[3]*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen
write('build/konstante_4.tex', make_SI(konstante[4]*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen

mean_k = np.mean(noms(konstante))
std_k = np.std(noms(konstante))
k = ufloat(mean_k, std_k)
write('build/konstante_mean.tex', make_SI(k*10**(-11), r'\coulomb\per\kilogram','e11', figures=2))      # type in Anz. signifikanter Stellen


## Errrdmagnetfeld

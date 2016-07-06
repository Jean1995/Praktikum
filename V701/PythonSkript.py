##################################################### Import system libraries ######################################################
import matplotlib as mpl
mpl.rcdefaults()
mpl.rcParams.update(mpl.rc_params_from_file('meine-matplotlibrc'))
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.mlab as mlab
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from scipy.misc import factorial
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

from scipy.optimize import curve_fit
from scipy import stats
from  matplotlib import pyplot as plt
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


########## CURVE FIT ##########from scipy.optimize import curve_fit
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


### Aufgabe a ###

def x_eff(x_0,p):
    """Rueckgabe der effektiven Länge
    Args:
        x_0: Abstand Detektor Quelle (m)
        p: Druck in Glaszylinder (bar)
    Returns:
        Effektive Länge in meter
    """
    return x_0 * p/(1013*10**(-3))

p_1, pulse_1, channel_1 = np.genfromtxt('messdaten/messung_1.txt', unpack=True)
p_1 = p_1/1000 # in bar
x_1 =  0.025 # Abstand Detektor Quelle für 1. Messung
ikse_eff = x_eff(x_1, p_1)
print(type(p_1))
write('build/tabulatore.tex', make_table([p_1*1000, ikse_eff*100, pulse_1, channel_1],[0, 2, 0, 0]))
write('build/tabulatore_texformat.tex', make_full_table(
    'Messdaten, erster Teil.',
    'tabulatore',
    'build/tabulatore.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                            # die Multicolumns sein sollen
    [
    r'$p \:/\: \si{\milli\bar}$',
    r'$x_{\text{eff}} \:/\: \si{\centi\metre}$',
    r'$\text{Impulse}$',
    r'$\text{Channel}$']))

p_mittel = 0.8 # druck wo halbiert wird

k = np.array([725, 750, 775, 800, 825, 850, 875])
k = k/1000
l = np.array([35579, 33246, 29562, 22986, 17301, 11904, 5620])
params = ucurve_fit(reg_linear, x_eff(x_1, k), l)
t_plot = np.linspace(np.amin(x_eff(x_1, k)), np.amax(x_eff(x_1, k)), 100)
o,i = params
plt.plot(t_plot, t_plot*o.n+i.n, 'y-', label='Linearer Fit')

h = ((23000-i)/o)
print(h)

plt.plot(x_eff(x_1, p_1), pulse_1, 'rx', label='Messdaten')
#plt.xlim(-0.01, 0.025)
#plt.ylim(-1000, 50000)
plt.axvline(x=noms(h), color='g', linestyle='--')
plt.axhline(y=46000, color='b', linestyle='--')
plt.axhline(y=23000, color='b', linestyle='--')
plt.xlabel(r'$x_\text{eff} \:/\: \si{\metre}$')
plt.ylabel(r'$\text{Impulse} $')
plt.legend(loc='best')
plt.grid()

#plt.axhline(0.019, color='g', linestyle='--')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a_1.pdf')

E_mit_1 = 406/1023 * 4 # Lineare Skala: Bei Channel 1023 ist das Maximum von 4 MeV, bei Channel 406 ist der mittlere Wert

write('build/x_mittel_1.tex', make_SI(h*100, r'\centi\metre', figures=1))
#write('build/E_mittel_1.tex', make_SI(E_mit_1, r'\mega\electronvolt', figures=3))


plt.clf()
plt.xlim(0, 0.025)
plt.plot(x_eff(x_1, p_1[0:29]), (channel_1[0:29]/1023)*4, 'rx', label='Messdaten')
#plt.plot(x_eff(x_1, p_1[0:1]), (channel_1[0:1]/1023)*4, 'rx', label='Messdaten')
plt.plot(x_eff(x_1, p_1[2:16]), (channel_1[2:16]/1023)*4, 'bx', label='Messdaten für linearen Fit')
#plt.plot(x_eff(x_1, p_1[19:29]), (channel_1[19:29]/1023)*4, 'rx', label='Messdaten')

params_1_fit = ucurve_fit(reg_linear, x_eff(x_1, p_1[2:16]), (channel_1[2:16]/1023)*4)             # linearer Fit
a1, b1 = params_1_fit
write('build/parameter_a_1.tex', make_SI(a1, r'\mega\electronvolt\per\metre', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b_1.tex', make_SI(b1, r'\mega\electronvolt', figures=2))      # type in Anz. signifikanter Stellen

E_mit_1 = a1*h/100+b1
write('build/E_mittel_1.tex', make_SI(E_mit_1, r'\mega\electronvolt', figures=1))

t_plot = np.linspace(0, 0.025)
plt.plot(t_plot, b1.n + t_plot*a1.n, 'b-', label='Linearer Fit')

#plt.xlim(-0.01, 0.025)
#plt.ylim(-1000, 50000)
plt.xlabel(r'$x_\text{eff} \:/\: \si{\metre}$')
plt.ylabel(r'$E \:/\: \si{\mega\electronvolt} $')
plt.legend(loc='best')
plt.grid()

#plt.axhline(0.019, color='g', linestyle='--')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a_1_1.pdf')

#### Aufgabe a_2 ###

plt.clf()

p_2, pulse_2, channel_2 = np.genfromtxt('messdaten/messung_2.txt', unpack=True)
p_2 = p_2/1000 # in bar
x_2 =  0.015 # Abstand Detektor Quelle für 1. Messung

ikse_eff2 = x_eff(x_2, p_2)
write('build/tabulatore2.tex', make_table([p_2*1000, ikse_eff2*100, pulse_2, channel_2],[0, 2, 0, 0]))
write('build/tabulatore2_texformat.tex', make_full_table(
    'Messdaten, zweiter Teil.',
    'tabulatore2',
    'build/tabulatore2.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                            # die Multicolumns sein sollen
    [
    r'$p \:/\: \si{\milli\bar}$',
    r'$x_{\text{eff}} \:/\: \si{\centi\metre}$',
    r'$\text{Impulse}$',
    r'$\text{Channel}$']))


p_mittel = 0.8 # druck wo halbiert wird

plt.plot(x_eff(x_2, p_2), pulse_2, 'rx', label='Messdaten')
#plt.xlim(-0.01, 0.025)
#plt.ylim(-1000, 50000)
#plt.axvline(x=x_eff(x_2,p_mittel), color='g', linestyle='--')
#plt.axhline(y=46500, color='b', linestyle='--')
#plt.axhline(y=23000, color='b', linestyle='--')
plt.xlabel(r'$x_\text{eff} \:/\: \si{\metre}$')
plt.ylabel(r'$\text{Impulse} $')
plt.legend(loc='best')
plt.grid()

#plt.axhline(0.019, color='g', linestyle='--')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a_2.pdf')

plt.clf()

plt.plot(x_eff(x_2, p_2), (channel_2/1023)*4, 'rx', label='Messdaten')
plt.plot(x_eff(x_2, p_2[2:19]), (channel_2[2:19]/1023)*4, 'bx', label='Messdaten für linearen Fit')
plt.xlim(0, 0.016)
#plt.ylim(-1000, 50000)
plt.xlabel(r'$x_\text{eff} \:/\: \si{\metre}$')
plt.ylabel(r'$E \:/\: \si{\mega\electronvolt} $')


params_2_fit = ucurve_fit(reg_linear, x_eff(x_2, p_2[2:19]), (channel_2[2:19]/1023)*4)             # linearer Fit
a2, b2 = params_2_fit
write('build/parameter_a_2.tex', make_SI(a2, r'\mega\electronvolt\per\metre', figures=1))       # type in Anz. signifikanter Stellen
write('build/parameter_b_2.tex', make_SI(b2, r'\mega\electronvolt', figures=2))      # type in Anz. signifikanter Stellen

t_plot = np.linspace(0, 0.016)
plt.plot(t_plot, b2.n + t_plot*a2.n, 'b-', label='Linearer Fit')

plt.legend(loc='best')
plt.grid()
#plt.axhline(0.019, color='g', linestyle='--')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_a_2_2.pdf')



#### Aufgabe 2 ( Statistik) ###

plt.clf()

nr, zaehlrate = np.genfromtxt('messdaten/messung_stat.txt', unpack=True)
#
#mu, sigma = np.mean(zaehlrate), np.std(zaehlrate)
#
#n, bins, patches = plt.hist(zaehlrate, 12, normed=1, facecolor='blue', alpha=0.75)
#
#y = mlab.normpdf( bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=1)
#
#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)

mu = np.mean(zaehlrate)
sigma = np.std(zaehlrate)
write('build/mu_stat.tex', make_SI(mu, r'', figures=2))
write('build/sigma_stat.tex', make_SI(sigma, r'', figures=2))


# the histogram of the data
n, bins, patches = plt.hist(zaehlrate, 10, normed=1, facecolor='blue', alpha=0.75, label="Messwerte")


def Gauss(x, mu, sigma):
     return 1/(sigma*np.sqrt(2*np.pi) ) * np.exp(-0.5*  ((x-mu)/sigma)**2 )

t_plot = np.linspace(9500, 12000, 1000)
plt.plot(t_plot, Gauss(t_plot, mu, sigma), 'r-', label='Gaußverteilung')


plt.xlabel(r'$\text{Zählrate}$')
plt.ylabel(r'$p$')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot_stat.pdf')


### Le Verteilung du Poisson (geliehen von einem Altprotokoll aus Ap_MaMa)

plt.clf()
nr, N = np.genfromtxt('messdaten/messung_stat.txt', unpack=True)
mu = np.mean(N)
sigma = np.std(N)

binnum = 10    #
n, low_range, binsize, extra = stats.histogram(N, binnum)
ind = np.arange(binnum)
width = 0.50
x = np.linspace(0, 10)   #

poisson = stats.poisson(5).pmf(ind)

plt.bar(ind+0.25, n/100., width, color="blue", label="Messwerte")

plt.plot(ind+0.5, poisson,  'rx', label="Poissonverteilung")

plt.ylabel(r'$p$')

plt.xticks(ind+width, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
plt.grid()
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.legend(loc='best')
plt.savefig("build/statistik.pdf")

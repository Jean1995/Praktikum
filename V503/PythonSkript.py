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



















import math

d = ufloat(7.6250*10**(-3),0.0051*10**(-3))
rho_luft = 1.1644 #wikipedia :P Luft 30 Grad
def f(U, n, t_auf, t_ab, d):
    f = 3*np.pi*n*10**(-5)*(9/4*n*10**(-5)/9.81*(0.001/t_ab-0.001/t_auf)/(886-1.1644))**(1/2)*(0.001/t_ab+0.001/t_auf)/(U/d)
    return f

def r(n, t_auf, t_ab):
    r = (9/4*n*10**(-5)/9.81*(0.001/t_ab-0.001/t_auf)/(886-1.1644))**(1/2)
    return r

#def err(n, t_auf, t_ab, del_t_auf, del_t_ab, d, d_err):
#    err = math.sqrt((3*math.pi*n*d*math.sqrt(9/4*n/g*(0.001)))**2+()**2+()**2)
#    return err

B = 6.17*10**(-5)
p = 1013.25*10**2
########## V = 190 R = 1.71 => T = 32 => n = 1.88
    #v_0 = 16.38
t_1_auf = np.array([5.81, 6.26])
t_1_ab  = np.array([4.13, 3.89])

write('build/Tabelle_1.tex', make_table([t_1_auf,t_1_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_1_texformat.tex', make_full_table(
    r'Messung 1 bei $U_1$, $T_1$.',
    'table:1',
    'build/Tabelle_1.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))


t_1_auf_mitt = ufloat(np.mean(t_1_auf), MeanError(noms(t_1_auf)))
print(t_1_auf_mitt)
t_1_ab_mitt =  ufloat(np.mean(t_1_ab),  MeanError(noms(t_1_ab)))
print(t_1_ab_mitt)
q_1 = f(190, 1.88, t_1_auf_mitt, t_1_ab_mitt, d)
print(q_1)
r_1 = r(1.88, t_1_auf_mitt, t_1_ab_mitt)
print(r_1)
q_1_neu = q_1*(1+B/(p*r_1))**(-3/2)
print(q_1_neu)
#write('build/q_1.tex', make_SI(unp.nominal_values(q_1) * 10**19, r'\coulomb', figures=5))
    #v_0 = 17.3
t_2_auf = np.array([15.81, 16.1])
t_2_ab  = np.array([7.64, 8.18])

write('build/Tabelle_2.tex', make_table([t_2_auf,t_2_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_2_texformat.tex', make_full_table(
    r'Messung 2 bei $U_1$, $T_1$.',
    'table:2',
    'build/Tabelle_2.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))

t_2_auf_mitt = ufloat(np.mean(t_2_auf), MeanError(noms(t_2_auf)))
t_2_ab_mitt =  ufloat(np.mean(t_2_ab),  MeanError(noms(t_2_ab)))
q_2 = f(190, 1.88, t_2_auf_mitt, t_2_ab_mitt, d)
print(q_2)
r_2 = r(1.88, t_2_auf_mitt, t_2_ab_mitt)
q_2_neu = q_2*(1+B/(p*r_2))**(-3/2)
    #v_0 = 11.93
t_3_auf = np.array([6.84, 6.64])
t_3_ab  = np.array([4.3, 4.89])

write('build/Tabelle_3.tex', make_table([t_3_auf,t_3_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_3_texformat.tex', make_full_table(
    r'Messung 3 bei $T_1$, $U_1$.',
    'table:3',
    'build/Tabelle_3.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))

t_3_auf_mitt = ufloat(np.mean(t_3_auf), MeanError(noms(t_3_auf)))
t_3_ab_mitt =  ufloat(np.mean(t_3_ab),  MeanError(noms(t_3_ab)))
q_3 = f(190, 1.88, t_3_auf_mitt, t_3_ab_mitt, d)
print(q_3)
r_3 = r(1.88, t_3_auf_mitt, t_3_ab_mitt)
q_3_neu = q_3*(1+B/(p*r_3))**(-3/2)
########## V = 302 R = 1.71 => T = 32 => n = 1.88
    #v_0 = 15.56
t_4_auf = np.array([6.15, 6.24, 5.41])
t_4_ab  = np.array([4.35, 4.07, 4.64])

write('build/Tabelle_4.tex', make_table([t_4_auf,t_4_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_4_texformat.tex', make_full_table(
    r'Messung 4 bei $U_2$, $T_1$.',
    'table:4',
    'build/Tabelle_4.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))

t_4_auf_mitt = ufloat(np.mean(t_4_auf), MeanError(noms(t_4_auf)))
t_4_ab_mitt =  ufloat(np.mean(t_4_ab),  MeanError(noms(t_4_ab)))
q_4 = f(302, 1.88, t_4_auf_mitt, t_4_ab_mitt, d)
r_4 = r(1.88, t_4_auf_mitt, t_4_ab_mitt)
print(r_4)
q_4_neu = q_4*(1+B/(p*r_4))**(-3/2)
    #v_0 = 14.83
t_5_auf = np.array([9.12, 9.12])
t_5_ab  = np.array([5.46, 6.83])

write('build/Tabelle_5.tex', make_table([t_5_auf,t_5_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_5_texformat.tex', make_full_table(
    r'Messung 5 bei $U_2$, $T_1$.',
    'table:5',
    'build/Tabelle_5.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))

t_5_auf_mitt = ufloat(np.mean(t_5_auf), MeanError(noms(t_5_auf)))
t_5_ab_mitt =  ufloat(np.mean(t_5_ab),  MeanError(noms(t_5_ab)))
q_5 = f(302, 1.88, t_5_auf_mitt, t_5_ab_mitt, d)
r_5 = r(1.88, t_5_auf_mitt, t_5_ab_mitt)
print(r_5)
q_5_neu = q_5*(1+B/(p*r_5))**(-3/2)
########## V = 250 R = 1.67 => T = 33 => n = 1.885
    #v_0 = 14.47
t_6_auf = np.array([3.83, 3.29, 3.44, 3.46])
t_6_ab  = np.array([2.58, 2.87, 2.95, 2.87])

write('build/Tabelle_6.tex', make_table([t_6_auf,t_6_ab],[2,2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_6_texformat.tex', make_full_table(
    r'Messung 6 bei $U_3$, $T_2$.',
    'table:6',
    'build/Tabelle_6.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$t_{\text{auf}} \:/\: \si{\second}$',
    r'$t_{\text{ab}} \:/\: \si{\second}$']))

t_6_auf_mitt = ufloat(np.mean(t_6_auf), MeanError(noms(t_6_auf)))
t_6_ab_mitt =  ufloat(np.mean(t_6_ab),  MeanError(noms(t_6_ab)))
q_6 = f(250, 1.885, t_6_auf_mitt, t_6_ab_mitt, d)
r_6 = r(1.885, t_6_auf_mitt, t_6_ab_mitt)
print(r_6)
q_6_neu = q_6*(1+B/(p*r_6))**(-3/2)

n = np.array([1,2,3,4,5,6])
q = np.array([q_1,q_2,q_3,q_4,q_5,q_6])

write('build/Tabelle_q.tex', make_table([n,unp.nominal_values(q)*10**19, unp.std_devs(q)*10**19],[0,3,3]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_q_texformat.tex', make_full_table(
    r'Berechnete Ladungen aus den ausgewählten Geschwindigkeiten $v_\text{ab}$ und $v_\text{auf}$.',
    'table:q',
    'build/Tabelle_q.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Messung}$',
    r'$q \:/\: 10^{-19}\si{\coulomb}$',
    r'$\increment{q} \:/\: 10^{-19}\si{\coulomb}$']))

plt.xlim(0.5, 6.5)
#plt.plot(n, unp.nominal_values(q)*10**(19), 'rx', label='Messdaten')
plt.errorbar(n, unp.nominal_values(q)*10**18, fmt='rx', yerr=unp.std_devs(q)*10**18, label='Messdaten')
plt.xlabel(r'$\text{Messung}$')
plt.ylabel(r'$q \:/\: 10^{-18}\si{\coulomb}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ladungen.pdf')


q_neu = np.array([q_1_neu,q_2_neu,q_3_neu,q_4_neu,q_5_neu,q_6_neu])

write('build/Tabelle_q_neu.tex', make_table([n,unp.nominal_values(q_neu)*10**19, unp.std_devs(q_neu)*10**19],[0,3,3]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_q_neu_texformat.tex', make_full_table(
    'korrigierte Ladungen.',
    'table:q_neu',
    'build/Tabelle_q_neu.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Messung}$',
    r'$q \:/\: 10^{-19\:/\:}\si{\coulomb}$',
    r'$\increment{q} \:/\: 10^{-19}\si{\coulomb}$']))

plt.clf()
plt.xlim(0.5, 6.5)
plt.errorbar(n, unp.nominal_values(q_neu)*10**18, fmt='rx', yerr=unp.std_devs(q_neu)*10**18, label='korrigierte Messdaten')
plt.xlabel(r'$\text{Messung}$')
plt.ylabel(r'$q \:/\: 10^{-18}\si{\coulomb}$')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/ladungen_neu.pdf')

#### All hail AP_MaMa:
def GCD(q,maxi):
    gcd=q[0]
    for i in range(1,len(q)):
        n=0
        while abs(gcd-q[i])>1e-19 and n <= maxi:
            if gcd > q[i]:
                gcd = gcd - q[i]
            else:
                q[i] = q[i] - gcd
            n = n+1
    return gcd

e_0 = GCD(q,30)
e_0 = unp.nominal_values(e_0)
write('build/e_0.tex', make_SI(e_0 * 10**(19), r'\coulomb','e-19', figures=3))
print(e_0)
e_rel = abs(e_0-1.6021766208*10**(-19))/(1.6021766208*10**(-19))*100
write('build/e_rel.tex', make_SI(e_rel, r'\percent', figures=2))
e_0_neu = GCD(q_neu,30)
e_0_neu = unp.nominal_values(e_0_neu)
write('build/e_0_neu.tex', make_SI(e_0_neu * 10**(19), r'\coulomb','e-19', figures=3))
e_neu_rel = abs(e_0_neu-1.6021766208*10**(-19))/(1.6021766208*10**(-19))*100
write('build/e_neu_rel.tex', make_SI(e_neu_rel, r'\percent', figures=2))

Faraday = 96485.3365 #https://de.wikipedia.org/wiki/Faraday-Konstante
N_a_lit = 6.022140857*10**23 #https://de.wikipedia.org/wiki/Avogadro-Konstante
N_a = Faraday/e_0
write('build/N_a.tex', make_SI(N_a * 10**(-23), r'\per\mol','e23', figures=3))
N_a_rel = abs(N_a-N_a_lit)/N_a_lit *100
write('build/N_a_rel.tex', make_SI(N_a_rel, r'\percent', figures=2))
N_a_neu = Faraday/e_0_neu
write('build/N_a_neu.tex', make_SI(N_a_neu * 10**(-23), r'\per\mol','e23', figures=3))
N_a_neu_rel = abs(N_a_neu-N_a_lit)/N_a_lit *100
write('build/N_a_neu_rel.tex', make_SI(N_a_neu_rel, r'\percent', figures=2))

r = np.array([r_1,r_2,r_3,r_4,r_5,r_6])

write('build/Tabelle_r.tex', make_table([n,unp.nominal_values(r)*10**7, unp.std_devs(r)*10**7],[0,3,3]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_r_texformat.tex', make_full_table(
    'Radien der Öltröpfchen.',
    'tabella',
    'build/Tabelle_r.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Messung}$',
    r'$r \:/\: 10^{-7}\si{\metre}$',
    r'$\increment{r} \:/\: 10^{-7}\si{\metre}$']))

e_0 = np.array([e_0, e_0_neu])
e_0_rel = np.array([e_rel, e_neu_rel])
N_a = np.array([N_a,N_a_neu])
N_a_rel = np.array([N_a_rel, N_a_neu_rel])
x = np.array([1,2])
write('build/Tabelle_Ende.tex', make_table([x,e_0*10**19, e_0_rel, N_a*10**(-23), N_a_rel],[0,3,0,3,0]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
write('build/Tabelle_Ende_texformat.tex', make_full_table(
    'Ergebnisse des Versuchs.',
    'tab:Ende',
    'build/Tabelle_Ende.tex',
    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
                              # die Multicolumns sein sollen
    [
    r'$\text{Methode}$',
    r'$e_0 \:/\: 10^{-19}\si{\coulomb}$',
    r'$\increment{e_0} \:/\: \si{\percent}$',
    r'$N_a \:/\: 10^{23}\si{\per\mol}$',
    r'$\increment{N_a} \:/\: \si{\percent}$']))




#http://www.chemie.de/lexikon/Faraday-Konstante.html
#http://www.chemie.de/lexikon/Elementarladung.html
#https://elearning.physik.uni-frankfurt.de/data/FB13-PhysikOnline/lm_data/lm_324/daten/kap_17/node31.htm
#http://www.chemie.de/lexikon/Avogadro-Konstante.html

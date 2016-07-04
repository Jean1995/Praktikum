# Überführt Messdaten in auslesbare Textdaten (optional)
# Überführt Messdaten in auslesbare Textdaten (optional)
import numpy as np


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








# B-Feld-Aufabe

D = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]) # in Zoll! Obacht!
I_1 = np.array([0, 0.28, 0.58, 0.89, 1.19, 1.52, 1.84, 2.18, 2.49]) # 250 V
I_2 = np.array([0, 0.30, 0.64, 1.00, 1.34, 1.68, 2.2, 2.39, 2.73]) # 300 V
I_3 = np.array([0, 0.32, 0.69, 1.07, 1.44, 1.81, 2.2, 2.58, 2.95]) # 350 V
D_kurz = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 0]) # in Zoll! Obacht!
I_4 = np.array([0, 0.35, 0.74, 1.14, 1.56, 1.94, 2.34, 2.75, 0]) # 400 V
I_5 = np.array([0, 0.4, 0.81, 1.25, 1.66, 2.09, 1.51, 2.94, 0]) # 450 V

np.savetxt('messdaten/messung_B_lang.txt', np.column_stack([D, I_1, I_2, I_3]), header="D [Zoll], I_1 [A], I_2 [A], I_3 [A]")
np.savetxt('messdaten/messung_B_kurz.txt', np.column_stack([D_kurz, I_4, I_5]), header="D [Zoll], I_4 [A], I_5 [A]")

mu_0 = 4*np.pi*10**(-7)
N = 20 #? geraten
R = 0.282 # ? geraten
L = 17.5*0.01
B_1 = mu_0 * 8/np.sqrt(125) * N/R * I_1
B_2 = mu_0 * 8/np.sqrt(125) * N/R * I_2
B_3 = mu_0 * 8/np.sqrt(125) * N/R * I_3
B_4 = mu_0 * 8/np.sqrt(125) * N/R * I_4
B_5 = mu_0 * 8/np.sqrt(125) * N/R * I_5
#write('Tabelle_d.tex', make_table([D,B_1*10**6,B_2*10**6,B_3*10**6,B_4*10**6,B_5*10**6],[2, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
#write('Tabelle_d_texformat.tex', make_full_table(
#    'B-Feldstärken.',
#    'tab:d',
#    'Tabelle_d.tex',
#    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                              # die Multicolumns sein sollen
#    [
#    r'$D \:/\: \text{in}$',
#    r'$B_1 \:/\: \si{\micro\tesla}$',
#    r'$B_2 \:/\: \si{\micro\tesla}$',
#    r'$B_3 \:/\: \si{\micro\tesla}$',
#    r'$B_4 \:/\: \si{\micro\tesla}$',
#    r'$B_5 \:/\: \si{\micro\tesla}$']))


#write('Tabelle_a.tex', make_table([D,I_1,I_2,I_3,I_4,I_5],[2, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
#write('Tabelle_a_texformat.tex', make_full_table(
#    'Messdaten zur Bestimmung der Ablenkung im B-Feld.',
#    'table:A2',
#    'Tabelle_a.tex',
#    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                              # die Multicolumns sein sollen
#    [
#    r'$D \:/\: \text{in}$',
#    r'$I_1 \:/\: \si{\ampere}$',
#    r'$I_2 \:/\: \si{\ampere}$',
#    r'$I_3 \:/\: \si{\ampere}$',
#    r'$I_4 \:/\: \si{\ampere}$',
#    r'$I_5 \:/\: \si{\ampere}$']))







# Erdmangetfeld
I_erd = 0.26 # Kompensationsstrom in Ampere
U_erd = 200 # Beschleunigungsspannung
phi = 70 #grad inklinationswinkel

# E-Feld-Aufgabe

D = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]) # in Zoll! Obacht!
U_1 = np.array([-8.55, -4.58, -1.1, 2.61, 6.18, 9.59, 13.26, 16.8, 20.4]) # 200 V
U_2 = np.array([-11.09, -6.36, -1.87, 2.91, 7.59, 12, 16.6, 20.98, 25.35]) # 250 V
U_3 = np.array([-13.35, -7.82, -1.96, 3.79, 8.92, 14.64, 20.09, 25.31, 30.81]) # 300 V
D_kurz = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]) # in Zoll! Obacht!
U_4 = np.array([-14.23, -7.67, -1.06, 5.38, 11.67, 17.93, 24.32, 30.78]) # 350 V
U_5 = np.array([-17.43, -10.13, -2.4, 5.17, 12.58, 19.83, 27, 34.23]) # 400 V

np.savetxt('messdaten/messung_E_lang.txt', np.column_stack([D, U_1, U_2, U_3]), header="D [Zoll], U_1 [V], U_2 [V], U_3 [V]")
np.savetxt('messdaten/messung_E_kurz.txt', np.column_stack([D_kurz, U_4, U_5]), header="D [Zoll], U_4 [V], U_5 [V]")


#write('Tabelle_b.tex', make_table([D,U_1,U_2,U_3,U_4,U_5],[2, 2, 2, 2, 2, 2]))     # Jeder fehlerbehaftete Wert bekommt zwei Spalten
#write('Tabelle_b_texformat.tex', make_full_table(
#    'Messdaten zur Bestimmung der Ablenkung im E-Feld.',
#    'table:b',
#    'Tabelle_b.tex',
#    [],              # Hier aufpassen: diese Zahlen bezeichnen diejenigen resultierenden Spaltennummern,
#                              # die Multicolumns sein sollen
#    [
#    r'$D \:/\: \text{in}$',
#    r'$U_1 \:/\: \si{\volt}$',
#    r'$U_2 \:/\: \si{\volt}$',
#    r'$U_3 \:/\: \si{\volt}$',
#    r'$U_4 \:/\: \si{\volt}$',
#    r'$U_5 \:/\: \si{\volt}$']))



# Frequenzkacke

v = np.array([39.74, 79.31, 129.3, 158.46]) # in... kHz (cant remember)
A = np.array([0.25, 9/32, 7/16, 5/16]) # in zöll
np.savetxt('messdaten/frequenzen.txt', np.column_stack([v, A]), header="v [kHz ?], A [zöll]")

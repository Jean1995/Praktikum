# Überführt Messdaten in auslesbare Textdaten (optional)

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

########## V = 190 R = 171
    #v_0 = 16.38
t_1_auf = np.array([5.81, 6.26])
t_1_ab  = np.array([4.13, 3.89])
t_1_auf_mitt = ufloat(np.mean(t_1_auf), MeanError(noms(t_1_auf)))
t_1_ab_mitt =  ufloat(np.mean(t_1_ab),  MeanError(noms(t_1_ab)))
    #v_0 = 17.3
t_2_auf = np.array([15.81, 16.1])
t_2_ab  = np.array([7.64, 8.18])
t_2_auf_mitt = ufloat(np.mean(t_2_auf), MeanError(noms(t_2_auf)))
t_2_ab_mitt =  ufloat(np.mean(t_2_ab),  MeanError(noms(t_2_ab)))
    #v_0 = 11.93
t_3_auf = np.array([6.84, 6.64])
t_3_ab  = np.array([4.3, 4.89])
t_3_auf_mitt = ufloat(np.mean(t_3_auf), MeanError(noms(t_3_auf)))
t_3_ab_mitt =  ufloat(np.mean(t_3_ab),  MeanError(noms(t_3_ab)))
########## V = 302 R = 171
    #v_0 = 15.56
t_4_auf = np.array([6.15, 6.24, 5.41])
t_4_ab  = np.array([4.35, 4.07, 4.64])
t_4_auf_mitt = ufloat(np.mean(t_4_auf), MeanError(noms(t_4_auf)))
t_4_ab_mitt =  ufloat(np.mean(t_4_ab),  MeanError(noms(t_4_ab)))
    #v_0 = 14.83
t_5_auf = np.array([9.12, 9.12])
t_5_ab  = np.array([5.46, 6.83])
t_5_auf_mitt = ufloat(np.mean(t_5_auf), MeanError(noms(t_5_auf)))
t_5_ab_mitt =  ufloat(np.mean(t_5_ab),  MeanError(noms(t_5_ab)))
########## V = 250 R = 167
    #v_0 = 14.47
t_6_auf = np.array([3.83, 3.29, 3.44, 3.46])
t_6_ab  = np.array([2.58, 2.87, 2.95, 2.87])
t_6_auf_mitt = ufloat(np.mean(t_6_auf), MeanError(noms(t_6_auf)))
t_6_ab_mitt =  ufloat(np.mean(t_6_ab),  MeanError(noms(t_6_ab)))

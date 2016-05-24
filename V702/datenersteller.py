# Überführt Messdaten in auslesbare Textdaten (optional)
import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

 # use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"python_custom_scripts")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from table import (
    make_table,
    make_SI,
    write,
)
from uncertainties import ufloat




#############Indium############

Indium = np.array([2085, 1867, 1768, 1676, 1544, 1521, 1388, 1404, 1373, 1234, 1273, 1187, 1095, 1054, 991, 972, 944])
np.savetxt('messdaten/Indium.txt', np.column_stack([Indium]), header="Impulse /220s")

############Rhodium############

Rhodium = np.array([642, 510, 431, 353, 255, 217, 191, 113, 104, 93, 95, 63, 72, 39, 32, 31, 34, 40, 28, 36, 33, 33, 22, 22, 21, 25, 17, 22, 24, 14, 22, 20, 8, 18, 18, 17, 19, 12, 15, 15, 10, 13, 15])
np.savetxt('messdaten/Rhodium.txt', np.column_stack([Rhodium]), header="Impulse /17s")

Rhodium2 = np.array([642, 510, 431, 353, 255, 217, 191, 113, 104, 93, 95, 63, 72, 39, 32, 31, 34])
np.savetxt('messdaten/Rhodium2.txt', np.column_stack([Rhodium2]), header="Impulse /17s")

Rhodium1 = np.array([40, 28, 36, 33, 33, 22, 22, 21, 25, 17, 22, 24, 14, 22, 20, 8, 18, 18, 17, 19, 12, 15, 15, 10, 13, 15])
np.savetxt('messdaten/Rhodium1.txt', np.column_stack([Rhodium1]), header="Impulse /17s")

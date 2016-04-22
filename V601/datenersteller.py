# Überführt Messdaten in auslesbare Textdaten (optional)
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

########## Aufgabenteil 0 (Temperaturen) ##########

T = np.array([26.1, 145.5, 161, 178, 106.6]) # Temperaturen in Celsius
np.savetxt('messdaten/0.txt', np.column_stack([T]), header="T [Celsius]")



########## Aufgabenteil 0 (Fucking Energieverteilung Bra) ##########

# Für T=26.1 C

del_U_a = 0.0394 #Ein Kasten in x-Richtung entspricht in etwa 0.0394V, wähle das als Delta
U_a = np.array([0*0.394, 3*0.394, 6*0.394, 9*0.394, 12*0.394, 15*0.394, 18*0.394 , 19*0.394, 20*0.394,  21*0.394, 24*0.394]) # Messe für jedes 3. Kasten a 0.394 Volt

I_a = np.array([1000-0*6.67, 1000-12*6.67, 1000-23*6.67, 1000-35*6.67, 1000-48*6.67, 1000-67*6.67, 1000-91*6.67, 1000-103*6.67, 1000-122*6.67,  1000-144*6.67, 0]) # Stromwert an i-ter Stelle
I_a_plus_delta = np.array([4*6.67, 4*6.67, 4*6.67, 4.5*6.67, 5.5*6.67, 7*6.67, 11*6.67,  12.5*6.67, 23*6.67, 4.5*6.67, 0      ]) # Stromwert an i-ter Stelle plus ein delta_U_a

# ERLÄUTERUNG: U_a ist das Array mit den Spannungen, an denen gemessen wurde
#               I_a ist der jeweilige Stromwert dort
#               I_a_plus_delta ist der Stromwert jeweils ein Delta weiter
#               (Die komischen Multiplikationen folgern aus der Ablesetechnik)

np.savetxt('messdaten/a_1.txt', np.column_stack([U_a, I_a, I_a_plus_delta, I_a - I_a_plus_delta]), header="U_a [Volt], I_a [nA], I_a_plus_delta [nA], Differenz [nA]")

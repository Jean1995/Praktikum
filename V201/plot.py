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

#################

# Werte und Tabelle
m_w, m_k, U_w, T_w, U_k, T_k, U_m, T_m = np.genfromtxt('daten.txt', unpack=True)
write('build/tabelle.tex', make_table([m_w, m_k, U_w, T_w, U_k, T_k, U_m, T_m], [2, 2, 2, 2, 2, 2, 2, 2]))

cg_mg = 224.84 # Joule pro Kelvin
cw_mw = 4.18 # Joule pro Kelvin pro Gramm

ck_0 = ( (cw_mw + cg_mg) * (T_m[0] - T_w[0]) ) / ( m_k[0] * (T_k[0] - T_m[0]) )
ck_1 = ( (cw_mw + cg_mg) * (T_m[1] - T_w[1]) ) / ( m_k[1] * (T_k[1] - T_m[1]) )
ck_2 = ( (cw_mw + cg_mg) * (T_m[2] - T_w[2]) ) / ( m_k[2] * (T_k[2] - T_m[2]) )
ck_3 = ( (cw_mw + cg_mg) * (T_m[3] - T_w[3]) ) / ( m_k[3] * (T_k[3] - T_m[3]) )
ck_4 = ( (cw_mw + cg_mg) * (T_m[4] - T_w[4]) ) / ( m_k[4] * (T_k[4] - T_m[4]) )

write('build/ck_blei.tex', str(ck_0))
write('build/ck_zinn1.tex', str(ck_1))
write('build/ck_zinn2.tex', str(ck_2))
write('build/ck_zinn3.tex', str(ck_3))
write('build/ck_graphit.tex', str(ck_4))
#





# Beispielwerte


c = ufloat(24601, 42)
write('build/wert_a.tex', make_SI(c*1e3, r'\joule\per\kelvin\per\gram' ))

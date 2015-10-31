#import matplotlib.pyplot as plt
#import numpy as np
#from uncertainties import ufloat
#import uncertainties.unumpy as unp
#
#d, Uabs = np.genfromtxt('daten2.txt', unpack=True)
#mb, mb_err = np.genfromtxt('Uabs_Ausgleichswerte.txt', unpack=True)
#Uabs = Uabs-Uoffset
#
#plt.plot(d, Uabs,'xr', label=r'$U_d$')
#
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#
#from scipy.optimize import curve_fit
#
#def f(x, a, b):
#    return a*x**(-2) + b


#Den x² Zusammenhang mittels ln darzustellen haut wegen des + b nicht hin :(

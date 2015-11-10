import numpy as np

#e)
v = np.array([20, 200, 400, 600, 800, 900, 950, 1000, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1300, 1400, 1500, 1700, 1900, 2100, 2300, 2500, 3000, 3500, 4000, 5000, 7000, 10000, 20000, 30000])
U = np.array([2.5, 2.18, 1.6, 1.04, 0.536, 0.416, 0.302, 0.170, 0.0984, 0.0864, 0.0336, 0.0104, 0.0152, 0.0776, 0.12, 0.25, 0.34, 0.44, 0.6, 0.776, 0.888, 0.992, 1.08, 1.27, 1.41, 1.51, 1.62, 1.8, 1.76, 1.78, 1.86])
#Vewendete Daten
#R_strich = 500
#C = Wert3
#R = 332

np.savetxt('frequenzmessung.txt', np.column_stack([v, U]), header="v [Hz], U [V]")


#a) Wheatstonebrücke
#Widerstand Wert 14
R_2    = np.array([332, 500, 1000])
R_2er  = R_2*0.002
R_3    = np.array([733, 645, 475])
R_4    = np.array([267, 355, 525])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005

np.savetxt('wheat.wert14.txt', np.column_stack([R_2, R_2er, R_3, R_4, R_34, R_34er]), header="R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")

#Widerstand Wert 11
R_2    = np.array([1000, 500, 332])
R_2er  = R_2*0.002
R_3    = np.array([331, 497, 598])
R_4    = np.array([669, 503, 402])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005


np.savetxt('wheat.wert11.txt', np.column_stack([R_2, R_2er, R_3, R_4, R_34, R_34er]), header="R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")


#b) Kapazitätsbrücke
#R/C-Kombination Wert 8
C_2    = np.array([750, 597, 399])
C_2er  = C_2*0.002
R_2    = np.array([238, 292, 431])
R_2er  = R_2*0.002
R_3    = np.array([716, 671, 579])
R_4    = np.array([284, 329, 421])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005

np.savetxt('kapa.kombiwert8.txt', np.column_stack([C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er]), header="C_2 [nF], C_2error [nF], R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")

#Kondensator Wert 3
C_2 = np.array([399, 750, 597])
C_2er  = C_2*0.002
R_2 = np.array([0, 0, 0])
R_2er  = R_2*0.002
R_3 = np.array([489, 642, 589])
R_4 = np.array([511, 358, 411])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005

np.savetxt('kapa.wert3.txt', np.column_stack([C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er]), header="C_2 [nF], C_2error [nF], R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")


#Kondensator Wert 1
C_2 = np.array([597, 750, 399])
C_2er  = C_2*0.002
R_2 = np.array([3, 0, 1])
R_2er  = R_2*0.002
R_3 = np.array([478, 532, 378])
R_4 = np.array([522, 468, 622])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005

np.savetxt('kapa.wert1.txt', np.column_stack([C_2, C_2er, R_2, R_2er, R_3, R_4, R_34, R_34er]), header="C_2 [nF], C_2error [nF], R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")



#c) Induktivitätsbrücke
#L/R-Kombination Wert 19
L_2    = np.array([20.1, 14.6])
L_2er  = L_2*0.002
R_2    = np.array([91, 68])
R_2er  = R_2*0.002
R_3    = np.array([572, 650])
R_4    = np.array([428, 350])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1]])
R_34er = R_34*0.005

np.savetxt('indu.kombiwert19.txt', np.column_stack([L_2, L_2er, R_2, R_2er, R_3, R_4, R_34, R_34er]), header="L_2 [mH], L_2error [mH], R_2 [ohm], R_2error [ohm], R_3 [ohm], R_4 [ohm], R_3/R_4, R_34error [ohm]")


#d) Maxwell-Brücke
#L/R-Kombination Wert 19
C_4    = np.array([450, 450, 450])
C_4er  = C_4*0.002
R_2    = np.array([1000, 322, 664])
R_2er  = R_2*0.002
R_3    = np.array([69, 190, 102])
R_3er  = R_3*0.002 #NICHT SICHER OB WIRKLICH 0,2%, DA R3 VARIABEL EINSTELLBAR WAR
R_4    = np.array([556, 548, 552])
R_34   = np.array([R_3[0]/R_4[0], R_3[1]/R_4[1], R_3[2]/R_4[2]])
R_34er = R_34*0.005

np.savetxt('max.kombiwert19.txt', np.column_stack([C_4, C_4er, R_2, R_2er, R_3, R_3er, R_4, R_34, R_34er]), header="C_4 [nF], C_4error [nF], R_2 [ohm], R_2error [ohm], R_3 [ohm], R_3error [ohm], R_4 [ohm], R_3/R_4, R_34error")

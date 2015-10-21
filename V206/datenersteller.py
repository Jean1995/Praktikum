import numpy as np
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
T1 = np.array([20.8, 21.6, 22.9, 24.3, 25.6, 27.0, 28.3, 29.6, 30.9, 32.1, 33.2, 34.3, 35.4, 36.4, 37.5, 38.4, 39.3, 40.3, 41.2, 42.0, 42.8, 43.7, 44.5, 45.2, 46.0, 46.8, 47.5, 48.2, 48.9, 49.6, 50.2])
Pb = np.array([5.50, 6.00, 6.30, 6.50, 6.90, 7.00, 7.30, 7.50, 7.80, 8.00, 8.20, 8.50, 8.80, 9.00, 9.20, 9.50, 9.60, 9.90, 10.00, 10.20, 10.50, 10.60, 10.90, 11.00, 11.20, 11.50, 11.50, 11.80, 12.00, 12.00, 12.20])
T2 = np.array([21.3, 21.0, 19.8, 18.7, 17.5, 16.6, 15.7, 14.9, 14.1, 13.3, 12.4, 11.7, 10.9, 10.3, 9.5, 8.9, 8.3, 7.7, 7.2, 6.7, 6.2, 5.7, 5.3, 5.0, 4.6, 4.3, 4.0, 3.8, 3.5, 3.3, 3.2])
Pa = np.array([5.30, 4.80, 4.80, 4.80, 4.80, 4.70, 4.60, 4.40, 4.30, 4.20, 4.10, 4.00, 4.00, 3.90, 3.80, 3.80, 3.75, 3.70, 3.70, 3.60, 3.60, 3.60, 3.60, 3.55, 3.50, 3.50, 3.50, 3.50, 3.45, 3.45, 3.40])
P = np.array([0, 125, 125, 125, 125, 125, 125, 125, 123, 122, 122, 122, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126])
np.savetxt('daten.txt', np.column_stack([t, T1, Pb, T2, Pa, P]), header="t T1 Pb T2 Pa P")

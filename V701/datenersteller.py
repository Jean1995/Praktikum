# Überführt Messdaten in auslesbare Textdaten (optional)
import numpy as np

###### Messung 1 #####
s_1 = 0.025
delta_t_1 = 90 # oops
p_1= np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
pulse_1 = np.array([46750, 46364, 45563, 45899, 45929, 45145, 44961, 44552, 43570, 44134, 43101, 43046, 41831, 42294, 41779, 41044, 40559, 39685, 38353, 35579, 33246, 29562, 22986, 17301, 11904, 5620, 2988, 1391, 495, 225, 0])
channel_1 = np.array([1023, 1023, 943, 911, 864, 832, 819, 775, 775, 743, 719, 711, 687, 660, 652, 639, 591, 574, 527, 511, 511, 511, 406, 411, 408, 404, 410, 398, 408, 401, 0])

np.savetxt('messung_1.txt', np.column_stack([p_1, pulse_1, channel_1]), header="p_1 [mbar], pulse_1, channel_1")

##### Messung 2 ####
s_2 = 0.015
delta_t_2 = 90 # oops i did it again
p_2= np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
pulse_2 = np.array([89057, 88886, 88362, 88278, 88424, 87728, 87117, 86813, 86582, 86683, 85484, 85372, 84597, 83819, 83948, 83268, 81514, 80627, 80220, 79126, 75598])
channel_2 = np.array([1023, 1023, 1023, 948, 972, 943, 896, 911, 859, 847, 839, 815, 792, 768, 736, 704, 687, 664, 652, 591, 511])

np.savetxt('messung_2.txt', np.column_stack([p_2, pulse_2, channel_2]), header="p_2 [mbar], pulse_2, channel_2")

#### Statistikmessung ####
nr = np.arange(1, 101)
pulse = np.array([11195, 10867, 10740, 10958, 10505, 10656, 10422, 10620, 10919, 10832, 11232, 10143, 10715, 10864, 10280, 10547, 10925, 10594, 10381, 10884, 10986, 10769, 10788, 10138, 10538, 11486, 10568, 11983, 10514, 10823, 10753, 10072, 10400, 10155, 10355, 10081, 10784, 10390, 10526, 10445, 10037, 11108, 10835, 10074, 10340, 10445, 10376, 10449, 11026, 10872, 11141,10568,10207,10949,10944,10757,10095,11409,10344,11263,10241,11608,11000,10106,11076,10067,10288,10246,10043,10747,10971,10730,10807,10598,10461,10616,11021,10098,10892,11063,10548,10085,10190,10779,10556,10549,10995,9829,10452,10166,10047,11497,10809,10813,10056,11050,11028,11091,9885,10395
])

np.savetxt('messung_stat.txt', np.column_stack([nr, pulse]), header="nr, pulse")

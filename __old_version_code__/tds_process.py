from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import math
import os

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def li_slope (xs, ys):
	"""get the slope of how background is changing."""
	slope_fit = (np.mean(xs) * np.mean(ys) - np.mean(xs*ys))/(np.mean(xs) ** 2 - np.mean(xs ** 2))
	return slope_fit

#import the data.
data_temp = np.genfromtxt(sys.argv[1], skip_header=3, skip_footer=3)
counts_all = data_temp[:,0]
temps_all = data_temp[:,1]

#match the starting point. Start from the point closest to 95 K.
startpoint = find_nearest(temps_all, 95)
counts_raw = counts_all[startpoint:startpoint+810]
temps = temps_all[startpoint:startpoint+810]

#Level the background
xs_raw = np.vstack((temps[:50], temps[-50:]))
ys_raw = np.vstack((counts_raw[:50], counts_raw[-50:]))
slope_fit = li_slope(xs_raw, ys_raw)
counts_sub = []
n = 0
while n < len(counts_raw):
	count_sub = counts_raw[n] - temps[n] * slope_fit
	counts_sub.append(count_sub)
	n = n + 1
counts_sub = np.array(counts_sub)

#import the background file
filepath_background = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '2017', '10_oct', 'oct18_17.d03')
data_temp_background = np.genfromtxt(filepath_background, skip_header=3, skip_footer=3)
counts_all_background = data_temp_background[:,0]
temps_all_background = data_temp_background[:,1]

#match the starting point. Start from the point closest to 95 K.
startpoint_background = find_nearest(temps_all_background,95)
counts_raw_background = counts_all_background[startpoint_background:startpoint_background+810]
temps_background = temps_all_background[startpoint_background:startpoint_background+810]

#level the background
xs_background = np.vstack((temps_background[:50], temps_background[-50:]))
ys_background = np.vstack((counts_raw_background[:50], counts_raw_background[-50:]))
slope_fit = li_slope(xs_background, ys_background)
counts_sub_background = []
n = 0
while n < len(counts_raw_background):
	count_sub_background = counts_raw_background[n] - temps_background[n] * slope_fit
	counts_sub_background.append(count_sub_background)
	n = n + 1
counts_sub_background = np.array(counts_sub_background)

while True:
	#background subtraction
	sensitivity = float(input("sensivitity factor: "))
	counts_bgsub = counts_sub/sensitivity - counts_sub_background

	f1 = plt.figure()
	plt.plot(temps_all, counts_all, linewidth = 1, label = "raw TDS")
	plt.plot(temps, counts_sub, linewidth = 1, label = "leveled TDS")
	plt.legend()
	plt.ylabel('Counts')
	plt.xlabel('Temperature (rate = 2 K/sec)')
	f2 = plt.figure()
	plt.plot(temps, counts_bgsub, linewidth = 1, label = "background subtracted TDS")
	plt.ylabel('Counts')
	plt.xlabel('Temperature (rate = 2 K/sec)')
	plt.legend()
	plt.show()
	message = input("Do you want to save the data to p{}? ".format(sys.argv[1]) + "(Y/N): ")
	if message == 'y':
		break
	elif message == 'n':
		continue
	else:
		quit()
#creat a txt file for igor process
file = open('p'+sys.argv[1],'w+')
file.write('IGOR\nWAVES Counts, Temperature, Time_elapsed\nBegin\n')
data_processed = zip(counts_bgsub, temps, np.arange(0, len(temps)-1, 0.2))
for counts, temperatures, times in data_processed:
	file.write('{} {} {}\n'.format(counts, temperatures, times))
file.write('End\n')
file.close()
print('Success!')

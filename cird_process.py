#Process the CIRD data.

from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from CeyerLibrary import generateDict
from scipy.optimize import curve_fit
import sys

def li_slope (xs, ys):
	"""get the slope of how background is changing."""
	slope_fit = (np.mean(xs) * np.mean(ys) - np.mean(xs*ys))/(np.mean(xs) ** 2 - np.mean(xs ** 2))
	return slope_fit

data_temp = np.genfromtxt(sys.argv[1], skip_header=3, skip_footer=3)

#Get a new column for time elapsed during CIRD, increment = 0.2s)
counts_raw = data_temp[:,0]
number_datapoints = len(counts_raw)
time_end = (number_datapoints - 1) * 0.2
times = np.arange(0., time_end + 0.2, 0.2)

#Level the background with a linear subtraction in order to get beam start time
xs = times[-250:]
ys = counts_raw[-250:]
slope_fit = li_slope(xs, ys)

counts_sub = []
time = 0
for time in times.tolist():
        count_sub = counts_raw[int(5 * time)] - time * slope_fit
        counts_sub.append(count_sub)
counts_sub = np.array(counts_sub)
bg_level = np.mean(counts_sub[-300:])

#get the estimated beam start time if the feature is obvious.
if np.max(counts_sub[:]) - bg_level < 200:
        print("No obvious feature is observed!")
else:
        highest_index = np.argmax(counts_sub[:])
        averages_highest = []
        range_candidate = range((highest_index-20),(highest_index + 21))
        for n in range_candidate:
                average_highest = np.mean(counts_sub[(n-5):(n+6)])
                averages_highest.append(average_highest)
        averages_highest = np.array(averages_highest)
        index_average_beamstart = np.argmax(averages_highest)
        index_beamstart = highest_index + (index_average_beamstart - 20 - 5)
        time_beamstart = times[index_beamstart]

        #now beam time has been calculated, so use 50 datapoints before the
        #beam time and 50 datapoints before the end points to get the background.
        xs = np.vstack((times[(index_beamstart-50):index_beamstart], times[-50:]))
        ys = np.vstack((counts_raw[(index_beamstart-50):index_beamstart], counts_raw[-50:]))
        slope_fit = li_slope(xs, ys)
        counts_sub = []
        time = 0
        for time in times.tolist():
                count_sub = counts_raw[int(5 * time)] - time * slope_fit
                counts_sub.append(count_sub)
        counts_sub = np.array(counts_sub)
        bg_level = np.mean(counts_sub[-50:])

        #fit the curve
        def func(x, a, b, c):
                return a * np.exp(-b * x) + c
        plt.plot(times, counts_sub, linewidth = 1, label = "background subtracted")
        plt.legend()
        plt.ylabel('Counts')
        plt.xlabel('Time (sec)')
        plt.show()
        time_beamstop = float(input("please input the estimated beam stop time: "))
        index_beamstop = int(time_beamstop * 5)
        xdata = times[index_beamstart:index_beamstop]
        ydata = counts_sub[index_beamstart:index_beamstop]
        popt, pcov =  curve_fit(func, xdata, ydata, p0=[100000, 0.07, bg_level])

        #draw and calculate the area under the curve.
        x1 = np.arange(time_beamstart, time_beamstop, 0.01)
        y1 = func(x1, popt[0], popt[1], popt[2])
        #def integ(x1, a, b):
                #return a * np.exp(-b * x1)
        area = quad(func, time_beamstart, time_beamstop, args = (popt[0], popt[1], 0))
        plt.fill_between(x1, y1, popt[2])

        a = str(popt[0])
        b = str(popt[1])
        c = str(popt[2])

        print("beam start time is at:", time_beamstart, "sec")
        print("the fit curve is: y =", a,"x e^(-",b,"x X) +", c  )
        print ("area =", area[0])

        while True:
                plt.plot(xdata, func(xdata, *popt), label = "fit")
                plt.plot(times, counts_raw, linewidth = 1, label = "raw")
                plt.plot(times, counts_sub, linewidth = 1, label = "background subtracted")
                plt.legend()
                plt.ylabel('Counts')
                plt.xlabel('Time (sec)')
                plt.show()
                message = input("Do you want to overwrite the file " + sys.argv[1] + "? " + "(Y/N): ")
                if message == 'y':
                        break
                elif message == 'n':
                        quit()

        #creat a txt file for igor process
        file = open('p'+sys.argv[1],'w+')
        file.write('IGOR\nWAVES Counts_CIRD, Time_CIRD,\nBegin\n')
        data_processed = zip(counts_sub, times)
        for count, time in zip(counts_sub, times):
                file.write('{} {}\n'.format(count, time))
        file.write('End\n')
        file.close()
        print('Success!')



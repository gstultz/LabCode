from LittleMachine_TDS_Lib import TDS, tds_sensitivity
import matplotlib.pyplot as plt
import sys
import numpy as np

# important paraments that can be changed.
high_p = 2.60 * 10 ** (-9)
middle_p = 1.16 * 10 ** (-9)
low_p = 5.38 * 10 ** (-10)

# other variables that are less often changed.
filename_background = 'oct31_17.d01'
background_sensitivity = 1.4754 * 10 ** 14
start_temp = 95
total_npts = 870
background_npts = 10  # number of datapoints used to level the TDS spectrum.
time_elapsed = np.linspace(0, 0.2 * total_npts, total_npts)
filename_before_cird = sys.argv[1]
filename_after_cird = sys.argv[2]
filename_sensitivity = sys.argv[3]


# Calculate sensitivity factor.
current_sensitivity = tds_sensitivity(filename_sensitivity, high_p=high_p,
                                      middle_p=middle_p, low_p=low_p)
# current_sensitivity = 1.20 * 10 ** 14
print('sensitivity factor is ', current_sensitivity)

# load and level background data.
background = TDS(filename_background)
background.load_data(start_temp=start_temp, background_npts=10,
                     total_npts=total_npts, show_plot=False)

# load and level TDS data before CIRD experiment,
# and perform background subtraction.
tds_before_cird = TDS(filename_before_cird)
tds_before_cird.load_data(start_temp=start_temp, total_npts=total_npts,
                          background_npts=10, show_plot=False)
tds_before_cird.background_subtraction(background=background,
                                       sensitivity=current_sensitivity / background_sensitivity,
                                       show_plot=False)
tds_before_cird.shift_x_axis()

# load and level TDS data after CIRD experiment,
# and perform background subtraction.
tds_after_cird = TDS(filename_after_cird)
tds_after_cird.load_data(start_temp=start_temp, total_npts=total_npts,
                         background_npts=10, show_plot=False)
tds_after_cird.background_subtraction(background=background,
                                      sensitivity=current_sensitivity / background_sensitivity,
                                      show_plot=False)
tds_after_cird.shift_x_axis()

# get the different of the TDS spectra before and after CIRD.
counts_difference = tds_before_cird.counts - tds_after_cird.counts
temp_average = 1 / 2 * (tds_before_cird.temps + tds_after_cird.temps)

# Make a plot showing (from the top to bottom):
# 1. background subtracted TDS spectra before and after CIRD.
# 2. difference of the TDS spectra before and after CIRD.
ax1 = plt.subplot(211)
ax1.plot(tds_before_cird.temps, tds_before_cird.counts, label='TDS before CIRD'
         + '_' + filename_before_cird)
ax1.plot(tds_after_cird.temps, tds_after_cird.counts, label='TDS after CIRD'
         + '_' + filename_after_cird)
plt.ylabel('Counts')
plt.xlabel('Temperature (K), ramp rate = 2 K/sec')
plt.legend()
plt.title('Background Subtracted TDS Spectra Before and After CIRD')

ax2 = plt.subplot(212)
ax2.plot(temp_average, counts_difference)
plt.ylabel('Counts')
plt.xlabel('Temperature (K), ramp rate = 2 K/sec')
plt.title('Difference of the TDS Spectra Before and After CIRD')
plt.tight_layout()
plt.show()

message = input("Do you want to save the data to p{}? ".format(sys.argv[1]) + "(Y/N): ")
if message == 'y':
    file = open(tds_before_cird.filepath[:-12] + 'p' + tds_before_cird.filename, 'w+')
    file.write(
      "IGOR\nWAVES Counts_before_CIRD_{}, temps_before_CIRD_{}, Counts_after_CIRD_{}, temps_after_CIRD_{},"
      " Difference_{}, temp_average_{}, Time_elapsed_{}, \nBegin\n"
        .format(sys.argv[1][0:8], sys.argv[1][0:8], sys.argv[1][0:8], sys.argv[1][0:8], sys.argv[1][0:8], sys.argv[1][0:8],
                sys.argv[1][0:8]))
    data_processed = zip(tds_before_cird.counts, tds_before_cird.temps, tds_after_cird.counts, tds_after_cird.temps,
                         counts_difference, temp_average, time_elapsed)
    for a, b, c, d, e, f, g in data_processed:
        file.write('{} {} {} {} {} {} {}\n'.format(a, b, c, d, e, f, g))
    file.write('End\n')
    file.close()
    print('Success!')
else:
    quit()

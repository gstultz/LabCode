import matplotlib.pyplot as plt
from LittleMachine_TDS_Lib import step_average, tds_sensitivity, TDS, generate_dataframe, write_file

# begin__________________important paraments________________________________
high_p = 3.08 * 10 ** (-9)
middle_p = 1.87 * 10 ** (-9)
low_p = 9.20 * 10 ** (-10)
percent_npts = 0.5

filename_before_cird = "dec20_17.d02"
filename_after_cird = "dec20_17.d03"
filename_sensitivity = "dec20_17.d04"
filename_background = "oct31_17.d01"  # This should not be changed.

dwell_time = 0.2
start_temp = 87     # The temperature in the TDS plot starts from here.
npts = 20            # number of datapoints used to level the TDS spectrum.
total_npts = 840
background_sensitivity = 1.4754 * 10 ** 14
# end__________________important paraments________________________________

# Initiate the TDS class.
tds_before_cird = TDS(filename_before_cird)
tds_after_cird = TDS(filename_after_cird)
background = TDS(filename_background)
sensitivity = TDS(filename_sensitivity)

# Load all the files, and plot the raw data.
sensitivity.load_data(dwell_time=dwell_time, show_plot=False)
tds_before_cird.load_data(dwell_time=dwell_time, show_plot=False)
tds_after_cird.load_data(dwell_time=dwell_time, show_plot=False)
background.load_data(dwell_time=dwell_time, show_plot=False)

# Level the background, and plot the leveled TDS data.
tds_before_cird.level_background(npts=npts, start_temp=start_temp, total_npts=total_npts, leveled_plot=False)
tds_after_cird.level_background(npts=npts, start_temp=start_temp, total_npts=total_npts, leveled_plot=False)
background.level_background(npts=npts, start_temp=start_temp, total_npts=total_npts, leveled_plot=True)

# calculate sensitivity factor, and the plot shows if the delimiter is correct.
averages_counts = step_average(sensitivity.data, percent_npts)
current_sensitivity = tds_sensitivity(averages_counts, high_p, middle_p,
                                      low_p, show_plot=False)

# Shift the spectra baseline to the x-axis, and then do background subtraction.
# Note that the order of these two steps does not matter.
tds_before_cird.shift_x_axis(npts)
tds_after_cird.shift_x_axis(npts)
background.shift_x_axis(npts)
tds_before_cird.background_subtraction(background, sensitivity=
                                       current_sensitivity/background_sensitivity,
                                       show_plot=True)
tds_after_cird.background_subtraction(background, sensitivity=
                                       current_sensitivity/background_sensitivity,
                                       show_plot=True)

# Generate a dataframe that contains all the raw and processed data, including
# the background. This dataframe can be used for generating plot, etc.
data_processed = generate_dataframe(tds_before_cird, tds_after_cird, background)
write_file(data_processed, tds_after_cird)

import sys
from LittleMachine_TDS_Lib import CIRD

cird = CIRD(sys.argv[1])
cird.load_data(show_plot=True)
cird.level_background()
cird.peak_fit(area_high=False)
cird.create_plot(area_high=False)
cird.write_file()

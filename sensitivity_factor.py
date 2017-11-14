import matplotlib.pyplot as plt
from LittleMachine_TDS_Lib import load_sensitivity

# Change this for the data interested
filename = 'oct19_17.d03'


# Load data
counts, time = load_sensitivity(filename)
plt.plot(time, counts)
plt.show()

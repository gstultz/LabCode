# Auger plot

from CeyerLibrary import loadAuger, generateDict
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

monthDict = generateDict()  #Set up the dictionary

#%% Parameters
filename = 'aug22_16.a01'
#yHeight = 200

#%% Import data
filefolder = '/Users/qingliu/Dropbox (MIT)/littlemachine/'
filepath = filefolder + '20' + filename[6:8] + '/' + monthDict[filename[:3]] + '_' + filename[:3] + '/' + filename
filehandle = open(filepath)
data = loadAuger(filepath)

#%% Normalize data
energy = data[:,1]
counts = data[:,0]
#counts = counts/max(counts)*10000
plt.plot(counts,energy)
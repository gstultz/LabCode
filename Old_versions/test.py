## -*- coding: utf-8 -*-
#"""
#Created on Thu Apr 20 18:50:30 2017

#@author: qingliu
#"""

## EELS fitting

#from CeyerLibrary import loadEELS, generateDict, index_of
##from numpy import sqrt, pi, exp, linspace, where
##from scipy.optimize import curve_fit
##from lmfit import Model
#from lmfit.models import LinearModel, GaussianModel, LorentzianModel,ExponentialModel,VoigtModel, StudentsTModel
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages

#monthDict = generateDict()  #Set up the dictionary

##%% Parameters
#filename = 'jun11_15.e01'
#yHeight = 250

##%% Import data
#filefolder = '/Users/qingliu/Dropbox (MIT)/littlemachine/'
#filepath = filefolder + '20' + filename[6:8] + '/' + monthDict[filename[:3]] + '_' + filename[:3] + '/' + filename
#filehandle = open(filepath)
#data = loadEELS(filepath)

##%% Normalize data
#energy = data[:,1]
#counts = data[:,0]

#x = energy
#y = counts/max(counts) * 10000

#ix1 = index_of(x, 500)
#ix2 = index_of(x, 1000)
#ix3 = index_of(x, 2000)
##%%
#def get_fwhm(x, y):
    
    #'''
    #Return the FWHM and Amplitude of the EELS elastic peak
    #'''
    
    #gmodel = GaussianModel()
    #pars = gmodel.guess(y, x=x)
    #result = gmodel.fit(y, pars, x=x)
    
    #fwhm = result.params['fwhm'].value
    #amplitude = result.params['amplitude'].value
    
    #return fwhm, amplitude 


##%%
#fig = plt.figure(figsize=(12,6))
#ax1 = fig.add_subplot(1, 1, 1)

#ax1.plot(x, y, 'bo', markersize = 4)
#ax1.plot(x, result.best_fit, 'r-')
#ax1.set_xlabel('wavenumber (cm-1)')
## Make the y-axis label and tick labels match the line color.
#ax1.set_ylabel('counts', color='b')
#for tl in ax1.get_yticklabels():
    #tl.set_color('b')

#ax2 = ax1.twinx()
#ax2.plot(x[ix1:ix3], y[ix1:ix3], 'bo',markersize = 4)
#ax2.plot(x[ix1:ix3], result.best_fit[ix1:ix3], 'r-')

#ax2.set_ylabel('counts', color='r')
#for tl in ax2.get_yticklabels():
    #tl.set_color('r')
    
#plt.ylim(0, yHeight)
#plt.xlim(min(energy), max(energy))
#plt.show()

##pp = PdfPages('multipage.pdf')
##pp.savefig(fig)
##
##pp.close()



import numpy as np
array1 = np.array([1.1, 2.2, 3.3])
array2 = np.array([1, 2, 3])

print ('the difference =', np.subtract(array1, array2))

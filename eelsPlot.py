# EELS plot

from CeyerLibrary import loadEELS, generateDict, index_of
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit.models import GaussianModel, LorentzianModel,VoigtModel
from matplotlib.backends.backend_pdf import PdfPages
monthDict = generateDict()  #Set up the dictionary

import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
#%% Parameters
filename = [
            
            
            'nov18_16.e05',
            'nov30_16.e01',
            'dec01_16.e01',
            'dec07_16.e01',
            'dec07_16.e02',
            'dec07_16.e03',
            
            
            ]
cutoff = [180, 260, 280, 200, 200, 180]
#cutoff = [180, 180, 180, 180, 180, 180]
#cutoff = [180, 180, 180, 180, 180, 180]


filefolder = '/Users/qingliu/Dropbox (MIT)/littlemachine/'
fig = plt.figure(figsize=(8,1.75*len(filename)))        
gs = gridspec.GridSpec(len(filename), 3)
#%% Import data
for i in range(len(filename)):
    filepath = filefolder + '20' + filename[i][6:8] + '/' + monthDict[filename[i][:3]] + '_' + filename[i][:3] + '/' + filename[i]
#    filehandle = open(filepath)
    data = loadEELS(filepath)

#%% Normalize data
    energy = data[:,1]
    counts = data[:,0]
    norCounts = counts/max(counts)*10000
    ix1 = index_of(energy, - min(energy))
    ix2 = index_of(energy, cutoff[i])
    yHeight = max(norCounts[ix2:])
    
    x = energy[:ix1]
    y = counts[:ix1]
    model = GaussianModel()
    pars = model.guess(y, x=x)
    result = model.fit(y, pars,x=x)
    fwhm = result.params['fwhm'].value

    # Plot
    ax0 = plt.subplot(gs[i, 0])
    ax0.plot(x, y, 'bo', markersize = 4)
    ax0.plot(x, result.best_fit, 'r-')
    ax0.set_xlabel('wavenumber (cm-1)',fontsize= 8)
    ax0.set_ylabel('counts', color='k',fontsize= 8)
    plt.title('FWHM ='+"{0:.0f}".format(fwhm),fontsize= 10)
    plt.xlim(-200, 200)
    ax0.locator_params(axis='x',nbins=5)     
    ax0.locator_params(axis='y',nbins=8)                                             
                                        
    ax1 = plt.subplot(gs[i, 1:3])
    ax1.plot(energy,norCounts, 'k-')
    ax1.set_xlabel('wavenumber (cm-1)',fontsize= 8)
    # Make the y-axis label and tick labels match the line color.
    for tl in ax1.get_yticklabels():
        tl.set_color('k')
    
    ax2 = ax1.twinx()
    ax2.plot(energy,norCounts, 'r-')
    ax2.set_ylabel('counts', color='r',fontsize= 8)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
        
    plt.ylim(0, yHeight)
    plt.xlim(min(energy), max(energy))
    plt.title(filename[i], fontsize= 10)
    ax2.locator_params(axis='x',nbins=15)     
    ax2.locator_params(axis='y',nbins=8)   
#    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=1.5)    
fig.tight_layout()    
plt.show()


pp = PdfPages('multipage.pdf')
pp.savefig(fig)

pp.close()

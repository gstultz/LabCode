# This program performs the Auger fitting

from CeyerLibrary import loadAuger, generateDict, index_of, derivativeGaussian
#from numpy import sqrt, pi, exp, linspace, where
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, LorentzianModel,ExponentialModel,VoigtModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
monthDict = generateDict()  #Set up the dictionary

#%% Important Parameters

filename = 'aug22_16.a02'

#%% Import and clean data
filefolder = '/Users/qingliu/Dropbox (MIT)/littlemachine/'
filepath = filefolder + '20' + filename[6:8] + '/' + monthDict[filename[:3]] + '_' + filename[:3] + '/' + filename
filehandle = open(filepath)
data = loadAuger(filepath)

counts = data[:,1]
energy = data[:,0]

ix1 = index_of(energy, 52) + 1 # index_of return index of array *at or below* value, +1 here
ix2 = index_of(energy, 72) + 1

x = energy[ix1:ix2]
y = counts[ix1:ix2]

#%% Constuct fitting model
background  = LinearModel()

pars = background.make_params()
pars['slope'].set(0, min = -1e-12, max = 1e-16)
pars['intercept'].set(0, min = -3, max = 5)

featureNi = Model(derivativeGaussian, prefix = 'Ni_')

pars.update(featureNi.make_params())
pars['Ni_cen'].set(55, min=53, max=65)
pars['Ni_wid'].set(3, min=2, max = 5)
pars['Ni_amp'].set(5, min=0, max = 10)

featureAu = Model(derivativeGaussian, prefix = 'Au_')

pars.update(featureAu.make_params())
pars['Au_cen'].set(67, min=60, max=75)
pars['Au_wid'].set(3, min=2, max = 5)
pars['Au_amp'].set(5, min=0, max = 10)

model = background + featureNi + featureAu

#%% Curve fitting 
result  = model.fit(y, pars, x=x)

#%% Print fitting results
print(result.fit_report(min_correl=0.25))

#%% Plot fitting results
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(x, y, 'bo', markersize = 4)
ax1.plot(x, result.best_fit, 'r-')
ax1.set_xlabel('wavenumber (cm-1)')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('counts', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

plt.xlim(min(x), max(x))
plt.show()

#%% Export results to PDF 
#pp = PdfPages('multipage.pdf')
#pp.savefig(fig)
#
#pp.close()
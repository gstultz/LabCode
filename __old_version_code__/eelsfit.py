# EELS fitting

from CeyerLibrary import loadEELS, generateDict, index_of
#from numpy import sqrt, pi, exp, linspace, where
#from scipy.optimize import curve_fit
#from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, LorentzianModel,ExponentialModel,VoigtModel, StudentsTModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

monthDict = generateDict()  #Set up the dictionary

#%% Parameters
filename = 'jun11_15.e03'
yHeight = 250

#%% Import data
filefolder = '/Users/qingliu/Dropbox (MIT)/littlemachine/'
filepath = filefolder + '20' + filename[6:8] + '/' + monthDict[filename[:3]] + '_' + filename[:3] + '/' + filename
filehandle = open(filepath)
data = loadEELS(filepath)

#%% Normalize data
energy = data[:,1]
counts = data[:,0]

x = energy
y = counts/max(counts) * 10000

ix1 = index_of(x, 500)
ix2 = index_of(x, 1000)
ix3 = index_of(x, 2000)
#%%
background  = LinearModel()

pars = background.make_params()
pars['slope'].set(0, min = -1e-12, max = 1e-16)
pars['intercept'].set(0, min = -10, max = 30)

#elasticPeak = GaussianModel(prefix = 'elas_')
elasticPeak = VoigtModel(prefix = 'elas_')
#elasticPeak = LorentzianModel(prefix = 'elas_')


pars.update(elasticPeak.make_params())
pars['elas_center'].set(0, min=-20, max=20)
pars['elas_sigma'].set(10, min=3, max = 90)
pars['elas_amplitude'].set(40, min=20)

lossPeak1= LorentzianModel(prefix = 'loss1_')
#lossPeak1= VoigtModel(prefix = 'loss1_')

pars.update(lossPeak1.make_params())
pars['loss1_center'].set(750, min=500, max=800)
pars['loss1_sigma'].set(70, min=50, max = 100)
pars['loss1_amplitude'].set(20, min=10)

lossPeak2= LorentzianModel(prefix = 'loss2_')
#lossPeak2= VoigtModel(prefix = 'loss2_')

pars.update(lossPeak2.make_params())
pars['loss2_center'].set(1500, min=1200, max=1700)
pars['loss2_sigma'].set(70, min=50, max = 200)
pars['loss2_amplitude'].set(20, min=10)

model = background + elasticPeak + lossPeak1 + lossPeak2
result  = model.fit(y, pars, x=x)

#%%
print(result.fit_report(min_correl=0.25))


#%%
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(x, y, 'bo', markersize = 4)
ax1.plot(x, result.best_fit, 'r-')
ax1.set_xlabel('wavenumber (cm-1)')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('counts', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(x[ix1:ix3], y[ix1:ix3], 'bo',markersize = 4)
ax2.plot(x[ix1:ix3], result.best_fit[ix1:ix3], 'r-')

ax2.set_ylabel('counts', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    
plt.ylim(0, yHeight)
plt.xlim(min(energy), max(energy))
plt.show()

#pp = PdfPages('multipage.pdf')
#pp.savefig(fig)
#
#pp.close()
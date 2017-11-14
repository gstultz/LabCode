
from numpy import where, sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import Model
from lmfit.models import LinearModel
#GaussianModel, LorentzianModel, ExponentialModel, VoigtModel

#%% functions to load EELS files

def generateDict():
    monthKeys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthValues = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    return dict(zip(monthKeys, monthValues))

def index_of(arrval, value):
    "return index of array *at or below* value "
    if value < min(arrval):  return 0
    return max(where(arrval<=value)[0])

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))   
    
def derivativeGaussian(x, amp, cen, wid):
    "1-d derivative gaussian: gaussian(x, amp, cen, wid)"
#    return (-amp/(sqrt(2*pi)*wid**3)) * exp(-(x-cen)**2 /(2*wid**2))
    return (-amp*(x-cen)) * exp(-(x-cen)**2 /(2*wid**2))
  

class Auger:
    """
    The class takes a filepath and a filename, and generate an Auger object.  
    """
    def __init__(self, filename):
        self.filename = filename
        self.filepath = None
        self.x = None
        self.y = None

    def load_data(self): 
        monthDict = generateDict()
        self.filepath = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' + self.filename[6:8], 
                                     monthDict[self.filename[:3]] + '_' + self.filename[:3], self.filename)
        if open(self.filepath, 'r').readline().split()[0] == 'IGOR': 
            print('Auger file {} has been preprocessed, loading the data...'.format(self.filename))
        else: 
            print('Auger file {} has not been preprocessed, now preprocessing...'.format(self.filename))
            self.preprocess_Auger(self.filepath)
            print('Preprocessing finished, now loading the data...'.format(self.filename))
        data = np.loadtxt(self.filepath, comments = ['E','W','B'], skiprows = 3)
        self.x = data[:, 0]
        self.y = data[:, 1]
        print('Data loaded successfully!')
    
    @staticmethod    
    def preprocess_Auger(filepath): 
        f = open(filepath, 'r')
        data = f.readlines()
        f.close()
        scan = 65
        i = 0
        file = open(filepath,'w')
        file.write('IGOR\nWaves Voltage' + chr(scan) + ', Auger' + chr(scan) +'\nBegin\n')
        curr_energy = list(map(float, data[i].split()))[0]
        while i < len(data) - 1:
            next_energy = list(map(float, data[i + 1].split()))[0]
            if abs(curr_energy - next_energy) > 0.1:
                file.write(data[i])
                if abs(curr_energy - next_energy) > 0.5:
                    scan += 1
                    file.write('End\n\nWaves Voltage' + chr(scan) + ', Auger' + chr(scan) +'\nBegin\n')
            curr_energy = next_energy
            i += 1
        file.write(data[i] + 'End\n')
        file.close()
        
    def plot_data(self, xmin=None, xmax=None): 
        if self.x is None: 
            print('Data has not been loaded yet, calling load_data()...')
            self.load_data()
        plt.figure(figsize=(15, 6))
        plt.plot(self.x, self.y)
        plt.xlim([xmin, xmax])
        plt.grid()
        plt.show()
        
    def fit_Au_coverage(self): 
        energy = self.x
        counts = self.y
        ix1 = index_of(energy, 53) + 1 # index_of return index of array *at or below* value, +1 here
        ix2 = index_of(energy, 72) + 1
        x = energy[ix1:ix2]
        y = counts[ix1:ix2]
        
        # Constuct fitting model
        background  = LinearModel()
        
        pars = background.make_params()
        pars['slope'].set(0, min = -1, max = 1)
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
        # Curve fitting 
        result  = model.fit(y, pars, x=x)
        # print(result.fit_report(min_correl=0.25))
        I_Ni = result.params.get('Ni_amp').value
        I_Au = result.params.get('Au_amp').value
        
        alpha = 1.5287 / 2.3433
        beta = 0.525287
        yita = I_Ni / I_Au
        
        Au_coverage = alpha * (1 + beta) / (alpha + yita)
        print('Au coverage is {:.3f} ML'.format(Au_coverage))
                                
        # Plot fitting results
        plt.figure(figsize=(12,6))
        plt.plot(x, y, 'bo', markersize = 4)
        plt.plot(x, result.best_fit, 'r-')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Voltge (V)')
        plt.xlim(min(x), max(x))
        plt.show()
 


        
        
        

# CeyerLibrary includes useful functions for data analysis
from numpy import genfromtxt, loadtxt, where, sqrt, pi, exp

#%% functions to load EELS files

def generateDict():
    monthKeys = ['jan', 'feb', 'mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    monthValues = ['01','02','03','04','05','06','07','08','09','10','11','12']
    return dict(zip(monthKeys, monthValues))

def index_of(arrval, value):
    "return index of array *at or below* value "
    if value < min(arrval):  return 0
    return max(where(arrval<=value)[0])
    
def loadEELS(filepath):
    data = loadtxt(filepath, comments = ['E','X'], delimiter = ',',skiprows = 3)
    return data


def loadEELS_v1(filepath):
    data = genfromtxt(filepath, delimiter = ',',skip_header = 3, skip_footer = 15)
    return data


def loadEELS_v2(filepath):
    counts = []
    wavenumber = []
    fhandle = open(filepath)
    indicator = False
    for line in fhandle:
        if line.startswith('BEGIN'): 
            indicator = True
            continue
        if line.startswith('END'):
            indicator = False
            continue
        if indicator is True:
            words = line.strip().split(',')
            #Sprint words
            counts.append(words[0])
            wavenumber.append(words[1])
        
    return counts, wavenumber


def loadAuger(filepath):
    data = loadtxt(filepath, comments = ['E','W','B'], skiprows = 3)
    return data


def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))   
    
def derivativeGaussian(x, amp, cen, wid):
    "1-d derivative gaussian: gaussian(x, amp, cen, wid)"
#    return (-amp/(sqrt(2*pi)*wid**3)) * exp(-(x-cen)**2 /(2*wid**2))
    return (-amp*(x-cen)) * exp(-(x-cen)**2 /(2*wid**2))
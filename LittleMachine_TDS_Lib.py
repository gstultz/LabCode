from numpy import where, sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import Model
from lmfit.models import LinearModel

from scipy.integrate import quad
from scipy.optimize import curve_fit
import sys
import math

#GaussianModel, LorentzianModel, ExponentialModel, VoigtModel

#%% useful functions
def generateDict():
    monthKeys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthValues = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    return dict(zip(monthKeys, monthValues))

def index_of(arrval, value):
    "return index of array *at or below* value "
    if value < min(arrval):  return 0
    return max(where(arrval<=value)[0])

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx
             
def li_slope (xs, ys):
	"""get the slope of how background is changing."""
	slope_fit = (np.mean(xs) * np.mean(ys) - np.mean(xs * ys)) / (np.mean(xs) ** 2 - np.mean(xs ** 2))
	return slope_fit

def linear_subtraction(x, y, npts):
    x_background = np.concatenate((x[:npts], x[-npts:]))
    y_background = np.concatenate((y[:npts], y[-npts:]))
    res = LinearModel().fit(y_background, x=x_background)
    slope = res.params.get('slope').value
    intercept = res.params.get('intercept').value
    y_subtracted = y - (intercept + slope * x)
    return y_subtracted   

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))   
    
def derivativeGaussian(x, amp, cen, wid):
    "1-d derivative gaussian: gaussian(x, amp, cen, wid)"
#    return (-amp/(sqrt(2*pi)*wid**3)) * exp(-(x-cen)**2 /(2*wid**2))
    return (-amp*(x-cen)) * exp(-(x-cen)**2 /(2*wid**2))

def load_sensitivity(filename):
    monthDict = generateDict()
    filepath = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' + filename[6:8],
                                 monthDict[filename[:3]] + '_' + filename[:3], filename)
    data = np.genfromtxt(filepath, skip_header=3, skip_footer=3)
    counts = data[:,0]
    return counts, np.arange(len(counts))

def tds_sensitivity(filename_sensitivity: object, high_p: object = 20 * 10 ** (-10), middle_p: object = 10 * 10 ** (-10),
                    low_p: object = 5 * 10 ** (-10)) -> object:
    """
    pressure exponent ^-10 has been included.
    """
    counts, time = load_sensitivity(filename_sensitivity)
    high_c_start = int(input('start of high pressure: '))
    high_c_stop = int(input('stop of high pressure: '))
    middle_c_start = int(input('start of middle pressure: '))
    middle_c_stop = int(input('stop of middle pressure: '))
    low_c_start = int(input('start of low pressure: '))
    low_c_stop = int(input('stop of low pressure: '))

    high_c = np.mean(counts[high_c_start:high_c_stop])
    middle_c = np.mean(counts[middle_c_start:middle_c_stop])
    low_c = np.mean(counts[low_c_start:low_c_stop])
    xs = np.array([0, low_p, middle_p, high_p])
    ys = np.array([0, low_c, middle_c, high_c])
    sensitivity = li_slope(xs, ys)
    return sensitivity
#%% 
class TDS:
    """
    The class takes a filepath and a filename, and generate a TDS object.  
    """
    def __init__(self, filename):
        self.filename = filename
        self.filepath = None
        self.counts = None
        self.temps = None

    def load_data(self, start_temp: object = 95, total_npts: object = 870, background_npts: object = 10, show_plot: object = True) -> object:
        """

        :rtype: object
        """
        monthDict = generateDict()
        self.filepath = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' + self.filename[6:8], 
                                     monthDict[self.filename[:3]] + '_' + self.filename[:3], self.filename)
        
        data = np.genfromtxt(self.filepath, skip_header=3, skip_footer = 3)
        counts_all = data[:, 0]
        temps_all = data[:, 1]
#        print('Data loaded successfully!')
        #match the starting point. Start from the point closest to 95 K.
        startpoint = find_nearest(temps_all, start_temp)
        if startpoint + total_npts > len(counts_all):
            raise ValueError('Please modify the start_temp or total_npts!')
        counts_raw = counts_all[startpoint : startpoint + total_npts]
        self.temps = temps_all[startpoint : startpoint + total_npts]
        self.counts = linear_subtraction(self.temps, counts_raw, npts=background_npts)
        
        if show_plot:
            plt.figure(figsize= (8,8))
            plt.plot(temps_all, counts_all, linewidth = 1, label = "Raw")
            plt.plot(self.temps, self.counts, linewidth = 1, label = "Leveled")
            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Temperature (K)')
            plt.grid()
            plt.title(self.filename + ', ramp rate = 2 K/sec')
        
    def background_subtraction(self, background, sensitivity=1.0, show_plot=True): 
        assert isinstance(background, TDS), 'Please provide a valid TDS background.'
        if len(self.temps) == len(background.temps):
            original_counts = self.counts
            self.counts = self.counts / sensitivity - background.counts
        else:
            print('Will deal with it later!')
            return None                        
        if show_plot:
            plt.figure(figsize= (8,8))
            plt.plot(self.temps, original_counts / sensitivity, linewidth = 1, label = "Sensitivity adjusted TDS")
            plt.plot(background.temps, background.counts, linewidth = 1, label = "Background TDS")
            plt.plot(self.temps, self.counts, linewidth = 1, label = "Final TDS")
            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Temperature (K)')
            plt.grid()
            plt.title(self.filename + ', ramp rate = 2 K/sec')
                    
        
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

#%%

class CIRD:
    """
    The class takes a filepath and a filename, and generate a CIRD object.  
    """
    def __init__(self, filename):
        self.filename = filename
        self.filepath = None
        self.counts = None
        self.times = None

    def load_data(self, dwell_time=0.2, background_npts=10, show_plot=True): 
        monthDict = generateDict()
        self.filepath = os.path.join(os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' + self.filename[6:8], 
                                     monthDict[self.filename[:3]] + '_' + self.filename[:3], self.filename)
        data = np.genfromtxt(self.filepath, skip_header=3, skip_footer = 3)
        counts_raw = data[:, 0]
        self.times = np.arange(len(counts_raw)) * dwell_time
#       self.counts = counts_raw                    
        self.counts = linear_subtraction(self.times, counts_raw, npts=background_npts)
        if show_plot:
            plt.figure(figsize= (8,8))
            plt.plot(self.times, counts_raw, linewidth = 1, label = "Raw")
            plt.plot(self.times, self.counts, linewidth = 1, label = "Leveled")
            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Time (s)')
            plt.grid()
            plt.title(self.filename)

    def background_subtraction(self, background, sensitivity=1.0, show_plot=True): 
        assert isinstance(background, TDS), 'Please provide a valid TDS background.'
        if len(self.temps) == len(background.temps):
            original_counts = self.counts
            self.counts = self.counts / sensitivity - background.counts
        else:
            print('Will deal with it later!')
            return None                        
        if show_plot:
            plt.figure(figsize= (8,8))
            plt.plot(self.temps, original_counts / sensitivity, linewidth = 1, label = "Sensitivity adjusted TDS")
            plt.plot(background.temps, background.counts, linewidth = 1, label = "Background TDS")
            plt.plot(self.temps, self.counts, linewidth = 1, label = "Final TDS")
            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Temperature (K)')
            plt.grid()
            plt.title(self.filename + ', ramp rate = 2 K/sec')
                    

                
"""Detect peaks in data based on their amplitude and other features."""

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

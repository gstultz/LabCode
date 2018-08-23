import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import loadtxt, maximum
import matplotlib.gridspec as gridspec
from lmfit.models import (
    LinearModel, GaussianModel, LorentzianModel, VoigtModel
)

from .util import generate_month_dict, index_of


def locate_filepath(filename):
    month_dict = generate_month_dict()
    month_folder = month_dict[filename[:3]] + '_' + filename[:3]
    filepath = os.path.join(
        os.path.expanduser('~'),
        'Dropbox (MIT)',
        'littlemachine',
        '20' + filename[6:8],
        month_folder,
        filename,
    )
    return filepath


class EELS:
    """
    Class for processing EELS files.

    This class takes an EELS filename and generate an EELS object. The main
    functionalities of this class include data cleaning, data visualization,
    and curve fitting.
    """

    def __init__(self, filename):
        self.filename = filename
        self.filepath = locate_filepath(self.filename)

    def load_data(self, ymax=200, spike_removal=False, overwrite=False):
        """
        Load data to self.data while removing unrelated information.
        """
        # Check if the EELS file has been preprocessed.
        # If not, call the preprocess_EELS function.
        if open(self.filepath, 'r').readline().split()[0] == 'IGOR':
            print('EELS {} already preprocessed, loading data...'.format(
                self.filename)
            )
        else:
            print('EELS {} not preprocessed, now preprocessing...'.format(
                self.filename)
            )
            self.preprocess_EELS(self.filepath, ymax, spike_removal, overwrite)

        self.data = loadtxt(
            self.filepath, comments=['E', 'X'], delimiter=',', skiprows=3
        )
        self.energy = self.data[:, 1]
        self.counts = self.data[:, 0]
        self.nor_counts = self.counts / max(self.counts) * 10000
        print('Data loaded successfully!')

    @staticmethod
    def preprocess_EELS(filepath, ymax=200, spike_removal=False,
                        overwrite=False):
        """
        Convert the EELS file into Igor text format, remove spike if requested.
        """
        meV_to_cm = 8.065
        with open(filepath, 'r') as f:
            data = f.readlines()
        df = pd.read_csv(filepath, comment='X')
        df.iloc[-1].plot(ylim=(0, ymax), grid=True, figsize=(8, 6))
        if spike_removal:
            threshold = df.max() > maximum(40, 70 * df.mean())
            print(sum(threshold), 'spikes found at the following locations:')
            spike_location = df.idxmax()[threshold]
            for energy, row in spike_location.iteritems():
                print(energy)
                df.loc[row, energy] = 0
            df.iloc[-1] = df.iloc[:-1].mean()
            df.iloc[-1].plot(ylim=(0, ymax), grid=True, figsize=(8, 6))
        if overwrite:
            file = open(filepath, 'w')
            file.write('IGOR\nWaves Counts, Voltage\nBegin\n')

            voltage = meV_to_cm * df.columns.values.astype('float')
            counts = df.iloc[-1].values

            for c, v in zip(counts[1:], voltage[1:]):
                file.write('{0:.3f}, {1:.3f}\n'.format(c, v))
            file.write('End\n')

            i = df.iloc[-1].name + 2
            while i < len(data):
                file.write(data[i])
                i += 1
            file.close()

    def plot_data(self, cutoff=200):
        """
        Visualize the Auger data as separate scans.
        """

        # If data not loaded, call load_data function
        if len(self.data) == 0:
            self.load_data()

        ix1 = index_of(self.energy, - min(self.energy))
        ix2 = index_of(self.energy, cutoff)
        yHeight = max(self.nor_counts[ix2:])

        x = self.energy[:ix1]
        y = self.counts[:ix1]
        model = GaussianModel()
        pars = model.guess(y, x=x)
        result = model.fit(y, pars, x=x)
        fwhm = result.params['fwhm'].value

        # Plot
        plt.figure(figsize=(10, 3))
        gs = gridspec.GridSpec(1, 3)
        ax0 = plt.subplot(gs[0, 0])
        ax0.plot(x, y, 'bo', markersize=4)
        ax0.plot(x, result.best_fit, 'r-')
        ax0.set_xlabel('wavenumber (cm-1)', fontsize=8)
        ax0.set_ylabel('counts', color='k', fontsize=8)
        plt.title('FWHM ='+"{0:.0f}".format(fwhm), fontsize=10)
        plt.xlim(-200, 200)
        ax0.locator_params(axis='x', nbins=5)
        ax0.locator_params(axis='y', nbins=8)

        ax1 = plt.subplot(gs[0, 1:3])
        ax1.plot(self.energy, self.nor_counts, 'k-')
        ax1.set_xlabel('wavenumber (cm-1)', fontsize=8)
        # Make the y-axis label and tick labels match the line color.
        for tl in ax1.get_yticklabels():
            tl.set_color('k')

        ax2 = ax1.twinx()
        ax2.plot(self.energy, self.nor_counts, 'r-')
        ax2.set_ylabel('counts', color='r', fontsize=8)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.ylim(0, yHeight)
        plt.xlim(min(self.energy), max(self.energy))
        plt.title(self.filename, fontsize=10)
        ax2.locator_params(axis='x', nbins=15)
        ax2.locator_params(axis='y', nbins=8)
        plt.show()

    def fit_EELS(self, yHeight=100):
        """
        Calculate the EELS loss features.

        Parameters
        ----------
        yHeight: int
            y value for plot

        Returns
        -------
        string
            fitted Au coverage with upper and lower bound for each scan
        """
        x = self.energy
        y = self.nor_counts

        ix1 = index_of(x, 500)
        # ix2 = index_of(x, 1000)
        ix3 = index_of(x, 2000)

        background = LinearModel()

        pars = background.make_params()
        pars['slope'].set(0, min=-1e-12, max=1e-16)
        pars['intercept'].set(0, min=-10, max=30)

        # elasticPeak = GaussianModel(prefix='elas_')
        elasticPeak = VoigtModel(prefix='elas_')
        # elasticPeak = LorentzianModel(prefix='elas_')

        pars.update(elasticPeak.make_params())
        pars['elas_center'].set(0, min=-20, max=20)
        pars['elas_sigma'].set(10, min=3, max=90)
        pars['elas_amplitude'].set(40, min=20)

        lossPeak1 = LorentzianModel(prefix='loss1_')
        # lossPeak1 = VoigtModel(prefix='loss1_')

        pars.update(lossPeak1.make_params())
        pars['loss1_center'].set(750, min=500, max=800)
        pars['loss1_sigma'].set(70, min=50, max=100)
        pars['loss1_amplitude'].set(20, min=10)

        lossPeak2 = LorentzianModel(prefix='loss2_')
        # lossPeak2 = VoigtModel(prefix='loss2_')

        pars.update(lossPeak2.make_params())
        pars['loss2_center'].set(1500, min=1200, max=1700)
        pars['loss2_sigma'].set(70, min=50, max=200)
        pars['loss2_amplitude'].set(20, min=10)

        model = background + elasticPeak + lossPeak1 + lossPeak2
        result = model.fit(y, pars, x=x)

        print(result.fit_report(min_correl=0.25))

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(x, y, 'bo', markersize=4)
        ax1.plot(x, result.best_fit, 'r-')
        ax1.set_xlabel('wavenumber (cm-1)')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('counts', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        ax2.plot(x[ix1:ix3], y[ix1:ix3], 'bo', markersize=4)
        ax2.plot(x[ix1:ix3], result.best_fit[ix1:ix3], 'r-')

        ax2.set_ylabel('counts', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.ylim(0, yHeight)
        plt.xlim(min(self.energy), max(self.energy))
        plt.show()

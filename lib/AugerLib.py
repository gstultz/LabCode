import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import where, sqrt, pi, exp
from lmfit import Model
from lmfit.models import LinearModel
from collections import OrderedDict


def generate_month_dict():
    """
    Return a dictionary that maps month names to digital format.
    """
    keys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    values = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']
    return dict(zip(keys, values))


def index_of(arr, value):
    """
    Return the largest index of an element in an array that is smaller
    than a given value. Array must be sorted.
    """
    if value < min(arr):
        return 0
    return max(where(arr <= value)[0])


def gaussian(x, amp, cen, wid):
    """
    Return a gaussian function given three parameters: amp, cen, wid.
    """
    return (amp / (sqrt(2 * pi) * wid)) * exp(-(x - cen)**2 / (2 * wid**2))


def derivative_gaussian(x, amp, cen, wid):
    """
    Return a derivative gaussian function for the Nickel transition
    """
    # return (-amp/(sqrt(2*pi)*wid**3)) * exp(-(x-cen)**2 /(2*wid**2))
    return (-amp * (x - cen)) * exp(-(x - cen)**2 / (2 * wid**2))


def derivative_gaussian_Au(x, yita, amp, cen, wid):
    """
    Return a derivative gaussian function for the Gold transition
    """
    return (-amp * yita * (x - cen)) * exp(-(x - cen)**2 / (2 * wid**2))


class Auger:
    """
    Class for processing Auger files.

    This class takes an Auger filename and generate an Auger object. The main
    functionalities of this class include data cleaning (separate the data
    into different scans as IGOR text), data visualization, and Au coverage
    determination.

    Example
    -------
    >>> filename = 'oct19_17.a02'
    >>> a = Auger(filename)
    >>> a.plot_data()
    Auger oct19_17.a02 already preprocessed, loading data...
    Data loaded successfully!
    >>> a.fit_Au_coverage()
    "{'A': array([ 0.45 ,  0.434,  0.466])}"

    Parameters
    ----------
    filename : str
        The name of a Auger file.

    Attributes
    ----------
    filename : str
        The name of a Auger file.
    filepath : str
        Path of the Auger file, automatically generated based on filename.
    data: OrderedDict
        Container that holds the Auger data
    num_of_scan: int
        Number of scans in the Auger file
    output: dict
        Fitted Au coverage with upper and lower bound.
    """

    def __init__(self, filename):
        self.filename = filename
        self.filepath = None
        self.data = OrderedDict()
        self.num_of_scan = 0
        self.output = {}

    def load_data(self):
        """
        Load data to self.data while removing unrelated information.
        """
        month_dict = generate_month_dict()
        month_folder = month_dict[self.filename[:3]] + '_' + self.filename[:3]
        self.filepath = os.path.join(os.path.expanduser('~'),
                                     'Dropbox (MIT)',
                                     'littlemachine',
                                     '20' + self.filename[6:8],
                                     month_folder,
                                     self.filename)

        # Check if the Auger file has been preprocessed.
        # If not, call the preprocess_Auger function.
        if open(self.filepath, 'r').readline().split()[0] == 'IGOR':
            print(
                'Auger {} already preprocessed, loading data...'.format(
                    self.filename)
            )
        else:
            print(
                'Auger {} not preprocessed, now preprocessing...'.format(
                    self.filename))
            self.preprocess_Auger(self.filepath)

        f = open(self.filepath, 'r')
        for line in f:
            if line.startswith(('IGOR', 'Waves', '\n')):
                continue
            if line.startswith('Begin'):
                self.num_of_scan += 1
                self.data[chr(64 + self.num_of_scan)] = []
                continue

            if line.startswith(('END', 'End')):
                self.data[chr(64 + self.num_of_scan)] = np.array(
                    self.data[chr(64 + self.num_of_scan)]
                )
                continue
            else:
                self.data[chr(64 + self.num_of_scan)].append(
                    list(map(float, line.strip().split()))
                )
        f.close()
        print('Data loaded successfully!')

    @staticmethod
    def preprocess_Auger(filepath):
        """
        Convert the Auger file into Igor text format,
        and divide the data into separate scans.
        """
        f = open(filepath, 'r')
        data = f.readlines()
        f.close()
        scan = 65  # represents "A" in ASCII character set.
        i = 0
        file = open(filepath, 'w')
        file.write('IGOR\nWaves Voltage' + chr(scan) +
                   ', Auger' + chr(scan) + '\nBegin\n')
        curr_energy = list(map(float, data[i].split()))[0]

        while i < len(data) - 1:
            next_energy = list(map(float, data[i + 1].split()))[0]
            if abs(curr_energy - next_energy) > 0.1:
                file.write(data[i])
                if abs(curr_energy - next_energy) > 1:
                    scan += 1
                    file.write('End\n\nWaves Voltage' + chr(scan) +
                               ', Auger' + chr(scan) + '\nBegin\n')
            curr_energy = next_energy
            i += 1

        file.write(data[i] + 'End\n')
        file.close()

    def plot_data(self):
        """
        Visualize the Auger data as separate scans.
        """

        # If data not loaded, call load_data function
        if len(self.data) == 0:
            self.load_data()

        fig, ax = plt.subplots(1, len(self.data), figsize=(
            min(15, 5 * len(self.data)), 2))
        if len(self.data) == 1:
            ax = [ax]
        for a, key in zip(ax, self.data.keys()):
            a.plot(self.data[key][:, 0], self.data[key][:, 1])
            a.legend(key)
            a.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.suptitle(self.filename)
        plt.show()

    def fit_Au_coverage(
            self,
            alpha=0.4209,
            beta=1.074,
            energy_calibration=False,
            target_energy=55,
            add_note=None,
            show_plot=True,
            output_figure_name=None):
        """
        Calculate the Au coverage for each valid scan.

        Parameters
        ----------
        alpha : float
            Sensitivity factor, defaut value is 0.4209
        beta : float
            Bulk contribution factor, default value is 1.074
        energy_calibration : bool
            Shift the energy (x axis value) if True, keep the original
            energy if False. Default False, set to True with caution!
            Explaination: In rare cases, the Auger energy appears to be
            shifted by a few eV. Because the boundaries and initial values
            of the fitting parameters are optimized for most Auger spectra,
            those Auger with energy shifted cannot be fitted well.
            This parameter allows the user to manully shift the energy to
            target_energy so the Au coverage can be better fitted.
        target_energy: float
            Energy of the highest point of the Nickel transition
        add_note: str
            Note to be printted on the Auger fitting plot. Default None.
        show_plot: bool
            Show the Auger fitting plot if True, otherwise if False.
            Default True.
        output_figure_name: str
            Provide a name if the Auger fitting plot should be saved. If
            None, no plot is saved. Default None.

        Returns
        -------
        string
            fitted Au coverage with upper and lower bound for each scan
        """

        # Go through all scans in the Auger file
        for key, v in self.data.items():
            energy = v[:, 0].copy()
            counts = v[:, 1].copy()

            # Check if the scan contains enough datapoints in selected range
            # If not, discard the scan and proceed to the next
            ix1 = index_of(energy, 53) + 1
            ix2 = index_of(energy, 73) + 1
            if ix2 - ix1 < 10:
                continue

            if energy_calibration:
                counts -= np.mean(counts)
                target_ix = index_of(energy, target_energy)
                peak_ix = np.argmax(counts[0:target_ix])
                energy += (energy[target_ix] - energy[peak_ix])
                # Recalculate ix1 and ix2
                ix1 = index_of(energy, 53) + 1
                ix2 = index_of(energy, 73) + 1

            # Select the data in certain energy range for curving fitting
            x = energy[ix1:ix2]
            y = counts[ix1:ix2]

            # Construct the fitting model in five steps
            # Step 1, linear background
            background = LinearModel()
            pars = background.make_params()
            pars['slope'].set(0, min=-1e-6, max=1e-6)
            pars['intercept'].set(0, min=-5, max=5)

            # Step 2, Ni transition
            featureNi = Model(derivative_gaussian, prefix='Ni_')
            pars.update(featureNi.make_params())
            pars['Ni_cen'].set(58, min=53, max=63)
            pars['Ni_wid'].set(3, min=2, max=5)
            pars['Ni_amp'].set(5, min=0, max=10)

            # Step 3, Au transition
            featureAu = Model(derivative_gaussian_Au, prefix='Au_')
            pars.update(featureAu.make_params())
            pars['Au_yita'].set(0.03, min=0, max=2.5)
            pars['Au_cen'].set(68, min=64, max=73)
            pars['Au_wid'].set(3, min=2, max=5)
            pars['Au_amp'].set(5, min=0, max=10)

            # Step 4, add additional constrains
            pars.add('Au_wid', expr='Ni_wid')
            pars.add('Au_amp', expr='Ni_amp')

            # Step 5, combine three components into the model
            model = background + featureNi + featureAu

            # Curve fitting and Au coverage calculation
            result = model.fit(y, pars, x=x)
            # print(result.fit_report(min_correl=0.25))
            Ni_cen = result.params.get('Ni_cen').value
            Au_cen = result.params.get('Au_cen').value

            rd = 2.8 / 2.49  # Au:Ni atom size ratio
            yita = result.params.get('Au_yita').value
            yita_err = result.params.get('Au_yita').stderr
            Au_coverage = alpha * (1 + beta) / (alpha * rd ** 2 + 1 / yita)
            Au_coverage_lb = alpha * (1 + beta) /\
                (alpha * rd ** 2 + 1 / (yita - yita_err))
            Au_coverage_hb = alpha * (1 + beta) /\
                (alpha * rd ** 2 + 1 / (yita + yita_err))
            components = result.eval_components()
            self.output[key] = np.around(
                [Au_coverage, Au_coverage_lb, Au_coverage_hb], decimals=3)

            # Construct the message to be displayed
            if add_note is None:
                message = ''
            else:
                message = add_note + '\n'
            message += r'$I_{Au}:I_{Ni}$' + ' = {:.3f}\n'.format(yita)
            message += r'$\theta$' + ' = {:.3f} ML'.format(Au_coverage)
            message += '\n' + r'$\theta_l$' + \
                ' = {:.3f} ML'.format(Au_coverage_lb)
            message += '\n' + r'$\theta_h$' + \
                ' = {:.3f} ML'.format(Au_coverage_hb)

            # Plot curve fitting results
            if show_plot:
                plt.figure(figsize=(7, 8))
                plt.plot(x, y, 'bo', markersize=5, label='Raw Data')
                plt.plot(x, result.best_fit, 'r-', label='Best Fit')

                plt.plot(x, components['linear'], 'k--', label='Background')
                plt.plot(x, components['Ni_'], 'g-', label='Ni Transition')
                plt.axvline(x=Ni_cen, color='g', linestyle='--',
                            label='Ni Center = {:.1f}'.format(Ni_cen))
                plt.plot(x, components['Au_'], 'y-', label='Au Transition')
                plt.axvline(x=Au_cen, color='y', linestyle='--',
                            label='Au Center = {:.1f}'.format(Au_cen))

                plt.xlabel('Energy (eV)')
                plt.ylabel('Voltge (V)')
                plt.xlim(min(x), max(x))
                plt.legend(fontsize=12, loc=3)
                plt.text(
                    x.max() - 2.8,
                    0.98 * plt.gca().get_ylim()[1] +
                    0.02 * plt.gca().get_ylim()[0],
                    message,
                    horizontalalignment='center',
                    verticalalignment='top',
                    bbox=dict(facecolor='yellow', alpha=0.2),
                    fontsize=13)
                # plt.title(self.filename + ', Scan ' + key, fontsize=16)
                if output_figure_name is not None:
                    plt.savefig(output_figure_name, bbox_inches='tight')
                plt.show()

        return str(self.output)

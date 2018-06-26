import pandas as pd
import numpy as np
from numpy import where, exp
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from lmfit.models import LinearModel, GaussianModel


def exp_decay(x, a, b, c):
    """Expression for exponential decay"""
    return a * exp(b * x) + c


def biexp_decay(x, a1, a2, b1, b2, c1, c2, d):
    """Expression for bi-exponential decay"""
    return a1 * exp(b1 * (x - c1)) + a2 * exp(b2 * (x - c2)) + d


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


class TDS:
    """
    Class for processing TDS files.

    This class takes a TDS filename and generate a TDS object. The main
    functionalities of this class include data background leveling, background
    subtraction, data visualization, and new data generation.
    """
    def __init__(self, filename):
        self.filename = filename
        self.filepath = None
        self.data = pd.DataFrame()
        self.sensitivity_file = False
        self.dwell_time = None
        self.area_low = None
        self.area_high = None

    def load_data(
            self,
            dwell_time=0.2,
            show_plot=True,
            is_sensitivity_file=False,
            smoothed=False):
        """
        Load data to self.data as a pandas dataframe and plot the raw TDS
        spectrum.

        Parameters
        ----------
        dwell_time : float
            Time spent to collect one datapoint. Defaut value is 0.2
        show_plot : bool
            If True, show the raw TDS spectrum, otherwise if False.
            Default is True.
        is_sensitivity_file: bool
            If True, the loaded file should be sensitivity measurement datafile
            so that sensitivity-related methods can be used.
            Default False.
        smoothed: bool
            If True, use Savizky-Golay filter to smooth TDS spectrum. This is
            rarely used.
            Default False.

        Returns
        -------
        """

        month_dict = generate_month_dict()
        month_folder = month_dict[self.filename[:3]] + '_' + self.filename[:3]
        self.filepath = os.path.join(os.path.expanduser('~'),
                                     'Dropbox (MIT)',
                                     'littlemachine',
                                     '20' + self.filename[6:8],
                                     month_folder,
                                     self.filename)

        self.data = pd.read_csv(self.filepath,
                                delim_whitespace=True,
                                skipfooter=3,
                                skiprows=3,
                                names=["counts_raw", "temp"],
                                engine="python")
        self.dwell_time = dwell_time

        if smoothed:
            self.data['counts_raw'] = savgol_filter(
                self.data['counts_raw'], 5, 1)

        self.data["time"] = dwell_time * np.arange(len(self.data))
        self.sensitivity_file = is_sensitivity_file

        if show_plot:
            if is_sensitivity_file:
                plt.figure(figsize=(10, 4))
                plt.plot(self.data["counts_raw"])
                plt.xlabel('Number of data points')
            else:
                plt.figure(figsize=(8, 4))
                plt.plot(self.data["time"], self.data["counts_raw"])
                plt.xlabel('Time [s]')
            plt.ylabel('Counts [Hz]')
            plt.grid()
            plt.show()

    def calculate_sensitivity_factor(
            self,
            plateau_index,
            plateau_pressure,
            show_plot=True):
        """
        This method is associated with the sensitivity and used to calculate the
        sensitivity factor, and plot it.

        Parameters
        ----------
        plateau_index: list
            Contains all the estimated indices of the start and stop positions
            for each plateau.
        plateau_pressure : list
            Contains the measured pressures corresponding for each plateau.
        show_plot : bool
            If True, show the scatter plot and fitted line to visualize the
            accuracy of sensitivity factor calculation.
            Default is True.

        Returns
        -------
        sensitivity_factor: float
        """
        assert self.sensitivity_file, \
            'Please make sure this is a sensitivity measure!'

        assert len(plateau_index) == 2 * len(plateau_pressure), \
            'Wrong number of indices!'

        if len(plateau_index) == 0 or len(plateau_pressure) == 0:
            print('Please provide plateau indices and pressures during the '
                  'measure!')
            return None

        counts_averages = []
        for i in range(len(plateau_pressure)):
            counts_averages.append(self.data["counts_raw"][
                plateau_index[2*i]:plateau_index[2*i+1]].mean()
            )

        plateau_pressure.append(0)
        counts_averages.append(0)
        X = np.array(plateau_pressure)
        y = np.array(counts_averages)
        sensitivity_factor, intercept = linregress(X, y)[0:2]
        if show_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(X, y, 'o', label='original data')
            plt.plot(X, intercept + sensitivity_factor * X, 'r',
                     label='fitted line')
            plt.show()

        return sensitivity_factor

    def level_background(self, method=1, start_temp=100, total_npts=800,
                         linear_bg_start_npts=10, linear_bg_end_npts=10,
                         sensitivity=1.0, show_plot=True):
        """
        Remove initial spikes by setting a new starting temp, and level TDS
        spectrum by subtracting a linear background, which is fitted based on
        method 1 or method 2 as explained below. Also, TDS spectrum is
        corrected with sensitivity factor.

        Parameters
        ----------
        method: int
            if 1, the datapoints selected to fit the linear background was based
            on those right before the initial spike and those at the end of the
            spectrum.
            if 2, the datapoints are based on those after the intial peak (the
            exact position is defined by start_temp) and those at the end of the
            spectrum.
            Default is 1.
        start_temp : int
            Define the startpoint of the TDS spectrum in order to remove the
            initial spike and define the datapoints used to fit the linear
            background as describe in method 2.
            Default is 100.
        total_npts: int
            Total number of datapoints to collect from the raw TDS spectrum.
            Default is 800.
        linear_bg_start_npts: int
            Number of datapoints at the beginning of TDS used to fit the linear
            background.
            Default is 10.
        linear_bg_end_npts: int
            Number of datapoints at the end of TDS used to fit the linear
            background.
            Default is 10.
        sensitivity: float
            Correct the mass spec sensitivity and normalize TDS. If 1, no
            sensitivity is corrected.
            Default is 1.0
        show_plot : bool
            If True, show the scatter plot and fitted line to visualize the
            accuracy of sensitivity factor calculation.
            Default is True.

        Returns
        -------
        """
        if len(self.data) == 0:
            self.load_data()
        assert method == 1 or method == 2, 'Invalid method number!'

        # Method 1, Level the plot
        if method == 1:
            ramp_start = (
                self.data['counts_raw'] > 1.5 * self.data['counts_raw'][0]
            ).idxmax() - 10

            assert ramp_start > 10, 'Not enough background!'
            bg_start = max(0, ramp_start - linear_bg_start_npts)

            X = pd.concat([
                self.data["time"][bg_start:ramp_start],
                self.data["time"][-linear_bg_end_npts:],
            ])
            y = pd.concat([
                self.data["counts_raw"][bg_start:ramp_start],
                self.data["counts_raw"][-linear_bg_end_npts:],
            ])
            slope, intercept = linregress(X, y)[0:2]

            self.data["counts_leveled"] = (self.data["counts_raw"]
                                           - slope * self.data["time"]
                                           - intercept)
            self.data["counts_leveled"] = (self.data["counts_leveled"]
                                           / sensitivity)

        # Slice the raw spectrum using start_temp and total_npts.
        startpoint = index_of(self.data["temp"], start_temp) + 1
        if startpoint + total_npts > len(self.data):
            raise ValueError('Please modify the start_temp or total_npts!')
        self.data = self.data[
            startpoint:startpoint + total_npts].reset_index(drop=True)
        self.data["time"] = self.dwell_time * np.arange(total_npts)

        # Method 2, Level the plot
        if method == 2:
            X = pd.concat([
                self.data["time"][:linear_bg_start_npts],
                self.data["time"][-linear_bg_end_npts:],
            ])
            y = pd.concat([
                self.data["counts_raw"][:linear_bg_start_npts],
                self.data["counts_raw"][-linear_bg_end_npts:],
            ])
            slope, intercept = linregress(X, y)[0:2]
            self.data["counts_leveled"] = (self.data["counts_raw"]
                                           - slope * self.data["time"]
                                           - intercept)
            self.data["counts_leveled"] = (self.data["counts_leveled"]
                                           / sensitivity)
        if show_plot:
            plt.figure(figsize=(8, 4))
            plt.plot(self.data["temp"], self.data["counts_leveled"])
            plt.xlabel('Temperature [K]')
            plt.ylabel('Counts [Hz]')
            plt.grid()
            plt.show()

    def background_subtraction(self, background, show_plot=True):
        """
        Subtract the background TDS so that peaks in TDS only represents gases
        that desorb from the crystal front surface.

        Parameters
        ----------
        background : object
            A TDS class initiated by a background datafile.
        show_plot : bool
            Default is True.

        Returns
        -------
        """
        assert "counts_leveled" in self.data.columns, \
            'TDS not leveled, please call level_background() first!'
        assert "counts_leveled" in background.data.columns, \
            'Data missing in background file, please provide valid background!'

        self.data["counts_bg_subtracted"] = (self.data["counts_leveled"]
                                             - background.data["counts_leveled"]
                                             )
        if show_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(self.data["temp"], self.data["counts_leveled"],
                     linewidth=1, label="Sensitivity adjusted TDS")
            plt.plot(background.data["temp"], background.data["counts_leveled"],
                     linewidth=1, label="Background TDS")
            plt.plot(self.data["temp"], self.data["counts_bg_subtracted"],
                     linewidth=1, label="Final TDS")
            plt.legend()
            plt.ylabel('Counts [Hz]')
            plt.xlabel('Temperature [K]')
            plt.grid()
            plt.title(self.filename + ', ramp rate = 2 K/sec')
            plt.show()

    def integrate_area(self, low_temp=110, split_temp=210, high_temp=450):
        """
        Calculate the area under two peaks using simple integration method.

        Parameters
        ----------
        low_temp : int
            The start temp (position) of the low temp peak.
            Default is 110.
        split_temp : int
            Two peaks overlapped and split_temp approximately separates them.
            Default is 210.
        high_temp: int
            The end temp (position) of the high temp peak.
            Default is 450.

        Returns
        -------
        """
        low_ix = index_of(self.data["temp"], low_temp) + 1
        split_ix = index_of(self.data["temp"], split_temp) + 1
        high_ix = index_of(self.data["temp"], high_temp) + 1
        self.area_low = (self.data["counts_leveled"][low_ix:split_ix].sum()
                         * self.dwell_time)
        self.area_high = (self.data["counts_leveled"][split_ix:high_ix].sum()
                          * self.dwell_time)

    def fit_area(self, counts_to_use=1, show_report=False, show_plot=True):
        x = self.data["temp"]
        if counts_to_use == 1:
            y = self.data["counts_bg_subtracted"]
        elif counts_to_use == 2:
            y = self.data["counts_leveled"]
        else:
            raise ValueError('Not sure what data to use for y')

        background = LinearModel()
        pars = background.make_params()
        pars['slope'].set(0, min=-2, max=0)
        pars['intercept'].set(0, min=-500, max=500)

        low_feature = GaussianModel(prefix='low_')
        pars.update(low_feature.make_params())
        pars['low_center'].set(165, min=160, max=185)
        pars['low_sigma'].set(30, min=5, max=50)
        pars['low_amplitude'].set(1e5, min=0)

        high_feature1 = GaussianModel(prefix='high1_')
        pars.update(high_feature1.make_params())
        pars['high1_center'].set(320, min=270, max=350)
        pars['high1_sigma'].set(30, min=5, max=80)
        pars['high1_amplitude'].set(2e5, min=0)

        high_feature2 = GaussianModel(prefix='high2_')
        pars.update(high_feature2.make_params())
        pars['high2_center'].set(380, min=300, max=400)
        pars['high2_sigma'].set(30, min=5, max=80)
        pars['high2_amplitude'].set(2e5, min=0)

        model = background + low_feature + high_feature1 + high_feature2
        result = model.fit(y, pars, x=x)
        components = result.eval_components()
        if show_report:
            print(result.fit_report(min_correl=0.25))
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(x, y, 'bo', markersize=4, label='raw data')
            plt.plot(x, result.best_fit, 'r-', label='best fit')
            plt.xlabel('Temperature [K]')
            plt.ylabel('Mass 2 Counts [Hz]')
            for model_name, model_value in components.items():
                plt.plot(x, model_value, '--', label=model_name)
            plt.legend()
            plt.show()
        return result

    def export_data(self, destination_path):
        columns = [self.filename + '_' + x for x in self.data.columns]
        df_output = pd.DataFrame(self.data.values, columns=columns)
        df_output.to_csv(os.path.join(destination_path, self.filename),
                         index=False)


class CIRD:
    """
    Class for processing CIRD files.

    This class takes a CIRD filename and generate a CIRD object. The main
    functionalities of this class include
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.DataFrame()
        self.filepath = None
        self.path = None
        self.dwell_time = None
        self.index_beamstart = None
        self.index_beamstop = None
        self.time_beamstart = None
        self.time_beamstop = None
        self.integrated_area = None

    def load_data(self, dwell_time=0.2, show_plot=True):
        """
        Load data to self.data as a pandas dataframe and plot the raw CIRD
        spectrum.

        Parameters
        ----------
        dwell_time : float
            Time spent to collect one datapoint. Defaut value is 0.2
        show_plot : bool
            If True, show the raw CIRD spectrum, otherwise if False.
            Default is True.

        Returns
        -------
        """

        month_dict = generate_month_dict()
        month_folder = month_dict[self.filename[:3]] + '_' + self.filename[:3]
        self.filepath = os.path.join(os.path.expanduser('~'),
                                     'Dropbox (MIT)',
                                     'littlemachine',
                                     '20' + self.filename[6:8],
                                     month_folder,
                                     self.filename)

        self.data = pd.read_csv(self.filepath,
                                delim_whitespace=True,
                                skipfooter=3,
                                skiprows=3,
                                names=["counts", "temp"],
                                engine="python")

        self.data.drop('temp', axis=1, inplace=True)
        self.data["time"] = dwell_time * np.arange(len(self.data))
        self.dwell_time = dwell_time
        if show_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data["time"], self.data["counts"])
            plt.xlabel('Time [s]')
            plt.ylabel('Counts [Hz]')
            plt.grid()
            plt.show()

    def level_background(self, npts=50, sensitivity=1.0, beam_start_time=np.nan,
                         beam_stop_time=np.nan):
        """
        Level the CIRD plot,
        Parameters
        ----------
        npts: int
            Total number of datapoints used to fit the linear background.
            Default is 50.
        sensitivity: float
            Sensitivity factor of the mass spec for a specific experiment.
            Default is 1.0
        beam_start_time: int
            If is None (np.nan), it tries to automatically find the beam start.
            If is specified, it will be manually input.
            Default is np.nan.
        beam_stop_time: int
            If is None (np.nan), beam_stop_time needs to be manually input.
            If is specified, will confirm and use the specified value.
            Default is np.nan.

        Returns
        -------
        """
        if len(self.data) == 0:
            self.load_data()

        if np.isnan(beam_start_time):
            self.index_beamstart = np.argmax(np.diff(self.data["counts"])) + 1
            self.time_beamstart = self.data["time"].iloc[self.index_beamstart]
            assert self.index_beamstart - npts >= 0, \
                "Not enough points before beam start! Consider adjusting npts!"
        else:
            self.time_beamstart = beam_start_time
            self.index_beamstart = int(beam_start_time / self.dwell_time)

        if np.isnan(beam_stop_time):
            self.time_beamstop = float(input("Estimated beam stop time: "))
        else:
            print('Current choice of beam stop time is: {}'
                  .format(beam_stop_time))
            if input('Choose again? Y/N: ') in ['Y', 'y']:
                self.time_beamstop = float(input("Estimated beam stop time: "))
            else:
                self.time_beamstop = beam_stop_time

        self.index_beamstop = int(self.time_beamstop / self.dwell_time)
        assert self.index_beamstop + npts < len(self.data), \
            "Invalid choice of beam stop! Consider lowering beam stop time or \
            reduce npts for background correction."

        self.data["counts_leveled"] = self.data["counts"] / sensitivity

        X = pd.concat([
            self.data["time"][(self.index_beamstart-npts):self.index_beamstart],
            self.data["time"][self.index_beamstop:self.index_beamstop+npts]
        ])
        y = pd.concat([
            self.data["counts_leveled"][
                (self.index_beamstart-npts):self.index_beamstart],
            self.data["counts_leveled"][
                self.index_beamstop:self.index_beamstop + npts]
        ])

        slope, intercept = linregress(X, y)[0:2]
        self.data["counts_leveled"] = (self.data["counts_leveled"]
                                       - self.data["time"] * slope - intercept)

    def integrate_area(self):
        """
        Calculate the area under under the curve in CIRD plot, and smooth the
        leveled plot using savgol filter.
        """
        self.integrated_area = self.data["counts_leveled"].iloc[
            self.index_beamstart:self.index_beamstop].sum() * self.dwell_time
        self.data['counts_smoothed'] = self.data["counts_leveled"]
        self.data['counts_smoothed'].iloc[:self.index_beamstart] = \
            savgol_filter(
                self.data['counts_leveled'].iloc[:self.index_beamstart], 51, 1)
        self.data['counts_smoothed'].iloc[self.index_beamstart:] = \
            savgol_filter(
                self.data['counts_leveled'].iloc[self.index_beamstart:], 51, 1)

    def fit_area(self, area_high=False, show_plot=True):
        X = self.data["time"].iloc[self.index_beamstart:self.index_beamstop]
        y = self.data["counts_leveled"].iloc[
            self.index_beamstart:self.index_beamstop]
        p0 = [300, 300, -0.1, -0.02, self.time_beamstart,
              self.time_beamstart, 0]
        param = curve_fit(biexp_decay, X, y, p0=p0, maxfev=3000)[0]
        self.data["exponential_fit"] = biexp_decay(self.data["time"], *param)
        self.data["exponential_fit"][:self.index_beamstart] = 0
        self.data["exponential_fit"][self.index_beamstop:] = 0
        area_param = param.copy()

        if not area_high:
            area_param[-1] = 0
        area = quad(
            biexp_decay, self.time_beamstart, self.time_beamstop,
            args=tuple(area_param)
        )[0]
        if show_plot:
            self.data.set_index("time")[
                ["exponential_fit", "counts_leveled"]].plot(figsize=(10, 5))
            plt.fill_between(
                X, biexp_decay(X, *param),
                0 if area_high else param[-1])
            plt.text(0, 500, "AREA = {}".format(round(area)), fontsize=12)
            plt.show()
        return area

    def export_data(self, destination_path, npts=100):
        df_output = self.data[self.index_beamstart - npts:].copy()
        df_output["time"] = df_output["time"] - df_output["time"].min()
        df_output.columns = [self.filename + '_' + x for x in self.data.columns]
        df_output.to_csv(
            os.path.join(destination_path, self.filename), index=False)

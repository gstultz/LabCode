import os
import pandas as pd
from numpy import exp
import numpy as np
from scipy.stats import linregress
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


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


def generate_month_dict():
    """
    Return a dictionary that maps month names to digital format.
    """
    keys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    values = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']
    return dict(zip(keys, values))


def biexp_decay(x, a1, a2, b1, b2, c1, c2, d):
    """Expression for bi-exponential decay"""
    return a1 * exp(b1 * (x - c1)) + a2 * exp(b2 * (x - c2)) + d

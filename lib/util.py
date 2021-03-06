import os
import numpy as np
from numpy import exp, where, sqrt, pi


def locate_filepath(filename):
    """
    Locate the data filepath.
    Note:
        MIT dropbox folder should be placed in the home directory.
        OS independent.
        Filepath is located through the datafile name.
    """
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


def index_of(arr, value):
    """
    Return the largest index of an element in an array that is smaller
    than a given value. Array must be sorted.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
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


def exp_decay(x, a, b, c):
    """Expression for exponential decay"""
    return a * exp(b * x) + c

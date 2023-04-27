# io_data.py is used to read data into generative model
# filename is the file that stored the simulated/experimental data
# all the processed data are in the same form, saved in DataArray

import numpy as np
import xarray as xr

def io_data(filename):
    """
    Returns simulated / experimental data in xarray.Dataset
    ---
    Parameters:
        filename: name of .nc file 
    """
    da = xr.open_dataset(filename)
    return da

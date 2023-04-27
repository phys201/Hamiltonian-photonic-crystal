# io_data.py is used to read data into generative model
# filename is the file that stored the simulated/experimental data
# all the processed data are in the same form, saved in DataArray

import numpy as np
import xarray as xr

def io_data(filename):
    """
    da: 'normf' 'spectrum' 'spectrum_std'
    meata data (exp): information of raw data incluting chip name, measurement condition
    meta data (simu): information of simulation paramenters
    """
    da = xr.open_dataset(filename)
    #print('information of loaded data: '+ da.attrs)
    return da

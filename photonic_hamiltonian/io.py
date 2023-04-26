import numpy as np
import xarray as xr

def load_data(filename):
    """
    returns a xarray of the data in a .npy format data file
    
    Parameters:
        filename (Str): filename of the .npy data file, make sure to include the correct file path    
        
    Return:
        data (Xarray): data with labels 'normf', 'spectrum', 'spectrum_std'
    """
    np_data = np.load(filename)
    x = np_data[0]
    y = np_data[1]
    dy = np_data[2]
    data = xr.Dataset(data_vars = {'spectrum': ('normf', y),
                                  'spectrum_std': ('normf', dy)},
                     coords = { 'normf': x})
    return data

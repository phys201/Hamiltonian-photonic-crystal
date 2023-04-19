import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as sg
from matplotlib import rcParams
from scipy import interpolate 

    
def io(filename):
    da = xr.open_dataset(filename)
    print(da)
    return da
    # def __init__(self,filename):
    #     self.da = xr.open_dataset(filename)
    #     return self.da

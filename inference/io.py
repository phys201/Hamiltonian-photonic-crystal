import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import pymc as pm
import scipy.signal as sg
from matplotlib import rcParams
import cv2 as cv
from scipy import interpolate 
def io(filename):
    da = xr.open_dataset(filename)
    return da

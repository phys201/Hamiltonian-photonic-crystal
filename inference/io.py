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

# load data cube, I (f,kx,ky), the datacube contains information from all the iso-frequency contour pictures
# it has be corrected (distortion, HDR, etc) 
filename = 'm18_sk_HDR_trans_p15_cor__OPR8_power20-20-100_twist_mapped_cube_1100_2_1700_lim=0.6.npy'
datacube = np.load(filename)

def get_rawband_new(datacube,GX,GM,dx=0,dm=0):
    """
    return band structure in a picked quadrant from 3D datacube
    ---
    datacube: 3D band structrue, all information from exp (np, 3D datacube)
    GX: 1,2,3,4     int
    GM: 1,3         int
    dx: shift of GX   int
    dm: shift of GM   int
    ---
    return gammam_raw: bandstructure along high symmetry k path(2D datacube). gammams_raw & gammamx_raw is part of it. 
    """
    mapped_cube=datacube    
    #read basic parameters
    dim=np.shape(mapped_cube)
    num_fps=dim[0]
    res=dim[1]
    
    #gammax_raw
    if GX == 1:
        gammax_raw = mapped_cube[:,res//2+dx,:]
        gammax_raw = gammax_raw[:,res//2:res]
    elif GX == 3:
        gammax_raw = mapped_cube[:,res//2+dx,:]
        gammax_raw = np.fliplr(gammax_raw[:,0:res//2])


    #gammam_raw, rescaled
    if GM == 1:
        gammam_raw = np.diagonal(mapped_cube,offset=dm,axis1=1,axis2=2)
        scale_percent = np.sqrt(2) # percent of original size
        width = int(res*scale_percent)
        height = int(gammam_raw.shape[0])
        dim = (width, height)
        gammam_raw = cv.resize(gammam_raw, dim, interpolation = cv.INTER_AREA)  # 2D reshaped
        gammam_raw = gammam_raw[:,int(width//2-res//2):int(width//2)]
    elif GM == 2:
        fmap = np.fliplr(mapped_cube)
        gammam_raw = np.diagonal(fmap,offset=dm,axis1=1,axis2=2)
        scale_percent = np.sqrt(2) # percent of original size ==fliput 
        width = int(res*scale_percent)
        height = int(gammam_raw.shape[0])
        dim = (width, height)
        gammam_raw = cv.resize(gammam_raw, dim, interpolation = cv.INTER_AREA)  # 2D reshaped
        gammam_raw = gammam_raw[:,int(width//2-res//2):int(width//2)]
    elif GM == 3: 
        gammam_raw = np.diagonal(mapped_cube,offset=dm,axis1=1,axis2=2)
        scale_percent = np.sqrt(2) # percent of original size
        width = int(res*scale_percent)
        height = int(gammam_raw.shape[0])
        dim = (width, height)
        gammam_raw = cv.resize(gammam_raw, dim, interpolation = cv.INTER_AREA)  # 2D reshaped
        gammam_raw = np.fliplr(gammam_raw[:,int(width//2):int(width//2+res//2)])
    elif GM == 4:
        fmap = np.fliplr(mapped_cube)
        gammam_raw = np.diagonal(fmap,offset=dm,axis1=1,axis2=2)
        scale_percent = np.sqrt(2) # percent of original size ==fliput 
        width = int(res*scale_percent)
        height = int(gammam_raw.shape[0])
        dim = (width, height)
        gammam_raw = cv.resize(gammam_raw, dim, interpolation = cv.INTER_AREA)  # 2D reshaped
        gammam_raw = np.fliplr(gammam_raw[:,int(width//2):int(width//2+res//2)])

    #make gammamx    
    gammamx_raw=np.zeros((num_fps,res))
    gammamx_raw[:,0:res//2]=gammam_raw
    gammamx_raw[:,res//2:res]=gammax_raw     # gx half
    return [gammam_raw,gammax_raw,gammamx_raw]

# basic information related to experiment
numf,numk,numk_ = datacube.shape
res = numk      # resolution in k spaces
lim = 0.3       # limitation of k range, in unit [2*pi/a], a is the lattice constant
c0 = 299792458  # speed of light, in unit [m/s]
a = 1000e-9     # lattice constant of photonic crystal, in unit [m]
wavelength = np.linspace(1100,1700,numf)     # unit in nm
frequency = c0/(wavelength*1e-9)*1e-12       # unit in THz
kx = np.linspace(-0.3,0.3,numk)              # unit in 2*pi/a
ky = kx

# extrat single k spectrum from raw data, get spectrum at 4 (kx,ky) location (C4 symmetry)
spcube = []
M = 0.05     # chose k point, along Gamma - M, M range(0,0.3), 0 is Gamma point
M_i = int((0.3-M)/0.6*numk)
GX = 1
dx = 0
dm = 0

xnew = np.linspace(frequency[0],frequency[-1],numf)    
ynew = np.linspace(-lim,lim,res)     

spectra = np.zeros((4,numf))
fnew =np.zeros((301,320))
for i in range(4):
    GM = i+1
    [tmp_m,tmp_x,tmp_mx]=get_rawband_new(datacube,GX,GM,dx,dm)    
    newfunc = interpolate.interp2d(np.linspace(-lim,lim,res),frequency, tmp_mx, kind='cubic')
    ff = newfunc(ynew, xnew)
    fnew =ff +fnew
    spectra[i] = np.mean(ff[:,M_i-1:M_i+1],axis = 1)
    
fnew=fnew[::-1,:]

# output extracted spectrum and metadata
normf = frequency*1e12*a/c0                  # frequency normalization
spectrum = np.mean(spectra,axis = 0)         # mean of 4 symmetry point intensity values
spectrum_std = np.std(spectra,axis = 0)      # standard deviation of 4 symmetry points intensity values
metadata_all = filename[:-4] 
metadata_list = metadata_all.split("_")

da = xr.DataArray(np.array([normf, spectrum,spectrum_std]).T,
                  dims=('index', 'variable'),
                  coords={'index': np.arange(len(normf)),
                          'variable': ['normf', 'spectrum', 'spectrum_std']})
da.attrs['metadata'] = metadata_all
da.to_netcdf("expdata_GM"+str(M)+".nc")

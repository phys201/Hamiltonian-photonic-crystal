import unittest
from unittest import TestCase
import numpy as np
import xarray as xr
from inference import io_data


filename = "data\expdata_singleKspectrum.nc"
data = io_data.load_data(filename)
class TestDataConversion(TestCase):
    
    def test_data_is_returned(self):
        self.assertTrue(isinstance(io_data.load_data(filename), xr.core.dataset.Dataset))  
        
    def test_data_file_notexist(self):
        self.assertRaises(FileNotFoundError, io_data.load_data, 'expdata.nc')
        
    def test_key_not_in_data(self):
        self.assertRaises(KeyError, data.__getitem__, 'abc')
        
    def test_key_in_data(self):
        self.assertTrue(data['normf'].all())
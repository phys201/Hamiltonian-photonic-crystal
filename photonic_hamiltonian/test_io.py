from unittest import TestCase
import numpy as np
import xarray as xr
import io


filename = "example_data.npy"
data = io.load_data(filename)
class TestDataConversion(TestCase):
    
    def test_data_is_returned(self):
        self.assertTrue(isinstance(io.load_data(filename), xr.DataSet))  
        
    def test_data_file_notexist(TestCase):
        self.assertRaises(FileNotFoundError, io.load_data("example_data.nc"))
        
    def test_key_not_in_data(self):
        self.assertRaises(KeyError, data['abc'])
        
    def test_key_in_data(self):
        self.assertTrue(data['normf'])
    


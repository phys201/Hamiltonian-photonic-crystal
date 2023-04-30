# this file contains tests of prior.py
import unittest
from unittest import TestCase
import numpy as np
import pymc as pm
from inference import Model, io_data


filename = "data\expdata_singleKspectrum.nc"
data = io_data.load_data(filename)
prior_bounds_standard = {'u11':[0.007, 0.01], 'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
    
test_Model = Model.Hamiltonian_model(data, prior_bounds_standard)
class TestModelConstruction(TestCase):
    def test_Model_is_returned(self):
        self.assertTrue(isinstance(test_Model, pm.Model))


    def test_initial_val_outside_bound(self):
        start = {'u11_interval__':-np.Inf, 'u20_interval__':0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'peak_width_interval__':0.001} 
        self.assertRaises(pm.exceptions.SamplingError, test_Model.check_start_vals, start)

    def test_initial_val_inside_bound(self):
        start1 = {'u11_interval__':0, 'u20_interval__':0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'peak_width_interval__':0.001} 
        self.assertEqual(None, test_Model.check_start_vals(start1))


    

prior_bounds2 = {'u11': [1, 0],'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
class TestPriorProperties(TestCase):
    def test_upper_cannot_be_smaller_than_lower(self):
        self.assertRaises(ValueError, Model.Hamiltonian_model, data, prior_bounds2)

start2 = start = {'u11_interval__':-np.Inf, 'u20_interval__':0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'peak_width_interval__':-np.Inf}
class TestLikelihoodProbability(TestCase):
    def test_peak_width_positive(self):
        self.assertRaises(pm.exceptions.SamplingError, test_Model.check_start_vals, start2)
        

        
u11 = 0.007
u20 = 1e-5
A0 = 0.5
A1, A2, A3, A4 = [1, 1.2, 1.5, 1]
An = [A1, A2, A3, A4]
sigma_L = 0.002
theta = [u11, u20, A0, A1, A2, A3, A4, sigma_L]
ex = 0.669
ey = 0.6346
Cn = np.real(np.linalg.eigvals([[ex,u11,u20,u11],
                                [u11,ey,u11,u20],
                                [u20,u11,ey,u11],
                                [u11,u20,u11,ex]]))
Cn = np.sort(Cn)
freq = 0.6
intensity = A0 + np.sum([Ai * np.exp(-(freq - Ci)**2 / (2 * sigma_L**2)) for Ai, Ci in zip(An, Cn)])

class TestFitCurve(TestCase):
    def test_fit_result(self):
        self.assertAlmostEqual(Model.fit_curve(freq, theta), intensity)
    def test_insufficient_parameters(self):
        theta_insuff = [u11, u20, A0, A1, A2, A3]
        self.assertRaises(ValueError, Model.fit_curve, freq, theta_insuff)
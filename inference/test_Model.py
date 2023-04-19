# this file contains tests of prior.py
from unittest import TestCase
import numpy as np
import pymc as pm

from inference import Model
from inference import io


filename = "simudata_GM0.05.nc"
data = io.io(filename) #put some filename here
prior_bounds_standard = {'h11':[0.003, 0.01], 'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
    
class TestModelConstruction(TestCase):
    def test_Model_is_returned(self):
        test_Model = Model.Hamiltonian_model(data, prior_bounds_standard)
        self.assertTrue(isinstance(test_Model, pm.Model))

    

    def test_initial_val_outside_bound(self):
        start = {'h11':0, 'u20':0, 'A0', 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]} #change all the values into number
        test_model = Model(data, prior_bounds)
        #test_initial_val = test_model.check_start_vals(start)
        self.assertRaises(pymc.exceptions.SamplingError, test_model.check_start_vals(start))

    def test_initial_val_inside_bound(self):
        self.assertEqual(None, test_model.check_start_vals(start1))

    

prior_bounds2 = {'h11': [1, 0],'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
class TestPriorProperties(TestCase):
    def test_upper_cannot_be_smaller_than_lower(self):
        self.assertRaises(ValueError, Model(data, prior_bounds2))


class TestLikelihoodProbability(TestCase):
    def test_peak_width_positive(self):
        test_uniform = prior.Uniform()
        self.assertEqual(test_uniform.p(0.5), 1)

class TestModelLogp(TestCase):
    point = {'h11':..., )
    hamiltonian = 
    Cn = np.eig...(hamli)
    likelihood = ...
    prior = log(1/(upper-lower))
    posterior = prior+likelihood
    def test_value(self):
        test_model = Model(data, bounds)
        self.assertAlmostEqual(test_model.point_logps([point]), posterior)



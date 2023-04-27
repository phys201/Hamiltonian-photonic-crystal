from unittest import TestCase
import numpy as np
import pymc as pm

import Model, io


filename = "example_data.npy"
data = [1,2,3] #put some filename here
prior_bounds_standard = {'h11':[0.007, 0.01], 'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
    
test_Model = Model.Hamiltonian_model(data, prior_bounds_standard)
class TestModelConstruction(TestCase):
    def test_Model_is_returned(self):
        self.assertTrue(isinstance(test_Model, pm.Model))

    def test_initial_val_outside_bound(self):
        start = {'h11':0, 'u20':0, 'A0':1.5, 'A1':3,'A2':3, 'A3':3, 'A4':3, 'peak_width':0.001} 
        self.assertRaises(pm.exceptions.SamplingError, test_Model.check_start_vals(start))

    def test_initial_val_inside_bound(self):
        start1 = {'h11':0.007, 'u20':1e-4, 'A0':1, 'A1':1,'A2':1, 'A3':1, 'A4':1, 'peak_width':0.008} 
        self.assertEqual(None, test_Model.check_start_vals(start1))


prior_bounds2 = {'h11': [1, 0],'u20':[-1e-3, 1e-3], 'A0':[0, 1.2], 'A1':[0.5, 2.5],'A2':[0.5, 2.5], 'A3':[0.5, 2.5], 'A4':[0.5, 2,5], 'peak_width':[0.003, 0.01]}
class TestPriorProperties(TestCase):
    def test_upper_cannot_be_smaller_than_lower(self):
        self.assertRaises(ValueError, Model.Hamiltonian_model(data, prior_bounds2))

start2 = {'h11':0, 'u20':0, 'A0':1.5, 'A1':3,'A2':3, 'A3':3, 'A4':3, 'peak_width':-0.005} 
class TestLikelihoodProbability(TestCase):
    def test_peak_width_positive(self):
        self.assertLessEqual(0,test_Model.check_start_vals(start2))

# class TestModelLogp(TestCase):
#     point = {'h11':..., )
#     hamiltonian = 
#     Cn = np.eig...(hamli)
#     likelihood = ...
#     prior = log(1/(upper-lower))
#     posterior = prior+likelihood
#     def test_value(self):
#         test_model = Model(data, bounds)
#         self.assertAlmostEqual(test_model.point_logps([point]), posterior)




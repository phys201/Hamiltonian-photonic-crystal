# this file contains tests of prior.py
import unittest
from unittest import TestCase
import numpy as np
import pymc as pm
from inference import Model, io_data
        
u11 = 0.007
u20 = 1e-5
e0 = 0.66
de = 0.01
A0 = 0.5
An = [1, 1.2, 1.5, 1]
A1, A2, A3, A4 = An
Qn = [0.003]*4
Q1, Q2, Q3, Q4 = Qn
theta = [u11, u20, e0, de, A0, A1, A2, A3, A4, Q1, Q2, Q3, Q4]
ex = e0 + de
ey = e0 - de
Cn = np.real(np.linalg.eigvals([[ex,u11,u20,u11],
                                [u11,ey,u11,u20],
                                [u20,u11,ey,u11],
                                [u11,u20,u11,ex]]))
Cn = np.sort(Cn)
Wn = Cn/Qn
freq = 0.6
intensity = A0 + np.sum([(Ai * Wi**2) / ((freq- Ci)**2 + Wi**2) for Ai, Ci, Wi in zip(An, Cn, Wn)])

class TestPredictionModel(TestCase):
    def test_fit_result(self):
        self.assertAlmostEqual(Model.prediction_model(theta, freq), intensity)
    def test_insufficient_parameters(self):
        theta_insuff = [u11, u20, A0, A1, A2, A3]
        self.assertRaises(ValueError, Model.prediction_model, theta_insuff, freq)


filename = "data\expdata_singleKspectrum_055.nc"
data = io_data.load_data(filename)
priors_example = {'u11':('Uniform', [0.007, 0.01]), 
                  'u20':('Uniform', [-1e-3, 1e-3]), 
                  'e0':('Uniform', [0.64, 0.68]),
                  'de':('Uniform', [0.01, 0.02]),
                  'A0':('Uniform', [0, 1.2]), 
                  'A1':('Uniform', [0.5, 2.5]), 
                  'A2':('Uniform', [0.5, 2.5]), 
                  'A3':('Uniform', [0.5, 2.5]), 
                  'A4':('Uniform', [0.5, 2.5]), 
                  'Q1':('Uniform', [0.003, 0.01]),
                  'Q2':('Uniform', [0.003, 0.01]),
                  'Q3':('Uniform', [0.003, 0.01]),
                  'Q4':('Uniform', [0.003, 0.01]), 
                  'sigma_y':('Uniform', [0.05, 0.2])}
    
model_test = Model.Hamiltonian_model(data, priors_example)
class TestModelConstruction(TestCase):
    def test_Model_is_returned(self):
        self.assertTrue(isinstance(model_test, pm.Model))


    def test_initial_val_outside_bound(self):
        start = {'u11_interval__':-np.Inf, 'u20_interval__':0, 'e0_interval__':1, 'de_interval__': 0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'Q1_interval__':0.001, 'Q2_interval__':0.001, 'Q3_interval__':0.001, 'Q4_interval__':0.001, 'sigma_y_interval__':0.001} 
        self.assertRaises(pm.exceptions.SamplingError, model_test.check_start_vals, start)

    def test_initial_val_inside_bound(self):
        start1 = {'u11_interval__':0, 'u20_interval__':0, 'e0_interval__':1, 'de_interval__': 0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'Q1_interval__':0.001, 'Q2_interval__':0.001, 'Q3_interval__':0.001, 'Q4_interval__':0.001, 'sigma_y_interval__':0.001} 
        self.assertEqual(None, model_test.check_start_vals(start1))
    

priors_example_2 = priors_example
priors_example_2['u11'] = ('Binomial', [0, 1])
class TestPriorProperties(TestCase):
    def test_prior_type_not_allowed(self):
        self.assertRaises(ValueError, Model.Hamiltonian_model, data, priors_example_2)

class TestLikelihoodProbability(TestCase):
    def test_peak_width_positive(self):
        start2 = {'u11_interval__':0, 'u20_interval__':0, 'e0_interval__':1, 'de_interval__': 0, 'A0_interval__':1.5, 'A1_interval__':3,'A2_interval__':3, 'A3_interval__':3, 'A4_interval__':3, 'Q1_interval__':-np.Inf, 'Q2_interval__':0.001, 'Q3_interval__':0.001, 'Q4_interval__':0.001, 'sigma_y_interval__':0.001} 
        self.assertRaises(pm.exceptions.SamplingError, model_test.check_start_vals, start2)

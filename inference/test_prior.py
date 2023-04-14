# this file contains tests of prior.py
from unittest import TestCase

import numpy as np

import prior

STANDARD_NORM = 1 / np.sqrt(2 * np.pi)
STANDARD_1_SD = STANDARD_NORM * np.exp(-1/2)


class TestUniformConstruction(TestCase):
    def test_Uniform_is_Prior_subclass(self):
        test_uniform = prior.Uniform(0, 1)
        self.assertTrue(isinstance(test_uniform, prior.Prior))

    def test_values_stored_correctly(self):
        lower, upper = (1, 2)
        test_uniform = prior.Uniform(lower, upper)
        self.assertEqual(test_uniform.lower, lower)
        self.assertEqual(test_uniform.upper, upper)

    def test_bounds_cannot_be_equal(self):
        self.assertRaises(ValueError, prior.Uniform, 0, 0)

    def test_upper_cannot_be_smaller_than_lower(self):
        self.assertRaises(ValueError, prior.Uniform, 0, -1)


class TestUniformProperties(TestCase):
    def test_default_values(self):
        test_uniform = prior.Uniform()
        self.assertEqual(test_uniform.lower, 0)
        self.assertEqual(test_uniform.upper, 1)

    def test_range(self):
        test_uniform = prior.Uniform(1, 3)
        self.assertEqual(test_uniform.range, 2)


class TestUniformProbability(TestCase):
    def test_inside_range(self):
        test_uniform = prior.Uniform()
        self.assertEqual(test_uniform.p(0.5), 1)

    def test_outside_range(self):
        test_uniform = prior.Uniform()
        self.assertEqual(test_uniform.p(2), 0)
        self.assertEqual(test_uniform.p(-1), 0)

    def test_wider_bounds(self):
        test_uniform = prior.Uniform(5, 10)
        self.assertEqual(test_uniform.p(7), 0.2)
        self.assertEqual(test_uniform.p(11), 0)


class TestGaussianConstruction(TestCase):
    def test_Gaussian_is_Prior_subclass(self):
        test_gaussian = prior.Gaussian(0, 1)
        self.assertTrue(isinstance(test_gaussian, prior.Prior))

    def test_values_stored_correctly(self):
        mean, sd = (1, 2)
        test_gaussian = prior.Gaussian(mean, sd)
        self.assertEqual(test_gaussian.mean, mean)
        self.assertEqual(test_gaussian.stdev, sd)

    def test_sd_must_be_positive(self):
        self.assertRaises(ValueError, prior.Gaussian, 0, 0)
        self.assertRaises(ValueError, prior.Gaussian, 0, -1)


class TestGaussianProperties(TestCase):
    def test_default_values(self):
        test_gaussian = prior.Gaussian()
        self.assertEqual(test_gaussian.mean, 0)
        self.assertEqual(test_gaussian.stdev, 1)

    def test_gaussian_variance(self):
        test_gaussian = prior.Gaussian(0, 2)
        self.assertEqual(test_gaussian.variance, 4)


class TestGaussianProbability(TestCase):
    def test_peak_value(self):
        test_gaussian = prior.Gaussian(0, 1)
        probability_at_peak = test_gaussian.p(0)
        self.assertEqual(probability_at_peak, STANDARD_NORM)

    def test_1_sd_value(self):
        test_gaussian = prior.Gaussian(0, 1)
        probability_at_1_sd = test_gaussian.p(1)
        self.assertEqual(probability_at_1_sd, STANDARD_1_SD)

    def test_nonstandard_normals(self):
        means = (2, -1, 3)
        stdevs = (0.5, 3, 9)
        for mean, sd  in zip(means, stdevs):
            test_gaussian = prior.Gaussian(mean, sd)
            self.assertAlmostEqual(test_gaussian.p(mean), STANDARD_NORM / sd)
            self.assertAlmostEqual(test_gaussian.p(mean + sd), STANDARD_1_SD / sd)

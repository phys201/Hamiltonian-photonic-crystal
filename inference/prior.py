# This file contains the Prior class and subclasses
from multiprocessing.sharedctypes import Value
import numpy as np


class Prior():
    def logp(self,value):        
        return np.log(self.p(value))
    def sample(self):
        raise NotImplementedError

class Uniform(Prior):
    def __init__(self, lower=0, upper=1):
        '''Uniform prior with given lower and upper bounds'''
        if self.upper != -np.inf and self.upper != np.inf:
            if not upper > lower:
                raise ValueError('upper bound must be greater than lower bound')
        
            self.upper = upper
            self.lower = lower
        # precalculate normalization since it's always the same
            self.norm = 1 / self.range 
        else:
            raise ValueError('prior range must be finite')
            
    @property
    def range(self):
        return self.upper - self.lower

    def p(self, value):
        '''Probability density of input value'''
        if self.lower < value < self.upper:
            return self.norm
        else:
            return 0


class Gaussian(Prior):
    def __init__(self, mean=0, stdev=1):
        '''Gaussian prior with given mean and standard deviation'''
        self.mean = mean
        if stdev <= 0:
            raise ValueError("standard deviation must be positive")
        else:
            self.stdev = stdev
        # precalculate normalization since it's always the same
        self.norm = 1/np.sqrt(2 * np.pi * self.stdev**2)

    @property
    def variance(self):
        return self.stdev**2

    def p(self, value):
        '''Probability density of input value'''
        return self.norm * np.exp(-(value-self.mean)**2/self.stdev**2/2)



class Jeffereys(Prior):
    def __init__(self, lower=0, upper=1):
        '''Uniform prior with given lower and upper bounds'''
        if not upper > lower:
            raise ValueError('upper bound must be greater than lower bound')
        self.upper = upper
        self.lower = lower
        # precalculate normalization since it's only dependent on the max and min
        self.norm = 1/np.log(self.upper/self.lower)

    @property
    def range(self):
        return self.upper - self.lower

    def p(self, value):
        '''Probability density of input value'''
        if self.lower < value < self.upper:
            return 1/value* self.norm
        else:
            return 0
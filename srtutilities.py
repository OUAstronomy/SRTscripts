#!/usr/bin/env python
'''
Name  : SRT Utilities, srtutilities.py
Author: Nickalas Reynolds
Date  : Fall 2017
Misc  : File will handle misc srt specific function
'''

# import modules
from os import system as _SYSTEM_
from sys import exit

class locations(object):

    def __init__(self):
        self.locations = {'norman':['35d13m21.2s','-97d26m22.1s',370,-4],\
                          'apo':['','',2300,-5]}

    def get_locations(self):
        return ', '.join([i for i in self.locations])

    def add_location(self,location):
        if not self.verify_location(location[0]):
            try:
                if len(location) == 5:
                    self.locations[location[0]] = [location[1],location[2],float(location[3]),int(location[4])]
                else:
                    raise RuntimeError('Incorrect format, please follow: \n\
                                        [{},{}] which is [name,lat,long,height,timezone] and \n\
                                        lat is North bias, long is East Bias, height is in meters, timezone is integer from utc\
                                        '.format('norman',','.join(self.locations['norman'])))
            except IndexError:
                raise RuntimeError('Incorrect format, please follow: \n\
                                    [{},{}] which is [name,lat,long,height,timezone] and \n\
                                    lat is North bias, long is East Bias,height is in meters, timezone is integer from utc\
                                    '.format('norman',','.join(self.locations['norman'])))

    def verify_location(self,checking):
        if checking in self.locations:
            return True
        else:
            return False

class special(object):

    def fwhm(self,sig):
        import numpy as np
        return 2. * ((2. * np.log10(2.)) ** 0.5) * (sig)

    def get_params(self):
        return self.params

    def set_params(self,params):
        self.params = params

    # single gaussian
    def gaussian(self,x,mu,sig,A):
        from numpy import inner,exp
        vector = inner(x - mu,x - mu)
        return A * exp(-0.5 * (vector/sig)** 2)

    # multigaussian single dimension
    def multigauss(self,x,mugrid,siggrid,Agrid):
        total = []
        for mu,sig,A in zip(mugrid,siggrid,Agrid):
            temp = self.gaussian(x,mu,sig,A)
            total += temp
        if len(total) != len(x):
            raise RuntimeError('Shape of multigauss and x range not the same')
            exit()
        else:
            return total

    def polynomial(self,x,coeff):
        total = []
        if (type(coeff) != list) or (type(coeff) != np.ndarray):
            coeff = [coeff,]
        for order in range(len(coeff)):
            temp = coeff[order] * x[:] ** order
            total += temp
        if len(total) != len(x):
            raise RuntimeError('Shape of polynomial and x range not the same')
            exit()
        else:
            return total

    # bessel function
    def bessel2(self,x ,order):
        from scipy.special import yv
        return yv(order, x)

if __name__ == "__main__":
    print("Testing module...")
    

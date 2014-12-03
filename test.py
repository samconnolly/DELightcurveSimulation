
"""
LCgenerationDE.py

Created on Fri Nov  7 16:18:50 2014

Author: Sam Connolly

Python version of Dimitris' light curve simulation algorithm.
Uses 'Lightcurve' objects to allow easy interactive use and easy plotting. 
Can use any PSD or PDF model, requires best fits to be known.

requires:
    numpy, pylab, scipy, math

"""



#--------- Modules ---------------------------------------------------------------

import time
import scipy.stats as st
from DELCgen import *
import math

def Gamma(x,kappa,theta):
    '''
    Gamma function - returns gamma(x), where x is an array
    
    inputs:
        x (array or float)   - input values
        kappa (float)        - power
        theta (float)        - exponent
    output:
        out (array or float) - output values
    '''
    if type(x) != float:
        d = (x > 0)*x
    elif x > 0:
        d = x
    else:
        return 0
    p1 = -d/theta
    p2 = kappa - 1.0
    out = np.exp(p1) * d**p2
    out = out*(x > 0)
    g = math.gamma(kappa)       # this is a gamma funtion, not distribution...
    norm = theta**kappa *g   
    out /= norm
    return out
    
def LogNormal(x,mu,sig):
    '''
    Log Normal function    - returns LogNormal(x), where x is an array
    
    inputs:
        x (array or float) - input values
        mu (float)         - average of target distribution
        sig (float)        - standard deviation of target distribution
    output:
        out (array or float) - output values
    '''
    if type(x) != float:
        d = (x > 0)*x + (x <= 0)
    elif x > 0:
        d = x
    else:
        return 0
    denom = d * sig * np.sqrt(2.0*np.pi)
    p_num = (np.log(d) - mu)**2.0
    p_den = 2.0*(sig**2.0)
    out = (1.0/denom) * np.exp(-p_num/p_den)
    out = out*(x > 0)
    return out

# File Route
route = "/Users/sdc1g08/Python/DELightcurveSimulation/"
datfile = "NGC4051.dat"

# Bending power law params
A,v_bend,a_low,a_high,c = 0.03, 2.3e-4, 1.1, 2.2, 0 
# Probability density function params
kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82
# Simulation params
RedNoiseL,RandomSeed,aliasTbin, tbin = 100,12,1,100 

#--------- Commands ---------------

# load data lightcurve
datalc = Load_Lightcurve(route+datfile)

# plot the data lightcurve and its PDF and PSD
#datalc.Plot_Lightcurve()

# estimate underlying variance od data light curve
datalc.STD_Estimate(BendingPL,(A,v_bend,a_low,a_high,c))

# simulate artificial light curve with Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                RedNoiseL,aliasTbin,RandomSeed)

# simulate artificial light curve with Emmanoulopoulos method COMPARE!
start_time = time.time()
delc = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([st.gamma,st.lognorm],[[kappa,0, theta],\
                                        [lnsig,0, np.exp(lnmu)]],[weight,1-weight]))
print "Inverse transform sampling time:",time.time()-start_time
start_time = time.time()
delc2 = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([[Gamma,LogNormal],[[kappa, theta],\
                                        [lnmu, lnsig]],[weight,1-weight]]),MixtureDist)
print "Rejection sampling time:",time.time()-start_time
# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc,delc2])

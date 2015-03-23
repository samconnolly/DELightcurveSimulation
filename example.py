
"""
DELCuse.py

Created on March  23  2014

Author: Sam Connolly

Example code for using the commands in DELCgen to simulate lightcurves

"""

from DELCgen import *
import scipy.stats as st


#------- Input parameters -------

# File Route
route = "/route/to/your/data/"
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
datalc.Plot_Lightcurve()

# estimate underlying variance of data light curve
datalc.STD_Estimate(BendingPL,(A,v_bend,a_low,a_high,c))

# simulate artificial light curve with Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                RedNoiseL,aliasTbin,RandomSeed)

# simulate artificial light curve with Emmanoulopoulos method, scipy distribution
delc = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([st.gamma,st.lognorm],[[kappa,0, theta],\
                                    [lnsig,0, np.exp(lnmu)]],[weight,1-weight]))

# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc],names=["Data LC","Timmer \& Koenig LC", "Emmanoulopoulos LC"])
#========================
# DELightcurveSimulation
#========================

Python version of Dimitris Emmanoulopoulos' light curve simulation algorithm.

#---------------
# Description:
#---------------

The code uses a 'Lightcurve' class which contains all of the data necessary
for simulation of artificial version, and for plotting. Lightcurve objects
can be easily created from data using the following command:

lc = Load_Lightcurve(fileroute)

Artificial lightcurves can be produced from it using the following commands:

# Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,PSDfunction, PSDparams, RedNoiseL, aliasTbin, RandomSeed)

#Emmanoulopoulos method
delc = Simulate_DE_Lightcurve(datalc,PSDfunction, PSDparams, PDFfunction, PDFparams)

#--- distributions ---
Any function can be used for the PSD and PDF, but the following exist in the
module:

PSD:
    BendingPL(v,A,v_bend,a_low,a_high,c) - Bending power law

PDF:
    Gamma(x,kappa,theta)    - Gamma function
    LogNormal(x,mu,sig)     - Log normal distribution

General:
    MixtureDist(x,f,args,weights) - Mixture distribution of any set of functions

#--- plotting ---
The following commands are attributes of the Lightcurve class:
    Plot_Lightcurve()       - Plot the lightcurve
    Plot_Periodogram()      - Plot the lightcurve's periodogram
    Plot_PDF()              - Plot the lightcurve's probability density function
    Plot_Stats()            - Plot the lightcurve, its periodogram and PDF

The following commands take Lightcurve objects as inputs:
    Comparison_Plots(lightcurves,bins=25,norm=True) - Plot multiple lightcurves
                                                        and their PSDs & PDFs
                                                        
#--- other attributes & methods of the Lightcurve class ----

# attributes
time            - The lightcurve's time array
flux            - The lightcurve's flux array
errors          - The lightcurve's flux error array
length          - The lightcurve's length
freq            - The lightcurve's periodogram's frequency array
psd             - The lightcurve's power spectral density array (if calculated)
mean            - The lightcurve's mean flux
std             - The lightcurve's standard deviation
std_est         - The lightcurve's estimated underlying SD (if calculated)
tbin            - The lightcurve's time bin size
fft             - The lightcurve's Fourier transform (if calculated)
periodogram     - The lightcurve's periodogram (if calculated)

# methods (functions)
STD_Estimate(PSDdist,PSDdistArgs) - Calculate the estimate of the underlying
                                    standard deviation (without Poisson noise),
                                    which is used in simulations if present
Fourier_Transform()               - Calculate the lightcurve's Fourier transform
                                    (calculated automatically if required)
Periodogram()                     - Calculate the lightcurve's periodogram
                                    (calculated automatically if required)

#----------------
# Example usage:
#----------------

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

# estimate underlying variance od data light curve
datalc.STD_Estimate(BendingPL,(A,v_bend,a_low,a_high,c))

# simulate artificial light curve with Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                RedNoiseL,aliasTbin,RandomSeed)

# simulate artificial light curve with Emmanoulopoulos method
delc = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                               MixtureDist, ([[Gamma,LogNormal],[[kappa, theta],\
                                        [lnmu, lnsig]],[weight,1-weight]]))

# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc])

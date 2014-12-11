# Emmanoulopoulos Lightcurve Simulation

####Python version of the Emmanoulopoulos light curve simulation algorithm.
### As according to Emmanoulopoulos et al 2013, 
## Monthly Notice of the Royal Astronomical Society, 433, 907

### Description:

The code uses a 'Lightcurve' class which contains all of the data necessary
for simulation of artificial version, and for plotting. Lightcurve objects
can be easily created from data using the following command:

lc = Load_Lightcurve(fileroute)

Artificial lightcurves can be produced from it using the following commands:

##### Timmer & Koenig (1995) method
From Timmer & Koenig, 1995,
    Astronomy & Astrophysics, 300, 707.
```python
tklc = Simulate_TK_Lightcurve(datalc,PSDfunction, PSDparams, RedNoiseL, aliasTbin, RandomSeed)
```

##### Emmanoulopoulos (2013) method
From Emmanoulopoulos et al., 2013, Monthly Notice of the Royal Astronomical Society, 433, 907.

```python
delc = Simulate_DE_Lightcurve(datalc,PSDfunction, PSDparams, PDFfunction, PDFparams)
```

### Distributions and functions 
Any function can be used for the PSD and PDF, however it is highly recommended
to use a scipy.stats random variate distribution for the PDF, as this allows 
inverse transfer sampling as opposed to rejection sampling, the latter of which
is *much* slower. In the case that a scipy RV function is used, however, care 
should be taken with the parameters, as they may not be what you expect (see
below). In addition, the following functions exist in the module:

##### PSDs
* BendingPL(v,A,v_bend,a_low,a_high,c) - Bending power law


#### General:
* MixtureDist(x,f,args,weights) - Mixture distribution of any set of functions

#### Scipy random variate distributions
A large number of these distributions are available and are genericised such
that each can be described with three arguments: shape, loc (location) and scale.
As a result, some of the parameters aren't what you might expect, and each
of these three parameters are needed when using a scipy RV function in this code.
These can be used in mixture distributions in the code, shown in examples below.
The scipy documentation is the best place to check this, but below are examples
of this in the form of the Gamma and lognormal distributions which can describe
AGN PDFs well:

* For a log normal distribution with a given mean and sigma (standard deviation):
```python
	scipy.stats.lognorm.pdf(x, shape=sigma,loc=0,scale=np.exp(mean))
```
* For a gamma distribution with a given kappa and theta:
```python
	scipy.stats.pdf(x, kappa,loc=0, scale=theta)
```

http://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html

### Plotting 
The following commands are attributes of the Lightcurve class:
* Plot_Lightcurve()       - Plot the lightcurve
* Plot_Periodogram()      - Plot the lightcurve's periodogram
* Plot_PDF()              - Plot the lightcurve's probability density function
* Plot_Stats()            - Plot the lightcurve, its periodogram and PDF

The following commands take Lightcurve objects as inputs:
* Comparison_Plots(lightcurves,bins=25,norm=True) - Plot multiple lightcurves and their PSDs & PDFs
                                                   
### Other attributes & methods of the Lightcurve class 

##### Attributes
* time            - The lightcurve's time array
* flux            - The lightcurve's flux array
* errors          - The lightcurve's flux error array
* length          - The lightcurve's length
* freq            - The lightcurve's periodogram's frequency array
* psd             - The lightcurve's power spectral density array (if calculated)
* mean            - The lightcurve's mean flux
* std             - The lightcurve's standard deviation
* std_est         - The lightcurve's estimated underlying SD (if calculated)
* tbin            - The lightcurve's time bin size
* fft             - The lightcurve's Fourier transform (if calculated)
* periodogram     - The lightcurve's periodogram (if calculated)

##### Methods (functions)
* STD_Estimate(PSDdist,PSDdistArgs) - Calculate the estimate of the underlying
                                    standard deviation (without Poisson noise),
                                    which is used in simulations if present
* Fourier_Transform()               - Calculate the lightcurve's Fourier transform
                                    (calculated automatically if required)
* Periodogram()                     - Calculate the lightcurve's periodogram
                                    (calculated automatically if required)


## Example usage:

```python
#------- Input parameters -------

from DELCgen import *
import scipy.stats as st

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

# simulate artificial light curve with Emmanoulopoulos method, scipy distribution
delc = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([st.gamma,st.lognorm],[[kappa,0, theta],\
                                    [lnsig,0, np.exp(lnmu)]],[weight,1-weight]))

#simulate artificial light curve with Emmanoulopoulos method, custom distribution
delc2 = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([[Gamma,LogNormal],[[kappa, theta],\
                                  [lnmu, lnsig]],[weight,1-weight]]),MixtureDist)                                

# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc,delc2])
```

# Emmanoulopoulos Lightcurve Simulation

#### Python version of the Emmanoulopoulos light curve simulation algorithm.
##### As according to Emmanoulopoulos et al 2013, Monthly Notices of the Royal Astronomical Society, 433, 907

##### A research note on the creation of this code has been published on ArXiv and is available here:

#####  [http://arxiv.org/abs/1503.06676](http://arxiv.org/abs/1503.06676)

#####  If you have questions, suggestions, problems etc. please email me at sdc1g08@soton.ac.uk

### Description:

The code uses a 'Lightcurve' class which contains all of the data necessary
for simulation of artificial version, and for plotting. Lightcurve objects
can be easily created from data using the following command:

```python
lc = Load_Lightcurve(fileroute)
```

They can also be created manually:
```python
lc = Lightcurve(time,flux,errors=None,tbin=100)
```
Where 'time' and 'flux' are data arrays of the lightcurve. 'Errors' is an optional array of
errors on the fluxes in the lightcurve, not used for simulation. 'tbin' is the sample rate
of the lightcurve, which *is* used in simulation.

**The input file must be a text file with three columns of time, flux and the error 
on the flux. The lightcurve must also be binned at regular intervals.** Headers, footers etc. are handled.

Artificial lightcurves can be produced with the same PSD and PDF as the data the following command:
```python
delc = datalc.Simulate_DE_Lightcurve()
```

Or, for a specific given PSD and PDF model using:

##### Timmer & Koenig (1995) method
From Timmer & Koenig, 1995, Astronomy & Astrophysics, 300, 707.
```python
tklc = Simulate_TK_Lightcurve(datalc,PSDfunction, PSDparams, RedNoiseL, aliasTbin, RandomSeed)
```

##### Emmanoulopoulos (2013) method
From Emmanoulopoulos et al., 2013, Monthly Notices of the Royal Astronomical Society, 433, 907.

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


#### General/PDF:
* MixtureDist(functions,n_args,frozen=None) - Mixture distribution CLASS for creating 
  an object which can calculate a value from or sample a mixture of any set of functions.
  The number of arguments of each function must be specified. Specific parameters can
  also be frozen at a given value for each function, such that the resultant function
  does not take this parameter as an argument.

  e.g. mix_model = Mixture_Dist([st.gamma,st.lognorm],[3,3],[[[2],[0]],[[2],[0],]])
  produces a mixture distribution consisting of a gamma distribution and a lognormal
  distribution, both of which have their 3rd parameter frozen at 0. The resultant
  function will therefore require 4 parameters (two for the gamma distribution and
  two for the lognormal distribution) **plus the weights of each**.
  
  The value of this function at a given value of 'x' for a given set of parameters can 
  then be obtained using mix_model.Value(x,params), where 'params' is a list of the
  parameters followed by the weights of each function in the mixture distribution, e.g.
  [f1_p1,f1_p2,f2_p1,f2_p1,w1,w2] in this case.
  
  The function can also be randomly sampled using mix_model.Sample(params,length=1),
  where 'params' is the function parameters given as described above and 'length'
  is the length of the resultant sample array, i.e. the number of samples drawn
  from the distribution.

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

### Fitting Models
The following commands are methods of the Lightcurve class:
* Fit_PSD() - Fit the lightcurve's periodogram with a given PSD model.
* Fit_PDF() - Fit the lightcurve with a given PDF model

### Simulating Light Curves
The following commands are methods of the Lightcurve class:
* delc = datalc.Simulate_DE_Lightcurve() - Simulate a lightcurve with 
the same PSD and PDF as the data lightcurve, using the Emmanoulopoulos method.


The following commands require a model and best-fit parameters as inputs:
* tklc = Simulate_TK_Lightcurve(datalc,PSDfunction, PSDparams, RedNoiseL, aliasTbin, RandomSeed)
- Simulate a lightcurve with a given PSD and PDF, using the Emmanoulopoulos method
* delc = Simulate_DE_Lightcurve(datalc,PSDfunction, PSDparams, PDFfunction, PDFparams)
- Simulate a lightcurve with a given PSD and PDF, using the Timmer and Koenig method.

### Plotting 
The following commands are methods of the Lightcurve class:
* Plot_Lightcurve()       - Plot the lightcurve
* Plot_Periodogram()      - Plot the lightcurve's periodogram
* Plot_PDF()              - Plot the lightcurve's probability density function
* Plot_Stats()            - Plot the lightcurve, its periodogram and PDF

The following commands take Lightcurve objects as inputs:
* Comparison_Plots(lightcurves,bins=25,norm=True) - Plot multiple lightcurves and their PSDs & PDFs

### Saving 
The following commands are methods of the Lightcurve class:
* Save_Lightcurve(filename)  - Save the lightcurve (time and flux) as a text file
* Save_Periodogram(filename) - Plot the periodogram (frequency and power) as a text file
                                                   
### Other Attributes of the Lightcurve class 
* time            - The lightcurve's time array
* flux            - The lightcurve's flux array
* errors          - The lightcurve's flux error array
* length          - The lightcurve's length
* freq            - The lightcurve's periodogram's frequency array
* periodogram     - The lightcurve's periodogram (if calculated)
* mean            - The lightcurve's mean flux
* std             - The lightcurve's standard deviation
* std_est         - The lightcurve's estimated underlying SD (if calculated)
* tbin            - The lightcurve's time bin size
* fft             - The lightcurve's Fourier transform (if calculated)
* psdModel        - The model used in fitting the lightcurve's periodogram
* psdFit          - The fit outcome from fitting the lightcurve's PDF, including the best fitting parameters 
* pdfModel        - The model used in fitting the lightcurve's probability density function 
* pdfFit          - The fit outcome from fitting the lightcurve's PSD, including the best fitting parameters

##### Other Methods (functions)
The following commands are methods of the Lightcurve class:
* STD_Estimate() 		    - Calculate the estimate of the underlying
                                    standard deviation (without Poisson noise),
                                    which is used in simulations if present
* Fourier_Transform()               - Calculate the lightcurve's Fourier transform
                                    (calculated automatically if required by another function)
* Periodogram()                     - Calculate the lightcurve's periodogram
                                    (calculated automatically if required by another function)

The following commands are global methods:

* RandAnyDist(f,args,a,b) - Generate random values from the distribution 'f' with parameters 'args', between 'a' and 'b'
* OptBins(data) - Calculate the optimum number of bins  to describe the PDF of a data set, using the method of Knuth et al. 2006.

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
```

![alt tag] (https://raw.githubusercontent.com/samconnolly/DELightcurveSimulation/master/LC.png)

```python
# estimate underlying variance od data light curve
datalc.STD_Estimate()


# simulate artificial light curve with Emmanoulopoulos method, using the PSD and PDF of the data
delc = datalc..Simulate_DE_Lightcurve() # defaults to bending PL and mix of gamma and lognormal dist.

# simulate artificial light curve with Timmer & Koenig method
tklc = Simulate_TK_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                RedNoiseL,aliasTbin,RandomSeed)

# simulate artificial light curve with Emmanoulopoulos method, scipy distribution
delc2 = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([st.gamma,st.lognorm],[[kappa,0, theta],\
                                    [lnsig,0, np.exp(lnmu)]],[weight,1-weight]))

# simulate artificial light curve with Emmanoulopoulos method, custom distribution
delc3 = Simulate_DE_Lightcurve(datalc,BendingPL, (A,v_bend,a_low,a_high,c),
                                ([[Gamma,LogNormal],[[kappa, theta],\
                                  [lnmu, lnsig]],[weight,1-weight]]),MixtureDist)                                

# plot lightcurves and their PSDs ands PDFs for comparison
Comparison_Plots([datalc,tklc,delc])
```

![alt tag] (https://raw.githubusercontent.com/samconnolly/DELightcurveSimulation/master/ComparisonPlots.png)

```python

# Save lightcurve and Periodogram as text files
delc.Save_Lightcurve('lightcurve.dat')
delc.Save_Periodogram('periodogram.dat')

```

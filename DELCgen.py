
"""
LCgenerationDE.py

Created on Fri Nov  7 16:18:50 2014

Author: Sam Connolly

Python version of the light curve simulation algorithm from Emmanoulopoulos et al., 
2013, Monthly Notices of the Royal Astronomical Society, 433, 907.
Uses 'Lightcurve' objects to allow easy interactive use and easy plotting. 
Can use any PSD or PDF model, requires best fits to be known.

requires:
    numpy, pylab, scipy, math

"""
import numpy as np
import pylab as plt
import scipy.integrate as itg
import numpy.fft as ft
import numpy.random as rnd

# ------ Distribution Functions --------------------------------------------------

def MixtureDist(x,f,args,weights):
    '''
    Mixture distribution applied to the values in array x, using a combination
    of all the functions in the array f
    
    inputs:
        x (array or float) - numerical array to which the mixture function will be applied
        f (array of functions) - list of all of the functions to mix
        args (2-D array) - numerical array of the parameters of each function
        weights (arra)  - numerical array of weights of each component
    outputs:
        data (array or float) - numerical array of the output values
    '''
    data = np.array(f[0](x,*args[0]))*weights[0]
    for i in range(1,len(f)):
        data += np.array(f[i](x,*args[1]))*weights[i]
    return data / np.sum(weights)

def BendingPL(v,A,v_bend,a_low,a_high,c):
    '''
    Bending power law function - returns power at each value of v, 
    where v is an array (e.g. of frequencies)
    
    inputs:
        v (array)       - input values
        A (float)       - normalisation 
        v_bend (float)  - bending frequency
        a_low ((float)  - low frequency index
        a_high float)   - high frequency index
        c (float)       - intercept/offset
    output:
        out (array)     - output powers
    '''
    numer = v**-a_low
    denom = 1 + (v/v_bend)**(a_high-a_low)
    out = A * (numer/denom) + c
    return out

def RandAnyDist(f,args,a,b,size=1):
    '''
    Generate random numbers from any distribution
    
    inputs:
        f (function f(x,**args)) - The distribution from which numbers are drawn
        args (tuple) - The arguments of f (excluding the x input array)
        a,b (float)  - The range of values for x
        size (int, optional) - The size of the resultant array of random values,
                                    returns a single value as default
    outputs:
        out (array) - List of random values drawn from the input distribution
    '''
    out = []
    while len(out) < size:    
        x = rnd.rand()*(b-a) + a  # random value in x range
        v = f(x,*args)            # equivalent probability
        p = rnd.rand()      # random number
        if p <= v:
            out.append(x)   # add to value sample if random number < probability
    if size == 1:
        return out[0]
    else:
        return out
        
def Cont_Mix_Sample(funcs,args,weights,length):
    '''
    Use inverse transform sampling to randomly sample a mixture of any set
    of scipy contiunous (random variate) distributions.
    
    inputs:
        funcs (array, scipy continuous functions) - Set of Functions to sample
        weights (array, floats)     - Weights of each function
        args (array, tuples)        - Arguments of each function
        length (int)                - Length of the random sample
    outputs:
        sample (array, float)       - The random sample
    '''
    mix    = np.random.random(size=length)  # random sample for function choice
    cumWeights = [sum(weights[:i]) for i in range(1,len(weights)+1)]
    sample = np.array([])
    
    # cycle through each distribution according to probability weights
    for i in range(len(funcs)):   
        if i > 0:
            sample = np.append(sample, funcs[i].rvs(args[i][0],loc=args[i][1],scale=args[i][2],size=len(np.where((mix>cumWeights[i-1])*(mix<=cumWeights[i]))[0]))) 
        else:
            sample = np.append(sample, funcs[i].rvs(args[i][0],loc=args[i][1],scale=args[i][2],size=len(np.where((mix<=cumWeights[i]))[0])))
            
    # randomly mix sample
    np.random.shuffle(sample)
    
    return sample 
               
#--------- Standard Deviation estimate -------------------------------------------
 
def SD_estimate(mean,v_low,v_high,PSDdist,PSDdistArgs):
    '''
    Estimate the standard deviation from a PSD model, by integrating between
    frequencies of interest, giving RMS squared variability, and multiplying
    by the mean squared. And square rooting.
    
    inputs:
        mean (float)        - mean of the data
        v_low (float)       - lower frequency bound  of integration
        v_high (float)      - upper frequency bound  of integration
        PSDdist (function)  - PSD distribution function
        PSDdistargs (var)   - PSD distribution best fit parameters
    outputs:
        out (float)     - the estimated standard deviation
    '''
    i = itg.quad(PSDdist,v_low,v_high,PSDdistArgs)
    out = [np.sqrt(mean**2.0 * i[0]),np.sqrt(mean**2.0 * i[1])]
    return out

#------------------ Simulation Functions -----------------------------------------

def TimmerKoenig(RedNoiseL,aliasTbin,randomSeed,tbin,LClength,std,mean,PSDmodel,PSDmodelArgs):    
    '''
    Generates an artificial lightcurve with the a given power spectral 
    density in frequency space, using the method from Timmer & Koenig, 1995,
    Astronomy & Astrophysics, 300, 707.

    inputs:
        RedNoiseL (int)        - multiple by which simulated LC is lengthened 
                                compared to the data LC to avoid red noise leakage
        aliasTbin (int)        - divisor to avoid aliasing
        lclength  (int)        - Length of simulated LC
        mean (float)           - mean amplitude of lightcurve to generate
        std (float)            - standard deviation of lightcurve to generate
        randomSeed (int)       - Random number seed
        PSDmodel (function)    - Function for model used to fit PSD
        PSDmodelArgs (various) - Arguments/parameters of best-fitting PSD model
   
    outputs:
        lightcurve (array)     - array of amplitude values (cnts/flux) with the same
                               timing properties as entered, length 1024 seconds,
                               sampled once per second.  
        fft (array)            - Fourier transform of the output lightcurve
        shortPeriodogram (array, 2 columns) - periodogram o the output lightcurve
                                                [freq, power]
        '''                    
    # --- create freq array up to the Nyquist freq & equivalent PSD --------------
    frequency = np.arange(1.0, (RedNoiseL*LClength)/2+1)/(RedNoiseL*LClength*tbin*aliasTbin)
    powerlaw = PSDmodel(frequency,*PSDmodelArgs)

    # -------- Add complex Gaussian noise to PL ---------------------------------
    rnd.seed(randomSeed)
    real = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,((RedNoiseL*LClength)/2))
    imag = (np.sqrt(powerlaw*0.5))*rnd.normal(0,1,((RedNoiseL*LClength)/2))
    positive = np.vectorize(complex)(real,imag) # array of positive, complex no.s                        
    noisypowerlaw = np.append(positive,positive.conjugate()[::-1])
    znoisypowerlaw = np.insert(noisypowerlaw,0,complex(0.0,0.0)) # add 0

    # --------- Fourier transform the noisy power law ---------------------------
    inversefourier = np.fft.ifft(znoisypowerlaw)  # should be ONLY  real numbers)
    longlightcurve = inversefourier.real  # take real part of the transform
 
    # extract random cut and normalise output lightcurve, produce its fft & periodogram
    if RedNoiseL == 1:
        lightcurve = longlightcurve
    else:
        extract = rnd.randint(LClength-1,RedNoiseL*LClength + 1)
        lightcurve = np.take(longlightcurve,range(extract,extract + LClength)) 
    lightcurve = (lightcurve-np.mean(lightcurve))/np.std(lightcurve)*std+mean
    fft = ft.fft(lightcurve)
    periodogram = ((2.0*tbin*aliasTbin)/(LClength*(np.mean(lightcurve)**2))) *np.absolute(fft)**2     
    shortPeriodogram = np.take(periodogram,range(1,LClength/2 +1))
    shortFreq = np.take(frequency,range(1,LClength/2 +1))*RedNoiseL
    shortPeriodogram = [shortFreq,shortPeriodogram]
    return lightcurve, fft, shortPeriodogram

# The Emmanoulopoulos Loop

def EmmanLC(time,flux,mean,std,RedNoiseL,aliasTbin,RandomSeed,tbin,PSDmodel, 
                PSDmodelArgs, PDFdistArgs, PDFdist="scipy",
                    maxIterations=1000,verbose=False):
    '''
    Produces a simulated lightcurve with the same power spectral density, mean,
    standard deviation and probability density function as those supplied.
    Uses the method from Emmanoulopoulos et al., 2013, Monthly Notices of the
    Royal Astronomical Society, 433, 907. Starts from a lightcurve using the
    Timmer & Koenig (1995, Astronomy & Astrophysics, 300, 707) method, then
    adjusts a random set of values ordered according to this lightcurve, 
    such that it has the correct PDF and PSD. Using a scipy.stats distribution
    recommended for speed.
    
    inputs:
        time (array)    - Times from data lightcurve
        flux (array)    - Fluxes from data lightcurve
        mean (float)    - The mean of the resultant lightcurve
        std (float)     - The standard deviation of the resultant lightcurve
        RedNoiseL (int) - multiple by which simulated LC is lengthened compared
                            to the data LC to avoid red noise leakage
        aliasTbin (int) - divisor to avoid aliasing
        RandomSeed (int)- seed used in random number generation, for repeatability
        tbin (int)      - lightcurve bin size
        PSDmodel (fn)   - Function for model used to fit PSD
        PSDmodelArgs (tuple,var) - arguments/parameters of best-fitting PSD model
        PDFdistArgs (tuple,var) - Distributions/params of best-fit PDF model(s)
                            If a scipy random variate is used, this must be in
                            the form: 
                                ([distributions],[[shape,loc,scale]],[weights])
        PDFdist (fn,optional) - Function for model used to fit PDF if not scipy
        maxIterations (int,optional) - The maximum number of iterations before
                                        the routine gives up (default = 1000)
        verbose (bool, optional) - If true, will give you some idea what it's
                                    doing, by telling you (default = False)
 
        surrogate (array, 2 column)     - simulated lightcurve [time,flux]
        PSDlast (array, 2 column)       - simulated lighturve PSD [freq,power]
        shortLC (array, 2 column)       - T&K lightcurve [time,flux]
        periodogram (array, 2 column)   - T&K lighturve PSD [freq,power]
        ffti (array)                    - Fourier transform of surrogate LC
    '''
    
    length = len(time)
    ampAdj = None
    
    # Produce Timmer & Koenig simulated LC
    if verbose:
        print "Running Timmer & Koening..."
    shortLC, fft, periodogram = \
            TimmerKoenig(RedNoiseL,aliasTbin,RandomSeed,tbin,len(flux),std,mean,PSDmodel,PSDmodelArgs)
    
    shortLC = [np.arange(len(shortLC))*tbin, shortLC]
    
    # Produce random distrubtion from PDF, up to max flux of data LC
    if PDFdist == "scipy":    # use inverse transform sampling if a scipy dist
        if verbose: 
            print "Inverse tranform sampling..."
        dist = Cont_Mix_Sample(PDFdistArgs[0],PDFdistArgs[1],PDFdistArgs[2],length)
    else: # else use rejection
        if verbose: 
            print "Rejection sampling... (slow!)"
        dist = RandAnyDist(PDFdist,PDFdistArgs,0,max(flux)*1.2,length)
        dist = np.array(dist)
        
    sortdist = dist[np.argsort(dist)] # sort!
    
    # Iterate over the random sample until its PSD (and PDF) match the data
    if verbose:
        print "Iterating..."
    i = 0
    oldSurrogate = np.array([-1])
    surrogate = np.array([1])
    
    while i < maxIterations and np.array_equal(surrogate,oldSurrogate) == False:
    
        oldSurrogate = surrogate
    
        if i == 0:
            surrogate =[time, dist] # start with random distribution from PDF
        else:
            surrogate = (ampAdj - np.mean(ampAdj)) / np.std(ampAdj) 
            surrogate = [time,(surrogate * std) + mean] # renormalised, adjusted LC
            
        ffti = ft.fft(surrogate[1])
        
        PSDlast = ((2.0*tbin)/(length*(mean**2))) *np.absolute(ffti)**2
        PSDlast = [periodogram[0],np.take(PSDlast,range(1,length/2 +1))]
        
        fftAdj = np.absolute(fft)*(np.cos(np.angle(ffti)) + 1j*np.sin(np.angle(ffti)))  #adjust fft
        LCadj = ft.ifft(fftAdj)
        LCadj = [time/tbin, ((LCadj - np.mean(LCadj))/np.std(LCadj)) * std + mean]
        
        PSDLCAdj = ((2.0*tbin)/(length*np.mean(LCadj)**2.0)) * np.absolute(ft.fft(LCadj))**2
        PSDLCAdj = [periodogram[0],np.take(PSDLCAdj, range(1,length/2 +1))]
        sortIndices = np.argsort(LCadj[1])
        sortPos = np.argsort(sortIndices)
        ampAdj = sortdist[sortPos]
        
        i += 1
        
    if verbose:
        print "Converged in {} iterations".format(i)
    
    return surrogate, PSDlast, shortLC, periodogram, ffti

#--------- Lightcurve Class & associated functions -------------------------------

class Lightcurve(object):
    '''
    Light curve class
    
    inputs:
        time (array)            - Array of times for lightcurve
        flux (array)            - Array of fluxes for lightcurve
        errors (array,optional) - Array of flux errors for lightcurve
        tbin (int,optional)     - width of time bin of lightcurve
    '''
    def __init__(self,time,flux,errors=None,tbin=100):
         self.time = time
         self.flux  = flux
         self.errors = errors
         self.length = len(time)
         self.freq = np.arange(1, self.length/2.0 + 1)/(self.length*tbin)
         self.psd = None 
         self.mean = np.mean(flux)
         self.std_est = None
         self.std = np.std(flux) 
         self.tbin = tbin
         self.fft = None
         self.periodogram = None

    def STD_Estimate(self,PSDdist,PSDdistArgs):
        '''
        set and return standard deviation estimate from a given PSD and freq range
        
        inputs:
            PDFdist (function)- Function for model used to fit PDF
            PDFdistArgs (var) - Arguments/parameters of best fitting PDF model
        '''
        self.std_est = SD_estimate(self.mean,self.freq[0],self.freq[-1],PSDdist,PSDdistArgs)[0]
        return self.std_est   
         
    def PSD(self,PSDdist,PSDdistArgs):            
        self.psd = PSDdist(self.freq,*PSDdistArgs)
        return self.psd

    def Fourier_Transform(self):
        '''
        Calculate and return the Fourier transform of the lightcurve
        '''
        self.fft = ft.fft(self.flux) # 1D Fourier Transform (as time-binned)
        return self.fft
    
    def Periodogram(self):
        '''
        Calculate, set and return the Periodogram of the lightcurve. Does the
        Fourier transform if not previously carried out.
        '''
        if self.fft == None:
            self.Fourier_Transform()
        periodogram = ((2.0*self.tbin)/(self.length*(self.mean**2)))\
                            * np.absolute(np.real(self.fft))**2
        freq = np.arange(1, self.length/2.0 + 1)/(self.length*self.tbin)
        shortFreq = np.take(freq,range(1,self.length/2 +1))
        shortPeriodogram = np.take(periodogram,range(1,self.length/2 +1))
        self.periodogram = [shortFreq,shortPeriodogram]
        return self.periodogram
        
    def Plot_Periodogram(self):
        '''
        Plot the periodogram of the lightcurvem after calculating it if necessary
        '''
        if self.periodogram == None:
            self.Create_Periodogram()
        p = plt.subplot(1,1,1)
        plt.scatter(self.periodogram[0],self.periodogram[1])
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')
        plt.xlim([0.9e-5,6e-3])
        plt.ylim([0.5e-4,1e4])
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)
        plt.show()

    def Plot_Lightcurve(self):
        '''
        Plot the lightcurve
        '''
        #plt.errorbar(self.time,self.flux,yerr=self.errors,)
        plt.scatter(self.time,self.flux)
        plt.title("Lightcurve")
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)
        plt.show()
        
    def Plot_PDF(self,bins=25,norm=True):
        '''
        Plot the probability density fucntion of the lightcurve
        '''
        plt.hist(self.flux,bins=bins,normed=norm)
        plt.title("PDF")
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3) 
        plt.show()
        
    def Plot_Stats(self,bins=25,norm=True):
        '''
        Plot the lightcurve together with its probability density fucntion and
        power spectral density
        '''
        if self.periodogram == None:
            self.Create_Periodogram()
        plt.subplot(3,1,1)
        plt.scatter(self.time,self.flux)
        plt.title("Lightcurve")
        plt.subplot(3,1,2)
        plt.hist(self.flux,bins=bins,normed=norm)
        plt.title("PDF")
        p=plt.subplot(3,1,3)
        plt.scatter(self.periodogram[0],self.periodogram[1])
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')
        plt.xlim([0.9e-5,6e-3])
        plt.ylim([0.5e-4,1e4])
        plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3) 
        plt.show()

def Comparison_Plots(lightcurves,bins=25,norm=True, names=None):
    '''
    plot multiple lightcurves, their PDFs and PSDs together, for comparison
    
    inputs:
        lightcurves (array Lightcurves)  - list of lightcurves to plot
        bins (int, optional)             - number of bins in PDF histograms
        norm (bool, optional)            - normalises PDF histograms if true
        names (array (string), optional) - list of ligtcurve names
    '''
    n = len(lightcurves)
    if names == None or len(names) != n:
        names = ["Lightcurve {}".format(l) for l in range(1,n+1)]
    i = 0
    for lc in lightcurves:
        if lc.periodogram == None:
            lc.Periodogram()
        plt.subplot(3,n,1+i)
        plt.scatter(lc.time,lc.flux)
        plt.title(names[i])
        plt.subplot(3,n,n+1+i)
        plt.hist(lc.flux,bins=bins,normed=norm)
        plt.title("PDF")
        p=plt.subplot(3,n,2*n+1+i)
        plt.scatter(lc.periodogram[0],lc.periodogram[1])
        plt.title("Periodogram")
        p.set_yscale('log')
        p.set_xscale('log')
        plt.xlim([0.9e-5,6e-3])
        plt.ylim([0.5e-4,1e4])
        i += 1 
    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,
                            hspace=0.3,wspace=0.3)      
    plt.show()    
        
def Load_Lightcurve(fileroute, header=0):
    '''
    Loads a data lightcurve as a 'Lightcurve' object, assuming a text file
    with three columns. Can deal with headers, but default is none.
    
    inputs:
        fileroute (string)      - fileroute to text file containg lightcurve data
        header (int, optional)  - number of lines to skip, i.e. header size
    outputs:
        lc (lightcurve)         - otput lightcurve object
    '''
    f = open(fileroute,'r')

    time,flux,error = [],[],[]
    
    h = 0
    for line in f:
        if h > header:
            t,f,e = line.split()
            time.append(float(t))
            flux.append(float(f))
            error.append(float(e))
        h += 1
       
    lc = Lightcurve(np.array(time), np.array(flux), np.array(error))
    
    return lc

def Simulate_TK_Lightcurve(lightcurve,PSDmodel,PSDmodelArgs,RedNoiseL=100,
                                                   aliasTbin=1,randomSeed=None):
    '''
    Creates a (simulated) lightcurve object from another (data) lightcurve object,
    using the Timmer & Koenig (1995) method. The estimated standard deviation is 
    used if it has been calculated.
    
    inputs:
        lightcurve (Lightcurve)   - Lightcurve object to be simulated from...
        PSDmodel (function)       - Function used to describe lightcurve's PSD
        PSDmodelArgs (various)    - Arguments/parameters of best fit PSD model
        RedNoiseL (int, optional) - Multiple by which to lengthen the lightcurve 
                                    to avoid red noise leakage
        aliasTbin (int, optional) - divisor to avoid aliasing
        randomSeed (int, optional)- seed for random value generation, to allow
                                        repeatability
    outputs:
        lc (Lightcurve)           - Lightcurve object containing simulated LC
    '''
    
    if lightcurve.std_est == None:
        std = lightcurve.std
    else:
        std = lightcurve.std

    shortLC, fft, periodogram = \
        TimmerKoenig(RedNoiseL,aliasTbin,randomSeed,lightcurve.tbin,
                 lightcurve.length,std,lightcurve.mean,
                     PSDmodel,PSDmodelArgs)
    lc = Lightcurve(lightcurve.time,shortLC,tbin=lightcurve.tbin)
    lc.fft = fft
    lc.periodogram = periodogram
    return lc

def Simulate_DE_Lightcurve(lightcurve,PSDmodel,PSDmodelArgs, PDFdistArgs,PDFdist="scipy",
                               RedNoiseL=100, aliasTbin=1,randomSeed=None,
                                   maxIterations=1000,verbose=False):
    '''
    Creates a (simulated) lightcurve object from another (data) lightcurve object,
    using the Emmanoulopoulos (2013) method. The estimated standard deviation is 
    used if it has been calculated.
    
    inputs:
        lightcurve (Lightcurve)   - Lightcurve object to be simulated from...
        PSDmodel (function)       - Function used to describe lightcurve's PSD
        PSDmodelArgs (various)    - Arguments/parameters of best fit PSD model
        PDFdistArgs (tuple,var) - Distributions/params of best-fit PDF model(s)
                            If a scipy random variate is used, this must be in
                            the form: 
                                ([distributions],[[shape,loc,scale]],[weights])
        PDFdist (fn,optional) - Function for model used to fit PDF if not scipy        
        RedNoiseL (int, optional) - Multiple by which to lengthen the lightcurve 
                                    to avoid red noise leakage
        aliasTbin (int, optional) - divisor to avoid aliasing
        randomSeed (int, optional)- seed for random value generation, to allow
                                        repeatability
        maxIterations (int,optional) - The maximum number of iterations before
                                        the routine gives up (default = 1000)
        verbose (bool, optional) - If true, will give you some idea what it's
                                    doing, by telling you (default = False)
        
    outputs:
        lc (Lightcurve)           - Lightcurve object containing simulated LC
    '''

    if lightcurve.std_est == None:
        std = lightcurve.std
    else:
        std = lightcurve.std
    
    surrogate, PSDlast, shortLC, periodogram, fft = \
        EmmanLC(lightcurve.time,lightcurve.flux,lightcurve.mean,std,
                    RedNoiseL,aliasTbin,randomSeed,lightcurve.tbin,
                        PSDmodel, PSDmodelArgs, PDFdistArgs, PDFdist,
                            maxIterations,verbose)
    lc = Lightcurve(surrogate[0],surrogate[1],tbin=lightcurve.tbin)
    lc.fft = fft
    lc.periodogram = PSDlast
    return lc

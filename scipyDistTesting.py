

import scipy.stats as st
import numpy as np
import pylab as plt
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

ln = st.lognorm
g  = st.gamma
kappa,theta,lnmu,lnsig,weight = 5.67, 5.96, 2.14, 0.31,0.82

x = np.arange(0,100,0.1)#np.linspace(ln.ppf(0.01, lnsig),ln.ppf(0.999, lnsig), 1000)
old = MixtureDist(x,[Gamma,LogNormal],[[kappa, theta],\
                                        [lnmu, lnsig]],[weight,1-weight])
new = ln.pdf(x, lnsig,loc=0,scale=np.exp(lnmu))*(1-weight)+\
        g.pdf(x, kappa,loc=0, scale=theta)*weight 
        
plt.plot(x, old,'r-', lw=5, alpha=0.6)
plt.plot(x, new,'b-', lw=5, alpha=0.6)

plt.show()


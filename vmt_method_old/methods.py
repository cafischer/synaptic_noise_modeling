from numpy import *
from header import *
from simplex import *

n_ival = 100

d={}                    # will contain the ISIs
for i in range(n_ival):
    d['%d'%(i+1)]=zeros(10, float)     


def gauss_f(x, m, s):
    """returns a Gaussian of mean 'm' and standard deviation 's' at point 'x'"""
    return exp(-(x-m)**2/2./s**2)/sqrt(2.*pi*s**2)


def smooth_g(a, s):
    """smoothes the array 'a' with a Gaussian filter, whos standard deviation is 's' time steps"""
    sa = size(a)-6*s        # cut gaussian at +- 3*sigma
    r = zeros(sa, float)
    g = array(arange(-3.*s,3.*s,1.))
    g = gauss_f(g, 0., s)
    norm=sum(g)
    for i in range(0, sa):
        r[i] = dot(a[i:6*s+i],g)/norm
             
    return r
        

def ivals(filename, d):
    """scans the voltage file and stores ISIs in the dictionary 'd'. """
    global n_ival
    file=open(filename,'r')
    nlines = 0
    for line in file:
        nlines += 1
    vm=zeros(nlines, float)

    i=0                 # relative index
    n=0                 # absolute index
    ni=0                # nb of intervals
    ns=0                # nb of spikes
    vm_old=-100.
    file.seek(0)
    for line in file:
        vm[i]=float(line)
        vma=vm[i]
        if((vm[i]>vt)&(vm_old<=vt)):    # spike?
            ns+=1
            if((i>=n_minISI+ahp+pre)&(ns>1)):   
            # interval long enough? Skip ival before 1st spike
                ni+=1
                d['%d'%ni]=resize(d['%d'%ni],(i-pre-ahp,))
                d['%d'%ni][0:i-pre-ahp]=vm[ahp:i-pre]      
                print n, ni, size(d['%d'%ni])
                
            vm[:]=0.            # set array to 0.
            i=-1                # restart counting
            
        if((i>=n_maxISI+ahp+pre)&(ns>=1)): # max length exceeded?
            ni+=1
            d['%d'%ni]=resize(d['%d'%ni],(n_maxISI,))
            print n, ni, size(d['%d'%ni])
            d['%d'%ni][0:i-pre-ahp]=vm[ahp:i-pre]      
            vm[:]=0.            # set array to 0.
            i=-1                # restart counting
            
            
        vm_old=vma
                
   
            
        i+=1
        n+=1
        if(ni>=n_ival): break
    
    file.close()
    
    print "\n", ni, " ISIs found.\n"
    l=array([[float(k),size(d[k])] for k in d.keys()])
    l=compress(l[:,0]<=ni,l[:,1])
    print 'minimal length: ', int(l[argmin(l)]), " data points"
    print 'maximal length: ', int(l[argmax(l)]), " data points\n"
    
    return ni
    

def minimise(f, pstart):
    """makes use of a simplex algorithm (cf. 'Numerical Recipes') to find the minimum of the scalar function 'f'
    starting from point 'pstart'"""
    nf = 0
    nd = len(pstart)
    y = zeros(nd+1, float)
    p = zeros((nd+1, nd,), float)
    p[:] = pstart

    for k in range(nd+1):
        if k > 0:
            p[k,k-1] += p[k,k-1]/2.
        y[k] = f(p[k])

    bp = amoeba(p, y, 1e-10, f, nf)
    return bp





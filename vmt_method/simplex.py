from numpy import *

TINY = 1.e-10
N_EVAL_MAX = 700


def swap(a, b):
    tmp = a
    a = b
    b = tmp


def amotry(p, y, psum, func, ihi, fac):     # check index range
    ndim = size(y)-1
    ptry = zeros(ndim, float)
    fac1 = (1.-fac)/ndim
    fac2 = fac1-fac
    for j in range(ndim):                   # use array methods
        ptry[j] = psum[j]*fac1-p[ihi][j]*fac2
        
    ytry = func(ptry)
    if ytry < y[ihi]:
        y[ihi] = ytry
        for j in range(ndim):               # use array methods
            psum[j] += ptry[j]-p[ihi][j]
            p[ihi][j] = ptry[j]

    return ytry
    

def amoeba(p, y, ftol, func, n_eval=0):        # check index ranges
    ndim = size(y)-1
    mpts = ndim + 1
    psum = sum(p, 0)
    const = 1
    while const < 2:
        ilo = 0
        if y[1] > y[2]:
            [ihi, inhi] = [1,2]
        else:
            [ihi, inhi] = [2,1]
        
        for i in range(mpts):
            if (y[i] <= y[ilo]): ilo=i
            
            if (y[i] > y[ihi]): [inhi,ihi] = [ihi,i]
            elif ((y[i]>y[inhi]) & (i != ihi)): inhi = i

        rtol = 2.*abs(y[ihi]-y[ilo])/(abs(y[ihi])+abs(y[ilo])+TINY)
        if rtol < ftol:
            swap(y[0],y[ilo])
            for i in range(ndim): swap(p[0][i], p[ilo][i])
            
            print n_eval, "function calls\n"
            break
        
        if n_eval >= N_EVAL_MAX:
            print "N_EVAL_MAX exceeded\n"
            break
        
        n_eval += 2
        ytry = amotry(p, y, psum, func, ihi, -1.0)
        if ytry <= y[ilo]:
            ytry = amotry(p,y,psum,func,ihi,2.0)
        elif (ytry >= y[inhi]):
            ysave = y[ihi]
            ytry = amotry(p,y,psum,func,ihi,0.5)
            if ytry >= ysave:
                for i in range(mpts):
                    if i != ilo:
                        for j in range(ndim):
                            p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j])
                            
                        y[i] = func(psum)

                n_eval += ndim
                psum = sum(p, 0)
            
        else:
            n_eval -= 1

    return p[ihi]
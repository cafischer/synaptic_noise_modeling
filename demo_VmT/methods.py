from numpy import *
from header import *
from simplex import *


def gauss_fun(x, m, s):
    """returns a Gaussian of mean 'm' and standard deviation 's' at point 'x'"""
    return exp(-(x-m)**2/2./s**2) / sqrt(2.*pi*s**2)


def smooth_with_gauss(a, s):
    """smoothes the array 'a' with a Gaussian filter, whos standard deviation is 's' time steps"""
    sa = size(a)-6*s        # cut gaussian at +- 3*sigma
    r = zeros(sa, float)
    gauss = array(arange(-3.*s, 3.*s, 1.))
    gauss = gauss_fun(gauss, 0., s)
    norm = sum(gauss)
    for i in range(0, sa):
        r[i] = dot(a[i:6*s+i], gauss) / norm
    return r
        

def get_vm_between_spikes(file_name, dt, vt, t_pre=5, t_post=10, max_chunks=100, min_chunk_len=1000, max_chunk_len=10000):
    """scans the voltage file and stores ISIs in the dictionary 'd'.
    :param t_pre: excluded time preceding spike
    :param t_post: excluded time after spike
    """

    pre_idx = int(round(t_pre / dt))
    post_idx = int(round(t_post / dt))

    vm = loadtxt(file_name)
    vm_chunks = {}

    spike_idxs = []
    for i in range(1, size(vm)):
        if (vm[i] > vt) and (vm[i-1] <= vt):
            spike_idxs.append(i)

    idxs_chunk = {}
    n_chunk = 0
    for spike_i, spike_j in zip(spike_idxs[:len(spike_idxs)], spike_idxs[1:]):
        if (spike_j - pre_idx) - (spike_i + post_idx) >= min_chunk_len:
            vm_chunks['%d' % n_chunk] = vm[spike_i + post_idx:spike_j - pre_idx]
            idxs_chunk['%d' % n_chunk] = (spike_i + post_idx, spike_j - pre_idx)
            n_chunk += 1
            if len(vm_chunks) >= max_chunks:
                break
    n_chunks = len(vm_chunks)
    len_chunks = array([len(vm_chunks[k]) for k in vm_chunks.keys()])

    print "\n", len(vm_chunks), " intervals between spikes found."
    print 'minimal length: ', min(len_chunks)
    print 'maximal length: ', max(len_chunks)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(arange(len(vm))*dt, vm, 'k')
    # for k in vm_chunks.keys():
    #     plt.plot(arange(idxs_chunk[k][0], idxs_chunk[k][1])*dt, vm_chunks[k])
    # plt.show()
    return vm_chunks, n_chunks
    

def minimise(f, pstart):
    """makes use of a simplex algorithm (cf. 'Numerical Recipes') to find the minimum of the scalar function 'f'
    starting from point 'pstart'"""
    nf = 0
    nd = 3                  # nb. of parameters
    y = zeros(nd+1, float)
    p = zeros((nd+1, nd), float)
    p[:] = pstart

    for k in range(1, nd+1):
        p[k, k-1] += p[k, k-1]/2.

    for k in range(nd + 1):
        y[k] = f(p[k])

    bp = amoeba(p, y, 1e-10, f, nf)
    return bp
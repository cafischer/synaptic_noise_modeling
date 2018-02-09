from numpy import *
from header import *
from simplex import *
import scipy.optimize
import numdifftools as nd
from random import Random


def create_pseudo_random_number_generator(seed):
    pseudo_random_number_generator = Random()
    pseudo_random_number_generator.seed(seed)
    return pseudo_random_number_generator


def generate_p_start(seed, lower_bounds, upper_bounds):
    generator = create_pseudo_random_number_generator(seed)
    p_start = array([generator.uniform(l_b, u_b) for l_b, u_b in zip(lower_bounds, upper_bounds)])
    return p_start


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
        

def get_vm_between_spikes(file_name, dt, vt, t_pre=5, t_post=10, max_chunks=100,
                          min_chunk_len=1000, max_chunk_len=10000):
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

    if len(spike_idxs) == 0:  # TODO
        for i_chunk in range(min(max_chunks, len(vm) / max_chunk_len)):
            vm_chunks['%d' % i_chunk] = vm[i_chunk * max_chunk_len:(i_chunk + 1) * max_chunk_len]
    else:
        idxs_chunk = {}
        i_chunk = 0
        for spike_i, spike_j in zip(spike_idxs[:len(spike_idxs)], spike_idxs[1:]):
            if min_chunk_len <= (spike_j - pre_idx) - (spike_i + post_idx) <= max_chunk_len:
                vm_chunks['%d' % i_chunk] = vm[spike_i + post_idx:spike_j - pre_idx]
                idxs_chunk['%d' % i_chunk] = (spike_i + post_idx, spike_j - pre_idx)
                i_chunk += 1
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
    

def minimize(fun, p_start, lower_bounds, upper_bounds, fatol=1e-10, maxfev=700):
    """makes use of a simplex algorithm (cf. 'Numerical Recipes') to find the minimum of the scalar function 'f'
    starting from point 'pstart'"""

    # old method:
    # nf = 0
    # nd = 3                  # nb. of parameters
    # y = zeros(nd+1, float)
    # p = zeros((nd+1, nd), float)
    # p[:] = pstart
    #
    # for k in range(1, nd+1):
    #     p[k, k-1] += p[k, k-1]/2.
    #
    # for k in range(nd + 1):
    #     y[k] = fun(p[k])
    #
    # bp = amoeba(p, y, ftol, fun, nf)

    result = scipy.optimize.minimize(fun, p_start, method='Nelder-Mead',
                                     options={'fatol': fatol, 'maxfev': maxfev})

    # # methods with jacobian
    # def jac(p):
    #     jac_value = squeeze(nd.Jacobian(fun, step=1e-12, method='central')(p))
    #     jac_value[isnan(jac_value)] = 0
    #     return jac_value
    #
    # assert all(lower_bounds < upper_bounds)
    # bounds = [(l_b, u_b) for l_b, u_b in zip(lower_bounds, upper_bounds)]
    # result = scipy.optimize.minimize(fun, p_start, method='L-BFGS-B', jac=jac, bounds=bounds,
    #                                  options={'ftol': fatol, 'maxfun': maxfev})

    print 'success: ', result.success
    print 'nfev: ', result.nfev
    return result.x, result.success
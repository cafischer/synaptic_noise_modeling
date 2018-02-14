from numpy import *
from pylab import *
import json
from methods import *
from header import get_params


def norm(ge, gi, se, si):
    """computes the norm"""

    # TODO: to prevent float division by zero error
    if se == 0:
        se += sys.float_info.epsilon
    if si == 0:
        si += sys.float_info.epsilon

    he1 = 1. - dt / te
    he2 = dt / te
    hi1 = 1. - dt / ti
    hi2 = dt / ti
    hlin_e = -ge*(he1-1.)*he2*te/(2.*dt*se**2)
    hcross_e = he1*te/(2.*dt*se**2)
    hsq_e = -(1. + he1**2)*te/(4.*dt*se**2)
    hrest_e = -ge**2*he2**2*te/(4.*dt*se**2)
    hlin_i = -gi*(hi1-1.)*hi2*ti/(2.*dt*si**2)
    hcross_i = hi1*ti/(2.*dt*si**2)
    hsq_i = -(1. + hi1**2)*ti/(4.*dt*si**2)
    hrest_i = -gi**2*hi2**2*ti/(4.*dt*si**2)

    hlin0_e = ge*(2.*dt - he1*he2*te)/(2.*dt*se**2)
    hsq0_e = -(2.*dt + he1**2*te)/(4.*dt*se**2)
    hrest0_e = -ge**2/(2.*se**2)
    hlinf_e = ge*he2*te/(2.*dt*se**2)
    hsqf_e = -te/(4.*dt*se**2)
    hplin_e = hlin0_e
    hpsq_e = hsq0_e
    hlin0_i = gi*(2.*dt - hi1*hi2*ti)/(2.*dt*si**2)
    hsq0_i = -(2.*dt + hi1**2*ti)/(4.*dt*si**2)
    hrest0_i = -gi**2/(2.*si**2)
    hlinf_i = gi*hi2*ti/(2.*dt*si**2)
    hsqf_i = -ti/(4.*dt*si**2)
    hplin_i = hlin0_i
    hpsq_i = hsq0_i

    # logarithm of norm
    nrml = log(-pi/hsq0_e)/2. - hlin0_e**2/4./hsq0_e + hrest0_e
    nrml += log(-pi/hsq0_i)/2. - hlin0_i**2/4./hsq0_i + hrest0_i

    for k in range(1,n-1):
        hplin_e = hlin_e - hplin_e*hcross_e/2./hpsq_e
        hpsq_e = hsq_e - hcross_e**2/4./hpsq_e
        hplin_i = hlin_i - hplin_i*hcross_i/2./hpsq_i
        hpsq_i = hsq_i - hcross_i**2/4./hpsq_i

        nrml += log(-pi/hpsq_e)/2. - hplin_e**2/4./hpsq_e + hrest_e
        nrml += log(-pi/hpsq_i)/2. - hplin_i**2/4./hpsq_i + hrest_i

    hplin_e = hlinf_e-hplin_e*hcross_e/2./hpsq_e
    hpsq_e = hsqf_e-hcross_e**2/4./hpsq_e
    hplin_i = hlinf_i-hplin_i*hcross_i/2./hpsq_i
    hpsq_i = hsqf_i-hcross_i**2/4./hpsq_i

    nrml += log(-pi/hpsq_e)/2. - hplin_e**2/4./hpsq_e + hrest_e
    nrml += log(-pi/hpsq_i)/2. - hplin_i**2/4./hpsq_i + hrest_i

    return nrml


def integrated(ge, gi, se, si):
    """computes the integrated probability"""
    a0 = a[1:size(a)-1]
    ap = a[2:]
    b0 = b[1:size(b)-1]
    bp = b[2:]
    bm = b[:size(b)-2]

    # TODO: to prevent float division by zero error
    if se == 0:
        se += sys.float_info.epsilon
    if si == 0:
        si += sys.float_info.epsilon

    # arrays shifted: h[''][k] refers to g_e^{k+1}
    he1 = 1. - dt / te
    he2 = dt / te
    hi1 = 1. - dt / ti
    hi2 = dt / ti

    h['lin'][1:n-2] = (-ge*(he1-1.)*he2*te/se**2 + a0*(-b0*(1.+hi1**2)+gi*hi2+hi1*(bm+bp-gi*hi2))*ti/si**2)/2./dt
    h['cross'][1:n-2] = (he1*te/se**2+a0*ap*hi1*ti/si**2)/2./dt
    h['sq'][1:n-2] = -((1.+he1**2)*te/se**2 + a0**2*(1.+hi1**2)*ti/si**2)/4./dt
    h['rest'][1:n-2] = -te*(ge*he2/se)**2/4./dt - (b0-bm*hi1-gi*hi2)**2*ti/(4.*dt*si**2)

    h['lin'][0] = -(ge*(he1-1.)*he2*te/se**2 + a[0]*(b[0]-b[1]*hi1+b[0]*hi1**2+gi*(hi1-1.)*hi2)*ti/si**2)/2./dt
    h['cross'][0] = (he1*te/se**2+a[0]*a[1]*hi1*ti/si**2)/2./dt
    h['sq'][0] = -((1.+he1**2)*te/se**2 + a[0]**2*(1.+hi1**2)*ti/si**2)/4./dt
    h['rest'][0] = -te*(ge*he2/se)**2/4./dt - (b[0]-gi*hi2)**2*ti/4./dt/si**2
    h['lin'][n-2] = (ge*he2*te/se**2 + a[n-2]*(b[n-3]*hi1-b[n-2]+gi*hi2)*ti/si**2)/2./dt
    h['sq'][n-2] = -(te/se**2 + a[n-2]**2*ti/si**2)/4./dt
    h['rest'][n-2] = -te*(ge*he2/se)**2/4./dt - (b[n-2]-b[n-3]*hi1-gi*hi2)**2*ti/(4.*dt*si**2)

    hlin0_e = ge*(2.*dt-he1*he2*te)/(2.*dt*se**2)
    hlin0_i = (2.*dt*gi+hi1*(b[0]-gi*hi2)*ti)/(2.*dt*si**2)
    hcross0_e = he1*te/(2.*dt*se**2)
    hcross0_i = a[0]*hi1*ti/(2.*dt*si**2)
    hsq0_e = -(2.*dt+he1**2*te)/(4.*dt*se**2)
    hsq0_i = -(2.*dt + hi1**2*ti)/(4.*dt*si**2)
    hrest0 = -(ge/se)**2/2. - (gi/si)**2/2.
    hp['lin'][0] = h['lin'][0] - hcross0_e*hlin0_e/2./hsq0_e - hcross0_i*hlin0_i/2./hsq0_i
    hp['sq'][0] = h['sq'][0] - hcross0_e**2/4./hsq0_e - hcross0_i**2/4./hsq0_i

    for k in range(1, n-1):
        hp['lin'][k] = h['lin'][k] - hp['lin'][k-1]*h['cross'][k-1]/2./hp['sq'][k-1]
        hp['sq'][k] = h['sq'][k] - h['cross'][k-1]**2/4./hp['sq'][k-1]

    # prl: logarithm of probability
    prlv = -hp['lin']**2/4./hp['sq'] + log(-pi/hp['sq'])/2. + h['rest']
    prl = -(hlin0_e**2/hsq0_e+hlin0_i**2/hsq0_i)/4. + log(pi**2/hsq0_e/hsq0_i)/2. + hrest0
    prl += sum(prlv)

    return prl


def prob(params):
    """returns the logarithm of the inverse probability (to be minimised)"""
    [ge, se, si] = [params[0], params[1], params[2]]
    gi = gtot - gl - ge
    return -(integrated(ge, gi, se, si) - norm(ge, gi, se, si))


if __name__ == '__main__':
    vm_dir = '../simulate_model/data/model_pas/v.txt'  # input file
    res_dir = 'p_opt.txt'  # output file
    params_dir = '../simulate_model/data/model_pas/model_params.json'
    n_smooth = 3
    n_ival = 100

    random_generator = create_pseudo_random_number_generator(seed)

    # read params
    with open(params_dir, 'r') as f:
        params = json.load(f)
        param_names = ['i_ext', 'gtot', 'c_m', 'gl', 'El', 'Ee', 'Ei', 'te', 'ti', 'spike_threshold', 'dt',
                       'p_lower_bounds', 'p_upper_bounds']
        (i_ext, gtot, c_m, gl, El, Ee, Ei, te, ti, spike_threshold, dt,
         p_lower_bounds, p_upper_bounds) = (params[k] for k in param_names)


    h = {}
    h_attr = ['lin', 'cross', 'sq', 'rest']
    hp = {}
    hp_attr = ['lin', 'sq']

    sf = open(res_dir, 'w')         # open file and save result
    sf.write('#\t\tcurrent interval\t\t\ttemporary average\n')
    sf.write('# success\t  ge\t  gi\t  se\t  si\t  ge\t  gi\t  se\t  si\n\n')
    sf.close()

    v_chunks, n_chunks = get_vm_between_spikes(vm_dir, dt, spike_threshold)
    res = zeros((n_chunks, 9,), float)

    for i_chunk in range(n_chunks):
        vm = smooth_with_gauss(v_chunks['%d' % i_chunk], n_smooth)
        n = size(vm)

        for attr in h_attr:
            h[attr] = zeros(n-1, float)
        for attr in hp_attr:
            hp[attr] = zeros(n-1, float)

        vm0 = vm[:size(vm)-1]
        vmp = vm[1:]

        a = -(vm0 - Ee) / (vm0 - Ei)
        b = -(gl * (vm0 - El) + c_m * (vmp - vm0) / dt - i_ext) / (vm0 - Ei)

        print "\nprocessing interval", i_chunk + 1
        p_start = generate_p_start(i_chunk, p_lower_bounds, p_upper_bounds)
        p_opt, success = minimize(prob, p_start, p_lower_bounds, p_upper_bounds)    # maximize probability

        res[i_chunk, 0] = float(success)        # minimization successful?
        res[i_chunk, 1] = p_opt[0]              # exc. mean
        res[i_chunk, 2] = gtot - gl - p_opt[0]  # inh. mean
        res[i_chunk, 3:5] = p_opt[1:3]          # exc. and inh. SDs

        where_success = res[:, 0] == 1.0
        res[i_chunk, 5:9] = mean(res[where_success, 1:5], 0)  # momentary average over all successful minimized

        sf = open(res_dir, 'a')         # open file and save result
        line = '%d\t\t\t' % (res[i_chunk, 0])
        for i in range(1, 9):
            line += '%2.5f\t' % (res[i_chunk, i])

        sf.write(line+'\n')
        sf.close()

    # TODO: could also try minimization that respects bounds (all parameters should be greater than 0)
    # TODO: could try to update p_start, either use previous or sample random
    # TODO: check equations
from header import *
from numpy import *
from pylab import *
from methods import *
from vmt_method.methods import get_vm_between_spikes
import matplotlib.pyplot as pl
from cell_characteristics import to_idx
from nrn_wrapper import Cell
import os
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input


save_dir = os.path.join(save_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

n_ival = 50                # nb of intervals analysed

h = {}
h['lin'] = zeros(1, float)
h['cross'] = zeros(1, float)
h['sq'] = zeros(1, float)
h['rest'] = zeros(1, float)

hp = {}
hp['lin'] = zeros(1, float)
hp['sq'] = zeros(1, float)


a = zeros(1, float)
b = zeros(1, float)


def norm(ge, gi, se, si):
    """computes the norm"""
    hlin_e = -ge*(he1-1.)*he2*te/(2.*dt*se**2)
    hcross_e = he1*te/(2.*dt*se**2)
    hsq_e = -(1. + he1**2)*te/(4.*dt*se**2)
    hrest_e = -ge**2*he2**2*te/(4.*dt*se**2)
    hlin_i = -gi*(hi1-1.)*hi2*ti/(2.*dt*si**2)
    hcross_i = hi1*ti/(2.*dt*si**2)
    hsq_i = -(1. + hi1**2)*ti/(4.*dt*si**2)
    hrest_i = -gi**2*hi2**2*ti/(4.*dt*si**2)
#
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
#
#   logarithm of norm
    nrml = log(-pi/hsq0_e)/2. - hlin0_e**2/4./hsq0_e + hrest0_e
    nrml += log(-pi/hsq0_i)/2. - hlin0_i**2/4./hsq0_i + hrest0_i
#
    for k in range(1,n-1):
        hplin_e = hlin_e - hplin_e*hcross_e/2./hpsq_e
        hpsq_e = hsq_e - hcross_e**2/4./hpsq_e
        hplin_i = hlin_i - hplin_i*hcross_i/2./hpsq_i
        hpsq_i = hsq_i - hcross_i**2/4./hpsq_i
#
        nrml += log(-pi/hpsq_e)/2. - hplin_e**2/4./hpsq_e + hrest_e
        nrml += log(-pi/hpsq_i)/2. - hplin_i**2/4./hpsq_i + hrest_i

#
    hplin_e = hlinf_e-hplin_e*hcross_e/2./hpsq_e
    hpsq_e = hsqf_e-hcross_e**2/4./hpsq_e
    hplin_i = hlinf_i-hplin_i*hcross_i/2./hpsq_i
    hpsq_i = hsqf_i-hcross_i**2/4./hpsq_i
#
    nrml += log(-pi/hpsq_e)/2. - hplin_e**2/4./hpsq_e + hrest_e
    nrml += log(-pi/hpsq_i)/2. - hplin_i**2/4./hpsq_i + hrest_i
#
#
    return nrml


def integrated(ge, gi, se, si):
    """computes the integrated probability"""
    a0 = a[1:size(a)-1]
    ap = a[2:]
    b0 = b[1:size(b)-1]
    bp = b[2:]
    bm = b[:size(b)-2]
#
    # arrays shifted: h[''][k] refers to g_e^{k+1}
    h['lin'][1:n-2] = (-ge*(he1-1.)*he2*te/se**2 + a0*(-b0*(1.+hi1**2)+gi*hi2+hi1*(bm+bp-gi*hi2))*ti/si**2)/2./dt
    h['cross'][1:n-2] = (he1*te/se**2+a0*ap*hi1*ti/si**2)/2./dt
    h['sq'][1:n-2] = -((1.+he1**2)*te/se**2 + a0**2*(1.+hi1**2)*ti/si**2)/4./dt
    h['rest'][1:n-2] = -te*(ge*he2/se)**2/4./dt - (b0-bm*hi1-gi*hi2)**2*ti/(4.*dt*si**2)
#
#
    h['lin'][0] = -(ge*(he1-1.)*he2*te/se**2 + a[0]*(b[0]-b[1]*hi1+b[0]*hi1**2+gi*(hi1-1.)*hi2)*ti/si**2)/2./dt
    h['cross'][0] = (he1*te/se**2+a[0]*a[1]*hi1*ti/si**2)/2./dt
    h['sq'][0] = -((1.+he1**2)*te/se**2 + a[0]**2*(1.+hi1**2)*ti/si**2)/4./dt
    h['rest'][0] = -te*(ge*he2/se)**2/4./dt - (b[0]-gi*hi2)**2*ti/4./dt/si**2
    h['lin'][n-2] = (ge*he2*te/se**2 + a[n-2]*(b[n-3]*hi1-b[n-2]+gi*hi2)*ti/si**2)/2./dt
    h['sq'][n-2] = -(te/se**2 + a[n-2]**2*ti/si**2)/4./dt
    h['rest'][n-2] = -te*(ge*he2/se)**2/4./dt - (b[n-2]-b[n-3]*hi1-gi*hi2)**2*ti/(4.*dt*si**2)
#
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


def prob(para):
    """returns the logarithm of the inverse probability (to be minimised)"""
    [ge, se, si] = [para[0],para[1],para[2]]
    gi = gtot-gl-ge
    return -(integrated(ge, gi, se, si) - norm(ge, gi, se, si))



sf = open(resfile, 'w')         # open file and save result
sf.write('#\t\tcurrent ival\t\t\ttemporary average\n')
sf.write('# npts    ge\t  gi\t  se\t  si\t  ge\t  gi\t  se\t  si\n\n')
sf.close()

# TODO n_ival = ivals(vmFile,d)          # get intervals from file
d, n_ival = get_vm_between_spikes(vmFile, dt, vt, t_pre, t_post, n_ival, n_minISI, n_maxISI)

res = zeros((n_ival,9,), float)

for iInt in range(n_ival):
    vm = smooth_g(d['%d'%(iInt)], n_smooth)     # smooth voltage trace
    n = size(vm)
    
    for name in h.keys():
        h[name] = resize(h[name],(n-1,))

    for name in hp.keys():
        hp[name] = resize(hp[name],(n-1,))

    vm0 = vm[:size(vm)-1]
    vmp = vm[1:]
    a = -(vm0-Ve)/(vm0-Vi)
    b = -(gl*(vm0-Vl)+C*(vmp-vm0)/dt-Iext)/(vm0-Vi)
    print "processing ISI", iInt+1, "with", n, "datapoints"
    bp = minimise(prob, g_start)    # maximise probability

    res[iInt, 0] = n                 # nb of datapoints
    res[iInt, 1] = bp[0]             # exc. mean
    res[iInt, 2] = gtot-gl-bp[0]     # inh. mean
    res[iInt, 3:5] = bp[1:3]         # exc. and inh. SDs
    # momentary averages
    res[iInt, 5:9] = mean(res[:iInt+1,:], 0)[1:5]
        
    sf = open(resfile, 'a')         # open file and save result
    line = '%d\t' % (res[iInt, 0])
    for i in range(1, 9):
        line += '%2.5f\t' % (res[iInt, i])
            
    sf.write(line+'\n')
    sf.close()

# simulate with fitted parameters and compare
ge0, gi0, std_e, std_i = res[iInt, 5:9]
i_amp = 0.0
with open(os.path.join(data_dir, '%.2f' % i_amp, 'simulation_params.json'), 'r') as f:
    simulation_params = json.load(f)
simulation_params['i_inj'] = np.ones(to_idx(simulation_params['tstop'], simulation_params['dt'])+1) * i_amp
with open(os.path.join(data_dir, '%.2f' % i_amp, 'noise_params.json'), 'r') as f:
    noise_params_true = json.load(f)
seed = float(np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp, 'seed.txt')))

noise_params = {'g_e0': ge0, 'g_i0': gi0, 'std_e': std_e, 'std_i': std_i,
                'tau_e': noise_params_true['tau_e'], 'tau_i': noise_params_true['tau_i'],
                'E_e': noise_params_true['E_e'], 'E_i': noise_params_true['E_i']}

cell = Cell.from_modeldir(model_dir, mechanism_dir)
ou_process = ou_noise_input(cell, **noise_params)
ou_process.new_seed(seed)
v_new, t_new, _ = iclamp_handling_onset(cell, **simulation_params)
v = np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp, 'v.txt'))
t = np.arange(0, simulation_params['tstop'] + simulation_params['dt'], simulation_params['dt'])

# plots
pl.figure()
pl.plot(t, v, '0.3', label='True')
pl.plot(t_new, v_new, '0.7', label='Fit')
pl.xlabel('Time (ms)')
pl.ylabel('Membrane Potential (mV)')
pl.xlim(0, 1000)
pl.legend()
pl.tight_layout()
pl.savefig(os.path.join(save_dir, 'v_fitted.png'))

pl.figure()
width = 0.2
true_params = np.array([noise_params_true['g_e0'], noise_params_true['g_i0'], noise_params_true['std_e'],
        noise_params_true['std_i']])
estimated_params = np.array([ge0, gi0, std_e, std_i])
error_percent = (estimated_params-true_params) / true_params * 100
pl.bar(np.array([0, 1, 2, 3]) - width / 2., true_params, width=width, color='0.3', label='True')
pl.bar(np.array([0, 1, 2, 3]) + width / 2., estimated_params, width=width, color='0.7', label='Fit')
for i, e in enumerate(error_percent):
    min_val = min(estimated_params[i], true_params[i])
    max_val = max(estimated_params[i], true_params[i])
    pl.annotate('', xy=(i+width/2., min_val), xytext=(i+width/2., max_val), arrowprops=dict(arrowstyle='<->'))
    pl.annotate('%.2f %%' % e, xy=(i+width*7./10., min_val), xytext=(i+width*7./10., max_val - (max_val-min_val)/2.),
                va='center', ha='left')
pl.xticks([0, 1, 2, 3], ['$g_e^0$', '$g_i^0$', '$\sigma_e$', '$\sigma_i$'])
pl.ylabel('Conductance (uS)')
pl.legend()
pl.tight_layout()
pl.savefig(os.path.join(save_dir, 'parameters_fitted.png'))
pl.show()
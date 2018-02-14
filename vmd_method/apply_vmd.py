from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import json
from scipy.optimize import curve_fit
from grid_cell_stimuli.remove_APs import remove_APs
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics import to_idx


def gauss(x, mu, sig):
    return np.exp(-(x-mu)**2 / (2*sig**2))


def fit_gauss_to_hist(v, n_bins):
    hist, bins = np.histogram(v, bins=n_bins)
    norm_fac = np.max(hist)
    hist = hist / norm_fac
    bin_midpoints = np.array([(e_s + e_e) / 2 for (e_s, e_e) in zip(bins[:-1], bins[1:])])
    p_opt, _ = curve_fit(gauss, bin_midpoints, hist, p0=[np.mean(v), np.std(v)])
    return p_opt, bin_midpoints, norm_fac


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    save_dir = './results/'
    data_dir = '../simulate_model/data/model_pas/'
    model_dir = '../simulate_model/models/model_pas.json'
    i_amp1 = 0.0
    i_amp2 = -0.5
    n_bins = 100
    params_dir = os.path.join(data_dir, '%.2f' % i_amp1, 'model_params.json')

    # params remove APs
    AP_threshold = -20
    t_before = 10
    t_after = 10

    # load params
    with open(params_dir, 'r') as f:
        params = json.load(f)
        param_names = ['i_ext', 'c_m', 'gl', 'El', 'cell_area', 'Ee', 'Ei', 'te', 'ti', 'spike_threshold',
                       'dt', 'tstop']
        (i_ext, c_m, g_pas, E_pas, cell_area, Ee, Ei, tau_e, tau_i, spike_threshold, dt, tstop) = (params[k] for k in param_names)

    # load v
    v1 = np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp1, 'v.txt'))
    v2 = np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp2, 'v.txt'))
    t = np.arange(0, tstop+dt, dt)

    # remove APs
    v1 = remove_APs(v1, t, AP_threshold, t_before, t_after)
    v2 = remove_APs(v2, t, AP_threshold, t_before, t_after)

    # fit gaussian to histogram
    (mu1, sig1), bin_midpoints1, norm_fac1 = fit_gauss_to_hist(v1, n_bins)
    (mu2, sig2), bin_midpoints2, norm_fac2 = fit_gauss_to_hist(v2, n_bins)

    pl.figure()
    pl.hist(v1, bins=n_bins, weights=np.ones(len(v1)) / norm_fac1, color='b', alpha=0.5, label='$i_{inj}$=%.2f' % i_amp1)
    pl.hist(v2, bins=n_bins, weights=np.ones(len(v2)) / norm_fac2, color='r', alpha=0.5, label='$i_{inj}$=%.2f' % i_amp2)
    pl.plot(bin_midpoints1, gauss(bin_midpoints1, mu1, sig1), 'b')
    pl.plot(bin_midpoints2, gauss(bin_midpoints2, mu2, sig2), 'r')
    pl.xlabel('Membrane potential (mV)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.legend()
    #pl.show()

    print mu1, sig1
    print mu2, sig2

    # compute synaptic noise parameter
    ge0 = (((i_amp1 - i_amp2) * (sig2**2 * (Ei - mu1)**2 - sig1**2 * (Ei - mu2)**2)) /
           (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ee - Ei) * (mu1 - mu2)**2)
           - ((i_amp1 - i_amp2) * (Ei - mu2) + (i_amp2 - g_pas * (Ei - E_pas)) * (mu1 - mu2)) /
           ((Ee - Ei) * (mu1 - mu2)))  # if g_pas in uS -> cell_area should be removed otherwise cm**2 would not match units
    gi0 = (((i_amp1 - i_amp2) * (sig2**2 * (Ee - mu1)**2 - sig1**2 * (Ee - mu2)**2)) /
           (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ei - Ee) * (mu1 - mu2)**2)
           - ((i_amp1 - i_amp2) * (Ee -mu2) + (i_amp2 - g_pas * (Ee - E_pas)) * (mu1 - mu2)) /
           ((Ei - Ee) * (mu1 - mu2)))
    var_e = ((2 * c_m * (i_amp1 - i_amp2) * (sig1**2 * (Ei - mu2)**2
                                             - sig2**2 * (Ei - mu1)**2))
             / (tau_e * ((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ee - Ei) * (mu1 - mu2)**2))  # c_m in nF -> cell_area should be removed otherwise cm**2 would not match units
    var_i = ((2 * c_m * (i_amp1 - i_amp2) * (sig1**2 * (Ee - mu2)**2
                                             - sig2**2 * (Ee - mu1)**2))
             / (tau_i * ((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ei - Ee) * (mu1 - mu2)**2))
    # ge0 = max(0, ge0)
    # gi0 = max(0, gi0)
    var_e = max(0, var_e)
    var_i = max(0, var_i)
    std_e = np.sqrt(var_e)
    std_i = np.sqrt(var_i)
    print 'ge0: %.5f' % ge0 + ' uS'
    print 'gi0: %.5f' % gi0 + ' uS'
    print 'std_e: %.5f' % std_e + ' uS'
    print 'std_i: %.5f' % std_i + ' uS'

    # simulate with fitted parameter and compare (for i_amp1)
    with open(os.path.join(data_dir, '%.2f' % i_amp1, 'simulation_params.json'), 'r') as f:
        simulation_params = json.load(f)
    simulation_params['i_inj'] = np.ones(to_idx(tstop, dt)+1) * i_amp1
    with open(os.path.join(data_dir, '%.2f' % i_amp1, 'noise_params.json'), 'r') as f:
        noise_params = json.load(f)

    noise_params['g_e0'] = ge0
    noise_params['g_i0'] = gi0
    noise_params['std_e'] = std_e
    noise_params['std_i'] = std_i

    cell = Cell.from_modeldir(model_dir)
    ou_process = ou_noise_input(cell, **noise_params)
    v_new, t_new, _ = iclamp_handling_onset(cell, **simulation_params)

    pl.figure()
    pl.plot(t, v1, 'k')
    pl.plot(t_new, v_new, 'r')
    pl.xlabel('Time(ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.show()
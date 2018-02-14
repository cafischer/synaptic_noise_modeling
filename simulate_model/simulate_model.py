from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics import to_idx
from neuron import h
from grid_cell_stimuli.downsample import antialias_and_downsample
from vmd_method.apply_vmd import fit_gauss_to_hist


def get_cell_area(L, diam):
    """
    Takes length and diameter of some cell segment and returns the area of that segment (assuming it to be the surface
    of a cylinder without the circle surfaces as in Neuron).
    :param L: Length (um).
    :type L: float
    :param diam: Diameter (um).
    :type diam: float
    :return: Cell area (cm).
    :rtype: float
    """
    return L * diam * np.pi * 1e-8  # cm


def compute_fft(y, dt):
    """
    Compute FFT on y.

    :param y: Input array.
    :type y: array
    :param dt: time step in sec.
    :type dt: float
    :return: FFT of y, associated frequencies
    """
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)  # dt in sec

    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_y = fft_y[idx]
    return fft_y, freqs


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    save_dir = './data/model_pas/'
    model_dir = './models/model_pas.json'
    # noise_params = {'g_e0': 0.0121, 'g_i0': 0.0573, 'std_e': 0.0034641016151377544, 'std_i': 0.005138093031466052,
    #                 'tau_e': 2.73, 'tau_i': 10.49, 'E_e': 0, 'E_i': -75}
    noise_params = {'g_e0': 0.0121, 'g_i0': 0.0573, 'std_e': 0.0085, 'std_i': 0.0043,
                    'tau_e': 2.73, 'tau_i': 10.49, 'E_e': 0, 'E_i': -75}
    i_amp = 0.0  # nA
    tstop = 5000  # ms
    dt = 0.01  # ms
    simulation_params = {'sec': ('soma', None), 'i_inj': np.ones(to_idx(tstop, dt)+1)*i_amp, 'v_init': -75, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # old diam: 3.3203887148738738

    cell = Cell.from_modeldir(model_dir)
    ou_process = ou_noise_input(cell, **noise_params)

    # simulate
    g_e = h.Vector()
    g_e.record(ou_process._ref_g_e)
    g_i = h.Vector()
    g_i.record(ou_process._ref_g_i)
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    g_e = np.array(g_e)[to_idx(simulation_params['onset'], dt):]
    g_i = np.array(g_i)[to_idx(simulation_params['onset'], dt):]

    # plot and save
    save_dir = os.path.join(save_dir, '%.2f' % i_amp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # traces
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t, v, 'k')
    ax[0].set_ylabel('Membrane \nPotential \n(mV)', fontsize=16)
    ax[1].plot(t, g_i, 'k')
    ax[1].set_ylabel('Inhibitory \nConductance \n(uS)', fontsize=16)
    ax[2].plot(t, g_e, 'k')
    ax[2].set_ylabel('Excitatory \nConductance \n(uS)', fontsize=16)
    ax[2].set_xlabel('Time (ms)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(save_dir, 'traces.png'))
    #plt.show()

    # histograms
    (mu1, sig1), bin_midpoints1, norm_fac1 = fit_gauss_to_hist(g_i, n_bins=100)
    print mu1, sig1
    (mu1, sig1), bin_midpoints1, norm_fac1 = fit_gauss_to_hist(g_e, n_bins=100)
    print mu1, sig1

    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].hist(v, bins=100, color='k', weights=np.ones(len(v))/len(v))
    ax[0].set_xlabel('Membrane \nPotential (mV)', fontsize=16)
    ax[1].hist(g_i, bins=100, color='0.8', weights=np.ones(len(g_i))/len(g_i), label='inhibitory', alpha=0.5)
    ax[1].hist(g_e, bins=100, color='0.2', weights=np.ones(len(g_e))/len(g_e), label='excitatory', alpha=0.5)
    ax[1].legend(fontsize=16)
    ax[1].set_xlabel('Conductance \n(uS)', fontsize=16)
    ax[0].set_ylabel('Fraction', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(save_dir, 'hists.png'))
    #plt.show()

    # power spectrum
    cutoff_freq = 250  # Hz
    dt_new_max = 1. / cutoff_freq * 1000  # ms
    transition_width = 5.0  # Hz
    ripple_attenuation = 60.0  # db
    v_downsampled, t_downsampled, filter = antialias_and_downsample(v, dt, ripple_attenuation, transition_width,
                                                                    cutoff_freq, dt_new_max)
    g_i_downsampled, t_downsampled, filter = antialias_and_downsample(g_i, dt, ripple_attenuation, transition_width,
                                                                    cutoff_freq, dt_new_max)
    g_e_downsampled, t_downsampled, filter = antialias_and_downsample(g_e, dt, ripple_attenuation, transition_width,
                                                                    cutoff_freq, dt_new_max)
    v_fft, freqs = compute_fft(v_downsampled, (t_downsampled[1]-t_downsampled[0]) / 1000)
    g_i_fft, freqs = compute_fft(g_i_downsampled, (t_downsampled[1]-t_downsampled[0]) / 1000)
    g_e_fft, freqs = compute_fft(g_e_downsampled, (t_downsampled[1]-t_downsampled[0]) / 1000)

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
    ax[0].loglog(freqs, np.abs(v_fft) ** 2, 'k', label='Membrane Potential')
    ax[0].set_ylabel('Power')
    ax[0].set_xlabel('Frequency')
    ax[1].loglog(freqs, np.abs(g_i_fft) ** 2, '0.8', label='inhibitory')
    ax[1].loglog(freqs, np.abs(g_e_fft) ** 2, '0.2', label='excitatory')
    ax[1].set_xlabel('Frequency')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'power_spectra.png'))
    plt.show()

    # params for optimization
    cell_area = get_cell_area(cell.soma.L, cell.soma.diam)  # cm**2
    gl = cell.soma(.5).g_pas * cell_area * 1e6  # S/cm**2 * cm**2 * 1e6 = uS
    c_m = cell.soma.cm * cell_area * 1e3  # uF/cm**2 * cm**2 * 1e3 = nF
    gtot = gl + noise_params['g_e0'] + noise_params['g_i0']
    p_lower_bounds = [0.0, 0.0, 0.0]
    p_upper_bounds = [gtot - gl, 0.1, 0.1]

    dt = simulation_params['dt']
    te = noise_params['tau_e']
    ti = noise_params['tau_i']

    model_params = {'i_ext': i_amp, 'gtot': gtot, 'c_m': c_m, 'gl': gl, 'El': cell.soma(.5).e_pas,
                    'cell_area': cell_area,
                    'Ee': noise_params['E_e'], 'Ei': noise_params['E_i'],
                    'te': noise_params['tau_e'], 'ti': noise_params['tau_i'],
                    'spike_threshold': -20, 'dt': simulation_params['dt'], 'tstop': simulation_params['tstop'],
                    'p_lower_bounds': p_lower_bounds, 'p_upper_bounds': p_upper_bounds}

    np.savetxt(os.path.join(save_dir, 'v.txt'), v)
    with open(os.path.join(save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    simulation_params['i_inj'] = i_amp
    with open(os.path.join(save_dir, 'simulation_params.json'), 'w') as f:
        json.dump(simulation_params, f, indent=4)
    with open(os.path.join(save_dir, 'noise_params.json'), 'w') as f:
        json.dump(noise_params, f, indent=4)


    # TODO: cutting conductances at zero might give some artifacts (if g_e < 0: g_e = 0)
    # TODO: Rudolph, 2004: seems to use smaller stds than stated in the paper, can also be seen in the plots
    # TODO: Rudolph, 2004: i_ext1, i_ext2 not stated
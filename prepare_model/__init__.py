from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_fitting.read_heka import get_i_inj_from_function, get_sweep_index_for_amp
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


def prepare_synaptic_parameter_fitting(save_dir, model_dir, mechanism_dir, noise_params, simulation_params,
                                       i_amp, seed=1, plot=True):
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    ou_process = ou_noise_input(cell, **noise_params)
    ou_process.new_seed(seed)

    # simulate
    g_e = h.Vector()
    g_e.record(ou_process._ref_g_e)
    g_i = h.Vector()
    g_i.record(ou_process._ref_g_i)
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    g_e = np.array(g_e)[to_idx(simulation_params['onset'], simulation_params['dt']):]
    g_i = np.array(g_i)[to_idx(simulation_params['onset'], simulation_params['dt']):]

    # plot and save
    save_dir = os.path.join(save_dir, '%.2f' % i_amp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # params for optimization
    cell_area = get_cell_area(cell.soma.L, cell.soma.diam)  # cm**2
    gl = cell.soma(.5).g_pas * cell_area * 1e6  # S/cm**2 * cm**2 * 1e6 = uS
    c_m = cell.soma.cm * cell_area * 1e3  # uF/cm**2 * cm**2 * 1e3 = nF
    gtot = gl + noise_params['g_e0'] + noise_params['g_i0']  # uS
    dt = simulation_params['dt']

    model_params = {'i_ext': i_amp, 'gtot': gtot, 'c_m': c_m, 'gl': gl, 'El': cell.soma(.5).e_pas,
                    'cell_area': cell_area,
                    'Ee': noise_params['E_e'], 'Ei': noise_params['E_i'],
                    'te': noise_params['tau_e'], 'ti': noise_params['tau_i'],
                    'spike_threshold': -20, 'dt': simulation_params['dt'], 'tstop': simulation_params['tstop']}

    np.savetxt(os.path.join(save_dir, 'v.txt'), v)
    with open(os.path.join(save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    simulation_params['i_inj'] = i_amp
    with open(os.path.join(save_dir, 'simulation_params.json'), 'w') as f:
        json.dump(simulation_params, f, indent=4)
    with open(os.path.join(save_dir, 'noise_params.json'), 'w') as f:
        json.dump(noise_params, f, indent=4)
    np.savetxt(os.path.join(save_dir, 'seed.txt'), np.array([seed]))

    if plot:
        # traces
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(t, v, 'k')
        ax[0].set_ylabel('Membrane \nPotential \n(mV)', fontsize=16, rotation=0, labelpad=60,
                         verticalalignment='center', horizontalalignment='center')
        max_g = max(np.max(g_e), np.max(g_i))
        ax[1].plot(t, g_i, 'k')
        ax[1].set_ylim(0, max_g)
        ax[1].set_ylabel('Inhibitory \nConductance \n(uS)', fontsize=16, rotation=0, labelpad=60,
                         verticalalignment='center', horizontalalignment='center')
        ax[2].plot(t, g_e, 'k')
        ax[2].set_ylim(0, max_g)
        ax[2].set_ylabel('Excitatory \nConductance \n(uS)', fontsize=16, rotation=0, labelpad=60,
                         verticalalignment='center', horizontalalignment='center')
        ax[2].set_xlabel('Time (ms)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.08)
        plt.savefig(os.path.join(save_dir, 'traces.png'))

        # histograms
        (mu1, sig1), _, _, bins_v = fit_gauss_to_hist(v, 0.02)
        print mu1, sig1
        (mu1, sig1), _, _, bins_gi = fit_gauss_to_hist(g_i, 0.0005)
        print mu1, sig1
        (mu1, sig1), _, _, bins_ge = fit_gauss_to_hist(g_e, 0.0005)
        print mu1, sig1
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].hist(v, bins_v, color='k', weights=np.ones(len(v)) / len(v))
        ax[0].set_xlabel('Membrane \nPotential (mV)', fontsize=16)
        ax[1].hist(g_i, bins_gi, color='0.8', weights=np.ones(len(g_i)) / len(g_i), label='inhibitory', alpha=0.5)
        ax[1].hist(g_e, bins_ge, color='0.2', weights=np.ones(len(g_e)) / len(g_e), label='excitatory', alpha=0.5)
        ax[1].legend(fontsize=16)
        ax[1].set_xlabel('Conductance \n(uS)', fontsize=16)
        ax[0].set_ylabel('Fraction', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(os.path.join(save_dir, 'hists.png'))

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
        v_fft, freqs = compute_fft(v_downsampled, (t_downsampled[1] - t_downsampled[0]) / 1000)
        g_i_fft, freqs = compute_fft(g_i_downsampled, (t_downsampled[1] - t_downsampled[0]) / 1000)
        g_e_fft, freqs = compute_fft(g_e_downsampled, (t_downsampled[1] - t_downsampled[0]) / 1000)
        fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        ax[0].loglog(freqs, np.abs(v_fft) ** 2, 'k', label='Membrane Potential')
        ax[0].set_ylabel('Power')
        ax[0].set_xlabel('Frequency')
        ax[1].loglog(freqs, np.abs(g_i_fft) ** 2, '0.8', label='inhibitory')
        ax[1].loglog(freqs, np.abs(g_e_fft) ** 2, '0.2', label='excitatory')
        #ax[1].loglog(freqs, 1. / freqs, 'y', label='1/f')
        #ax[1].loglog(freqs, 1. / freqs**2, 'g', label='$1/f^2$')
        ax[1].set_xlabel('Frequency')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'power_spectra.png'))
        plt.show()


def prepare_passive_parameter_fitting(save_dir, model_dir, mechanism_dir, noise_params, simulation_params,
                                       i_amp, seed=1, plot=True):
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    ou_process = ou_noise_input(cell, **noise_params)
    ou_process.new_seed(seed)

    # simulate
    simulation_params['i_inj'] = get_i_inj_from_function('IV', [get_sweep_index_for_amp(i_amp, 'IV')],
                                                         simulation_params['tstop'], simulation_params['dt'])[0]
    g_e = h.Vector()
    g_e.record(ou_process._ref_g_e)
    g_i = h.Vector()
    g_i.record(ou_process._ref_g_i)
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    g_e = np.array(g_e)[to_idx(simulation_params['onset'], simulation_params['dt']):]
    g_i = np.array(g_i)[to_idx(simulation_params['onset'], simulation_params['dt']):]

    # plot and save
    save_dir = os.path.join(save_dir, '%.2f' % i_amp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # params for optimization
    cell_area = get_cell_area(cell.soma.L, cell.soma.diam)  # cm**2
    gl = cell.soma(.5).g_pas * cell_area * 1e6  # S/cm**2 * cm**2 * 1e6 = uS
    c_m = cell.soma.cm * cell_area * 1e3  # uF/cm**2 * cm**2 * 1e3 = nF
    gtot = gl + noise_params['g_e0'] + noise_params['g_i0']  # uS
    dt = simulation_params['dt']

    model_params = {'i_ext': i_amp, 'gtot': gtot, 'c_m': c_m, 'gl': gl, 'El': cell.soma(.5).e_pas,
                    'cell_area': cell_area,
                    'Ee': noise_params['E_e'], 'Ei': noise_params['E_i'],
                    'te': noise_params['tau_e'], 'ti': noise_params['tau_i'],
                    'spike_threshold': -20, 'dt': simulation_params['dt'], 'tstop': simulation_params['tstop']}

    np.savetxt(os.path.join(save_dir, 'v.txt'), v)
    np.savetxt(os.path.join(save_dir, 'i_inj.txt'), simulation_params['i_inj'])
    with open(os.path.join(save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    simulation_params['i_inj'] = i_amp
    with open(os.path.join(save_dir, 'simulation_params.json'), 'w') as f:
        json.dump(simulation_params, f, indent=4)
    with open(os.path.join(save_dir, 'noise_params.json'), 'w') as f:
        json.dump(noise_params, f, indent=4)
    np.savetxt(os.path.join(save_dir, 'seed.txt'), np.array([seed]))

    if plot:
        # traces
        fig, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(t, v, 'k')
        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'v.png'))
        plt.show()
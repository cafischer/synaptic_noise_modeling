import numpy as np
import json
import os
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_characteristics import to_idx
from neuron import h
import matplotlib.pyplot as pl


def simulate_model(save_dir, model_dir, mechanism_dir, noise_params, simulation_params, seed=1, plot=True):
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

    #
    g_pas = cell.soma(.5).g_pas  # S/cm**2
    e_pas = cell.soma(.5).e_pas  # mV

    if plot:
        # traces
        fig, ax = pl.subplots(3, 1, sharex=True)
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
        pl.tight_layout()
        pl.subplots_adjust(hspace=0.08)
        pl.show()
    return v, t, g_e, g_i, g_pas, e_pas


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    save_dir = './synapse_interp/data/model_pas'
    model_dir = '../prepare_model/models/model_pas.json'
    mechanism_dir = None

    # stim params
    # TODO: this one should not play a role now:
    stim = {
        "fr": 50,  # stimulation frequency (Hz)
        "delay": 0,  # delay b/w exc and inh (sec)
        "excAmp": 0,  # amplitude of exc stimulation
        "bias": 0  # bias of stimulation
    }

    # run params
    dt = 0.0001  # time step (sec)
    tstop = 1.0  # total simulation time (sec)
    t = np.arange(dt, tstop+dt, dt)
    run = {
        "random_stim": 1,
        "dt": dt,
        "T": tstop,
        "N": 100,  # number of particles
        "N_emsteps": 10,  # number of EM iterations
        "numBin": 10,  # width of each basis function in number of dt's  # TODO: maybe bigger
        "tt": t.tolist(),  # time array
        "lt": len(t)  # length of time array
    }

    simulation_params = {'sec': ('soma', None), 'i_inj': np.ones(to_idx(tstop*1000, dt*1000)), 'v_init': -75,
                         'tstop': tstop * 1000 - dt * 1000, 'dt': dt * 1000, 'celsius': 35, 'onset': 200}


    # model params
    noise_params = {'g_e0': 0.0121, 'g_i0': 0.0573, 'std_e': 0.0035, 'std_i': 0.0051,
                    'tau_e': 2.73, 'tau_i': 10.49, 'E_e': 0, 'E_i': -75}

    varObsNoise = 1e-06
    v, t, g_e, g_i, g_pas, e_pas = simulate_model(save_dir, model_dir, mechanism_dir, noise_params, simulation_params,
                                                  seed=1, plot=True)
    g_e *= 1000  # nS
    g_i *= 1000  # nS
    v_with_obsnoise = v + np.random.normal(0, np.sqrt(varObsNoise))

    v_noise = 0  # voltage evolution variance TODO
    model = {
        "tranProbName": "exp",  # could also use 'poiss', but the estimation is assuming Exp
        "tau_e": noise_params['tau_e'] / 1000.,  # exc time constant (sec)
        "tau_i": noise_params['tau_i'] / 1000.,  # inh time constant (sec)
        "v_e": noise_params['E_e'],  # exc reversal potential (mV)
        "v_i": noise_params['E_i'],  # inh reversal potential (mV)
        "g_leak": g_pas,  #  TODO: unit??? 1/membrane time constant (sec)
        "v_leak": e_pas,  # leak potential (mV)
        "v_noise": v_noise,  # voltage evolution variance
        "varObsNoise": varObsNoise,  # observation noise variance TODO
        "k_e_true": noise_params['g_e0'],  # exc weights; this parameter is estimated in the EM step TODO
        "k_i_true": noise_params['g_i0'],  # inh weights; this parameter is estimated in the EM step TODO
        "sdtv": np.sqrt(dt)*v_noise,  # TODO: std of noise added to Vm during integration, dont know why they added that
        "a_e": np.exp(-dt/(noise_params['tau_e'] / 1000.)),  # decay factor in one time step
        "a_i": np.exp(-dt/(noise_params['tau_i'] / 1000.)),  # decay factor in one time step
        "vs": map(lambda x: [x], v.tolist()),
        "obsNoiseyV": map(lambda x: [x], v_with_obsnoise.tolist()),
        #"true_mu_e": [(np.ones(len(v)) * noise_params['g_e0']).tolist()],  # TODO: temporary excitatory mean (e.g. sliding window)?
        #"true_mu_i": [(np.ones(len(v)) * noise_params['g_i0']).tolist()],  # TODO: temporary inhibitory mean (e.g. sliding window)?
        "gs_e": map(lambda x: [x], g_e.tolist()),  # TODO: unit?
        "gs_i": map(lambda x: [x], g_i.tolist())  # TODO: unit?
    }

    # save all dicts
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'model.json'), 'w') as f:
        json.dump(model, f, indent=4)
    with open(os.path.join(save_dir, 'stim.json'), 'w') as f:
        json.dump(stim, f, indent=4)
    with open(os.path.join(save_dir, 'run.json'), 'w') as f:
        json.dump(run, f, indent=4)
from __future__ import division
import numpy as np
import os
from prepare_model import prepare_synaptic_parameter_fitting, prepare_passive_parameter_fitting
from nrn_wrapper import load_mechanism_dir
from cell_characteristics import to_idx


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    save_dir = './data/model_stellate/'
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5/cell.json'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    noise_params = {'g_e0': 0.0121, 'g_i0': 0.0573, 'std_e': 0.0034641016151377544, 'std_i': 0.005138093031466052,
                    'tau_e': 2.73, 'tau_i': 10.49, 'E_e': 0, 'E_i': -75}
    # noise_params = {'g_e0': 0.008, 'g_i0': 0.02, 'std_e': 0.0005, 'std_i': 0.001,
    #                 'tau_e': 2.5, 'tau_i': 10.0, 'E_e': 0, 'E_i': -75}
    i_amp = -0.5  # nA
    tstop = 100000  # ms
    dt = 0.01  # ms
    simulation_params = {'sec': ('soma', None), 'i_inj': np.ones(to_idx(tstop, dt)+1)*i_amp, 'v_init': -75,
                         'tstop': tstop, 'dt': dt, 'celsius': 35, 'onset': 200}

    prepare_synaptic_parameter_fitting(save_dir, model_dir, mechanism_dir, noise_params, simulation_params, i_amp,
                                      seed=1, plot=True)

    # prepare_passive_parameter_fitting(os.path.join(save_dir, 'passive_parameter_fitting'), model_dir, mechanism_dir,
    #                                   noise_params, simulation_params, i_amp, plot=True)
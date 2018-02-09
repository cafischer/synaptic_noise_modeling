import numpy as np
import matplotlib.pyplot as plt
import os
import json
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_fitting.investigate_grid_cell_stimuli.model_noise.with_OU import ou_noise_input
from cell_characteristics import to_idx


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


if __name__ == '__main__':
    load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

    save_dir = './data/model_pas/'
    model_dir = './models/model_pas.json'
    noise_params = {'g_e0': 0.05, 'g_i0': 0.1, 'std_e': 0.5, 'std_i': 0.1, 'tau_e': 2.5, 'tau_i': 5.0, 'E_e': 0, 'E_i': -75}
    # 0.16014	-0.01014	-0.00000	0.00741  # -0.15760	0.30760	-0.00000	0.23352
    i_amp = 1.0
    tstop = 5000
    dt = 0.01
    simulation_params = {'sec': ('soma', None), 'i_inj': np.ones(to_idx(tstop, dt)+1)*i_amp, 'v_init': -75, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    cell = Cell.from_modeldir(model_dir)
    ou_process = ou_noise_input(cell, **noise_params)

    # simulate
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot and save
    save_dir = os.path.join(save_dir, '%.2f' % i_amp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(t, v)
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v.png'))
    #plt.show()

    np.savetxt(os.path.join(save_dir, 'v.txt'), v)

    # params for optimization
    cell_area = get_cell_area(cell.soma.L, cell.soma.diam)  # cm**2
    gl = cell.soma(.5).g_pas * cell_area * 1e6  # S/cm**2 * cm**2 * 1e6 = uS
    C = cell.soma.cm * cell_area * 1e3  # uF/cm**2 * cm**2 * 1e3 = nF
    gtot = gl + noise_params['g_e0'] + noise_params['g_i0']
    p_lower_bounds = [0.0, 0.0, 0.0]
    p_upper_bounds = [gtot - gl, 0.1, 0.1]

    dt = simulation_params['dt']
    te = noise_params['tau_e']
    ti = noise_params['tau_i']
    he1 = 1. - dt / te
    he2 = dt / te
    hi1 = 1. - dt / ti
    hi2 = dt / ti

    model_params = {'i_ext': i_amp, 'gtot': gtot, 'C': C, 'gl': gl, 'Vl': cell.soma(.5).e_pas,
                    'Ve': noise_params['E_e'], 'Vi': noise_params['E_i'], 'te': noise_params['tau_e'], 'ti': noise_params['tau_i'],
                    'spike_threshold': -20, 'dt': simulation_params['dt'], 'n_smooth': 3, 'n_ival': 100,
                    'p_lower_bounds': p_lower_bounds, 'p_upper_bounds': p_upper_bounds,
                    'he1': he1, 'he2': he2, 'hi1': hi1, 'hi2': hi2}
    with open(os.path.join(save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
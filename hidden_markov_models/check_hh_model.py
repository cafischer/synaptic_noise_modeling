import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_characteristics import to_idx
from cell_fitting.optimization.simulate import iclamp_handling_onset
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    model_dir = os.path.join('/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/cells/hhCell.json')
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/hodgkinhuxley'
    tstop = 500
    dt = 0.01
    #amp = 2.0
    amp = 0.5
    start_step = 100
    #end_step = 102
    end_step = 400
    v_init = -75
    celsius = 6.3

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)


    i_inj = np.zeros(to_idx(tstop, dt) + 1)
    i_inj[to_idx(start_step, dt):to_idx(end_step, dt)] = amp

    simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': celsius, 'onset': 0}

    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    pl.figure()
    pl.plot(t, v, 'k')
    pl.show()
import json
import os
from nrn_wrapper import load_mechanism_dir

load_mechanism_dir("/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/OU_process")

# vmFile = '../prepare_model/data/model_pas/0.00/v.txt'
# params_dir = '../prepare_model/data/model_pas/0.00/model_params.json'
# data_dir = '../prepare_model/data/model_pas/'
# model_dir = '../prepare_model/models/model_pas.json'
# mechanism_dir = None
# save_dir = './results/model_pas'

vmFile = '../prepare_model/data/model_stellate/0.00/v.txt'
params_dir = '../prepare_model/data/model_stellate/0.00/model_params.json'  # TODO
data_dir = '../prepare_model/data/model_stellate/'
model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5/cell.json'
mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
save_dir = './results/model_stellate/'

resfile = os.path.join(save_dir, 'p_opt.txt')

with open(params_dir, 'r') as f:
    params = json.load(f)
    param_names = ['i_ext', 'gtot', 'c_m', 'gl', 'El', 'Ee', 'Ei', 'te', 'ti', 'spike_threshold', 'dt']
    (Iext, gtot, C, gl, Vl, Ve, Vi, te, ti, vt, dt) = (params[k] for k in param_names)

t_pre = 5.                  # excluded time preceding spike
t_post = 10.                # excluded time after spike

n_smooth = 3

n_minISI = 1000             # min nb of datapoints in interval
n_maxISI = 10000            # max nb of datapoints in interval

g_start = [0.03, 0.001, 0.001]  # [0.03, 0.001, 0.001]
#'g_e0': 0.012, 'g_i0': 0.057, 'std_e': 0.0035, 'std_i': 0.0051,
pre = int(t_pre/dt)           # excluded pre spike steps
ahp = int(t_post/dt)          # excluded post spike steps

he1 = 1.-dt/te
he2 = dt/te
hi1 = 1.-dt/ti
hi2 = dt/ti

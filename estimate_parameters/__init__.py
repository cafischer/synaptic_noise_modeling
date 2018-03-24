from cell_characteristics.estimate_passive_parameter import estimate_passive_parameter
import os
import numpy as np
import json


# parameters to estimate: gtot, C, gl, Vl, Ve, Vi, te, ti
i_amp_save = 0.0
i_amp = -0.5
save_dir = '../prepare_model/data/model_stellate/'
data_dir = '../prepare_model/data/model_stellate/passive_parameter_fitting'
params_dir = os.path.join(data_dir, '%.2f' % i_amp, 'model_params.json')
v = np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp, 'v.txt'))
i_inj = np.loadtxt(os.path.join(data_dir, '%.2f' % i_amp, 'i_inj.txt'))
with open(params_dir, 'r') as f:
    params = json.load(f)
    param_names = ['i_ext', 'c_m', 'gl', 'El', 'gtot', 'cell_area', 'Ee', 'Ei', 'te', 'ti', 'spike_threshold',
                   'dt', 'tstop']
    (i_ext, c_m, g_pas, E_pas, gtot, cell_area, Ee, Ei, tau_e, tau_i, spike_threshold, dt, tstop) = (params[k] for k in
                                                                                               param_names)
t = np.arange(0, tstop+dt, dt)

# C, gl from small hyperpolarizing step
c_m_est, _, _, cell_area_est, _, g_tot_est = estimate_passive_parameter(v, t, i_inj)
g_tot_est = g_tot_est * 1e6 * cell_area_est  # uS
g_syn = gtot - g_pas
g_pas_est = g_tot_est - g_syn  # somewhat less than gtot?!
c_m_est *= 1e-3  # nF

print 'g_tot/g_tot est. (uS): %.3f / %.3f' % (gtot, g_tot_est)
print 'g_pas/g_pas est. (uS): %.3f / %.3f' % (g_pas, g_pas_est)
print 'cm/cm est. (nF): %.3f / %.3f' % (c_m, c_m_est)

# others
# E_pas_est = -85  # from models?
# print 'E_pas/E_pas est. (uS): %.3f / %.3f' % (E_pas, E_pas_est)

# # te, ti see Destexhe 2008 power spectrum -> looks weird, formula also requies other synaptic parameter?
# te = 3  # from Destexhe
# ti = 10  # from Destexhe
#
# # Ve, Vi from literature / standards
# Ve = 0
# Vi = -75

# save
save_dir = os.path.join(save_dir, '%.2f' % i_amp_save)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# model_params = {'i_ext': i_amp_save, 'gtot': g_tot_est, 'c_m': c_m_est, 'gl': g_pas_est, 'El': E_pas,
#                 'cell_area': cell_area,
#                 'Ee': Ee, 'Ei': Ei,
#                 'te': tau_e, 'ti': tau_i,
#                 'spike_threshold': spike_threshold, 'dt': dt, 'tstop': tstop}

model_params = {'i_ext': i_amp_save, 'gtot': gtot, 'c_m': c_m, 'gl': g_pas, 'El': E_pas,
                'cell_area': cell_area,
                'Ee': Ee, 'Ei': Ei,
                'te': tau_e, 'ti': tau_i,
                'spike_threshold': spike_threshold, 'dt': dt, 'tstop': tstop}
for k in model_params:
    percent = model_params[k] * 0.30
    model_params[k] = np.random.uniform(model_params[k] - percent, model_params[k] + percent)

with open(os.path.join(save_dir, 'model_params_est_rand.json'), 'w') as f:
    json.dump(model_params, f, indent=4)
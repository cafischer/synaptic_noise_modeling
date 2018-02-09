from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.optimize import curve_fit
from analyze_in_vivo.model_noise.estimate_passive_parameters import estimate_passive_parameters


def gauss(x, mu, sig):
    return np.exp(-(x-mu)**2 / (2*sig**2))


def fit_gauss_to_hist(v, n_bins):
    hist, bins = np.histogram(v, bins=n_bins)
    norm_fac = np.max(hist)
    hist = hist / norm_fac
    bin_midpoints = np.array([(e_s + e_e) / 2 for (e_s, e_e) in zip(bins[:-1], bins[1:])])
    p_opt, _ = curve_fit(gauss, bin_midpoints, hist, p0=(np.mean(v_mat_new[i]), np.std(v_mat_new[i])))
    return p_opt, bin_midpoints, norm_fac


if __name__ == '__main__':

    save_dir = './results/'
    data_dir = '../simulate_model/data/'
    amp1 = 0.0
    amp2 = 0.5
    n_bins = 25
    t_before = 3
    t_after = 6

    # load params TODO
    Ei = 0  # mV
    Ee = -75  # mV
    E_pas = -85  # mV
    tau_e = 2.5  # ms
    tau_i = 5  # ms
    c_m = 1  # uF/cm^2
    cell_area = 1
    g_pas = 1
    dt = 0.1
    tstop = 5000

    # load v
    v1 = np.loadtxt(os.path.join(data_dir, '%.2f' % amp1, 'v.txt'))
    v2 = np.loadtxt(os.path.join(data_dir, '%.2f' % amp1, 'v.txt'))
    t = np.arange(0, tstop+dt, dt)

    # remove APs TODO

    # fit gaussian to histogram
    p_opt, bin_midpoints, norm_fac = fit_gauss_to_hist(v1, n_bins)

    pl.figure()
    pl.hist(v1, bins=n_bins, weights=np.ones(len(v1))/norm_fac)
    pl.plot(bin_midpoints, gauss(bin_midpoints, p_opt[0], p_opt[1]), 'r')
    pl.plot(bin_midpoints, gauss(bin_midpoints, np.mean(v1), np.std(v1)), 'g')
    pl.xlabel('Membrane potential (mV)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.show()

    mu1, sig1 = p_opt

    save_dir_img = os.path.join(save_dir)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    ge0 = (((i_amp1 - i_amp2) * (sig2**2 * (Ei - mu1)**2 - sig1**2 * (Ei - mu2)**2)) /
           (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ee - Ei) * (mu1 - mu2)**2)
           - ((i_amp1 - i_amp2) * (Ei -mu2) + (i_amp2 - g_pas * cell_area * 1e6 * (Ei - E_pas)) * (mu1 - mu2)) /
           ((Ee - Ei) * (mu1 - mu2)))
    gi0 = (((i_amp1 - i_amp2) * (sig2**2 * (Ee - mu1)**2 - sig1**2 * (Ee - mu2)**2)) /
           (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ei - Ee) * (mu1 - mu2)**2)
           - ((i_amp1 - i_amp2) * (Ee -mu2) + (i_amp2 - g_pas * cell_area * 1e6 * (Ee - E_pas)) * (mu1 - mu2)) /
           ((Ei - Ee) * (mu1 - mu2)))
    var_e = max(0,
                (2 * cell_area * c_m * 1e3 * (i_amp1 - i_amp2) * (sig1 ** 2 * (Ei - mu2) ** 2 - sig2 ** 2 * (Ei - mu1) ** 2))
                / (tau_e * ((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ee - Ei) * (mu1 - mu2) ** 2))
    var_i = max(0,
                (2 * cell_area * c_m * 1e3 * (i_amp1 - i_amp2) * (sig1 ** 2 * (Ee - mu2) ** 2 - sig2 ** 2 * (Ee - mu1) ** 2))
                / (tau_i * ((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ei - Ee) * (mu1 - mu2) ** 2))

    print 'ge0: %.5f' % ge0 + ' uS'
    print 'gi0: %.5f' % gi0 + ' uS'
    print 'std_e: %.5f' % np.sqrt(var_e) + ' uS'
    print 'std_i: %.5f' % np.sqrt(var_i) + ' uS'
    pl.show()
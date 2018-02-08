def get_params():
    Iext = 0.  # constant injected current in nA
    gtot = 0.146928  # total input conductance in uS
    C = 0.33  # capacitance in nF
    gl = 0.0187  # leak conductance in uS
    Vl = -99.  # leak reversal potential in mV
    Ve = 0.  # rev. potential of exc. in mV
    Vi = -75.  # rev. potential of inh. in mV
    te = 2.728  # exc. corr. time constant in ms
    ti = 10.49  # inh. corr. time constant in ms

    spike_threshold = -20.  # threshold for spike detection
    dt = 0.048  # time step in ms

    n_smooth = 3
    n_ival = 100  # nb of intervals analysed

    p_start = [0.03, 0.001, 0.001]

    he1 = 1. - dt / te
    he2 = dt / te
    hi1 = 1. - dt / ti
    hi2 = dt / ti
    return Iext, gtot, C, gl, Vl, Ve, Vi, te, ti, spike_threshold, dt, n_smooth, n_ival, p_start, he1, he2, hi1, hi2


# target values of the file 'vm_trace.txt'
# ge=0.032057
# gi=0.096171
# se=0.008014
# si=0.024043
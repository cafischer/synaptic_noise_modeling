from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from hodgkin_huxley_model import *
pl.style.use('paper')


def create_hh_model(plot=False):
    cell = Cell(cm=1, length=10, diam=50, ionchannels=[get_pas_hh(), get_nat_hh(), get_kdr_hh()])
    hhmodel = HMMModel(cell, exc_synapse=Synapse(0, 2.5), inh_synapse=Synapse(-70, 10.0))

    # define start values
    V0 = -75
    ge0 = 0  # (S/cm^2)
    gi0 = 0  # (S/cm^2)
    p_gates0 = [ionchannel.init_gates(V0) for ionchannel in cell.ionchannels]

    # generate testing data
    tstop = 500
    dt = 0.01
    n_timesteps = int(round(tstop / dt)) + 1
    # i_inj = lambda x: 2.0 if 100 <= x <= 102 else 0.0
    i_inj = lambda x: 0.5 if 100 <= x <= 400 else 0.0

    p_gates = [0] * n_timesteps
    n_vars = 3
    Z = np.zeros((n_timesteps, n_vars, 1))
    X = np.zeros((n_timesteps, 1, 1))
    A = np.zeros((n_timesteps, n_vars, n_vars))  # will stay empty at 0
    b = np.zeros((n_timesteps, n_vars, 1))  # will stay empty at 0
    p_gates[0] = p_gates0

    mu0 = np.array([[V0], [ge0], [gi0]])
    tiny_var = 1e-8  # just so that var is > 0
    P0 = np.diag([tiny_var, 1.e-6, 2.e-6])  # must be >0 for inverses etc.
    Lambda = np.diag([tiny_var, 1.e-6, 2.e-6])  # must be >0 for inverses etc.
    Sigma = np.array([[1e-1]])  # must be >0 for inverses etc.

    Z[0] = mu0 + np.array([np.diag(P0) * np.random.randn(n_vars)]).T
    X[0] = np.dot(hhmodel.C, Z[0]) + np.array([np.diag(Sigma) * np.random.randn(len(Sigma))]).T

    for n in range(1, n_timesteps):
        ts = n * dt
        trans_mat, trans_const, p_gates[n] = hhmodel.get_trans(Z[n-1, 0], Z[n-1, 1], Z[n-1, 2], p_gates[n-1],
                                                               i_inj(ts), dt)
        Z[n] = (np.dot(trans_mat, Z[n-1]) + trans_const) + np.array([np.diag(Lambda) * np.random.randn(n_vars)]).T
        X[n] = np.dot(hhmodel.C, Z[n]) + np.array([np.diag(Sigma) * np.random.randn(len(Sigma))]).T
        A[n] = trans_mat
        b[n] = trans_const

    if plot:
        t = np.arange(0, tstop + dt, dt)
        pl.figure()
        pl.plot(t, Z[:, 0], 'k', label='without obs. noise')
        pl.plot(t, X[:, 0], 'b', label='with obs. noise')
        pl.ylabel('Membrane Potential (mV)')
        pl.xlabel('Time (ms)')
        pl.legend(fontsize=16)
        pl.tight_layout()

        pl.figure()
        p_nat = np.array([p[1] for p in p_gates])
        p_kdr = np.array([p[2] for p in p_gates])
        pl.plot(t, p_nat[:, 0], 'r', label='nat m')
        pl.plot(t, p_nat[:, 1], 'b', label='nat h')
        pl.plot(t, p_kdr, 'g', label='kdr n')
        pl.ylabel('Open Probability')
        pl.xlabel('Time (ms)')
        pl.legend(fontsize=16)
        pl.tight_layout()

        pl.figure()
        pl.plot(t, Z[:, 1], 'r', label='$g_e$')
        pl.plot(t, Z[:, 2], 'b', label='$g_i$')
        pl.ylabel('Synaptic Noise Conductance (S/cm$^2$)')
        pl.xlabel('Time (ms)')
        pl.legend(fontsize=16)
        pl.tight_layout()
        pl.show()

    return Z, X, mu0, P0, A, b, Lambda, hhmodel.C, Sigma, n_timesteps, n_vars


def get_pas_hh():
    channel_pas = IonChannel(0.0003, -54.4, 0, None, lambda v: [], lambda v: [])
    return channel_pas


def get_kdr_hh():
    def alpha_n(v):
        return .01 * (-(v + 55) / (np.exp(-(v + 55) / 10) - 1))

    def beta_n(v):
        return .125 * np.exp(-(v + 65) / 80)

    def inf_gates(v):
        return [alpha_n(v) / (alpha_n(v) + beta_n(v))]

    def tau_gates(v):
        return [1 / (alpha_n(v) + beta_n(v))]

    channel_kdr = IonChannel(0.036, -77, 1, [4], inf_gates, tau_gates)
    return channel_kdr


def get_nat_hh():
    def alpha_m(v):
        return .1 * -(v + 40) / (np.exp(-(v + 40) / 10) - 1)

    def beta_m(v):
        return 4 * np.exp(-(v + 65) / 18)

    def alpha_h(v):
        return .07 * np.exp(-(v + 65) / 20)

    def beta_h(v):
        return 1 / (np.exp(-(v + 35) / 10) + 1)

    def inf_gates(v):
        return [alpha_m(v) / (alpha_m(v) + beta_m(v)),
                alpha_h(v) / (alpha_h(v) + beta_h(v))]

    def tau_gates(v):
        return [1 / (alpha_m(v) + beta_m(v)),
                1 / (alpha_h(v) + beta_h(v))]

    channel_nat = IonChannel(0.12, 50, 2, [3, 1], inf_gates, tau_gates)
    return channel_nat
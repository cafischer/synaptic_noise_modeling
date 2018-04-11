import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import multivariate_normal
from hodgkin_huxley_model import *
pl.style.use('paper')


def kalman_filter(X, mu0, V0, A, Gamma, C, Sigma, n_timesteps, n_vars):
    mus = np.zeros((n_timesteps, n_vars))
    Vs = np.zeros((n_timesteps, n_vars))
    cs = np.zeros(n_timesteps)
    P = np.zeros((n_timesteps, n_vars))
    K = np.zeros((n_timesteps, n_vars))

    # forward
    P[0] = np.dot(np.dot(A, Vs[0]), A.T) + Gamma
    K[1] = np.dot(np.dot(P[0], C.T), np.linalg.inv(np.dot(np.dot(C, P[0]), C.T) + Sigma))
    mus[0] = mu0
    Vs[0] = V0

    mus[1] = mus[0] + np.dot(K[1], (X[1] - np.dot(C, mus[0])))
    Vs[1] = np.dot((np.eye(n_vars) - np.dot(K[1], C)), Vs[0])
    cs[1] = multivariate_normal.pdf(X[1], np.dot(C, mus[0]), np.dot(np.dot(C, Vs[0]), C.T) + Sigma)

    for n in range(2, n_timesteps):
        P[n-1] = np.dot(np.dot(A, Vs[n-1]), A.T) + Gamma
        K[n] = np.dot(np.dot(P[n-1], C.T), np.linalg.inv(np.dot(np.dot(C, P[n-1]), C.T) + Sigma))

        mus[n] = np.dot(A, mus[n-1]) + np.dot(K[n], (X[n] - np.dot(np.dot(C, A), mus[n-1])))
        Vs[n] = np.dot((np.eye(n_vars) - np.dot(K[n], C)), P[n-1])
        cs[n] = multivariate_normal.pdf(X[n], np.dot(np.dot(C, A), mus[n-1]), np.dot(np.dot(C, P[n-1]), C.T) + Sigma)

    # backward
    mu_gamma = np.zeros((n_timesteps, n_vars))  # of gamma(z_n)
    V_gamma = np.zeros((n_timesteps, n_vars))  # of gamma(z_n)
    V_xi = np.zeros((n_timesteps, n_vars))  # of xi(z_n-1, z_n)
    J = np.zeros((n_timesteps, n_vars))

    for n in range(n_timesteps-2, -1, -1):
        J[n] = np.dot(np.dot(Vs[n], A.T), np.linalg.inv(P[n]))

        mu_gamma[n] = mus[n] + np.dot(J[n], (mu_gamma[n+1] - np.dot(A, mus[n_timesteps-1])))
        V_gamma[n] = Vs[n] + np.dot(np.dot(J[n], (V_gamma[n+1] - P[n])), J[n].T)

    for n in range(1, n_timesteps):
        V_xi[n] = np.dot(J[n-1], V_gamma[n])

    # Likelihood for monitoring
    p_x = np.prod(cs)
    log_p_x = np.log(p_x)

    return mu_gamma, V_gamma, V_xi, J, log_p_x


def do_expectation(mu_gamma, V_gamma, J):
    E_z_n = mu_gamma
    E_z_n_z_n_min_1 = np.dot(J[:-1], V_gamma[1:]) + np.dot(mu_gamma[:-1], np.array([mu_gamma[1:]]).T)
    E_z_n_z_n = V_gamma + np.dot(mu_gamma, np.array([mu_gamma]).T)
    return E_z_n, E_z_n_z_n_min_1, E_z_n_z_n


def do_maximization():
    pass

if __name__ == '__main__':
    # create model
    def inf_gates(v):
        return []
    def tau_gates(v):
        return []

    channel_pas = IonChannel(0.0003, -54.4, 0, None, inf_gates, tau_gates)

    def alpha_m(v):
        return .1 * -(v+40) / (np.exp(-(v+40) / 10) -1)
    def beta_m(v):
        return 4 * np.exp(-(v+65) / 18)
    def alpha_h(v):
        return .07 * np.exp(-(v+65) / 20)
    def beta_h(v):
        return 1 / (np.exp(-(v+35) / 10) + 1)

    def inf_gates(v):
        return [alpha_m(v) / (alpha_m(v) + beta_m(v)),
                alpha_h(v) / (alpha_h(v) + beta_h(v))]

    def tau_gates(v):
        return [1 / (alpha_m(v) + beta_m(v)),
                1 / (alpha_h(v) + beta_h(v))]

    channel_nat = IonChannel(0.12, 50, 2, [3, 1], inf_gates, tau_gates)

    def alpha_n(v):
        return .01 * (-(v+55) / (np.exp(-(v+55) / 10) - 1))
    def beta_n(v):
        return .125 * np.exp(-(v+65)/80)
    def inf_gates(v):
        return [alpha_n(v) / (alpha_n(v) + beta_n(v))]
    def tau_gates(v):
        return [1 / (alpha_n(v) + beta_n(v))]

    channel_kdr = IonChannel(0.036, -77, 1, [4], inf_gates, tau_gates)

    ionchannels = [channel_pas, channel_nat, channel_kdr]
    cell = Cell(cm=1, length=10, diam=50, ionchannels=ionchannels)
    exc_synapse = Synapse(0, 2.5)
    inh_synapse = Synapse(-70, 10.0)
    hhmodel = HMMModel(cell, exc_synapse, inh_synapse)

    # define start values
    V0 = -75
    ge0 = 0  # (S)
    gi0 = 0  # (S)
    p_gates0 = [ionchannel.init_gates(V0) for ionchannel in cell.ionchannels]

    # generate testing data
    tstop = 500
    dt = 0.01
    n_timesteps = int(round(tstop/dt)) + 1
    #i_inj = lambda x: 2.0 if 100 <= x <= 102 else 0.0
    i_inj = lambda x: 0.5 if 100 <= x <= 400 else 0.0

    p_gates = [0] * n_timesteps
    n_vars = 3
    X = np.zeros((n_timesteps, n_vars))
    p_gates[0] = p_gates0
    X[0, :] = [V0, ge0, gi0]
    Gamma = np.array([0, 1.e-11, 2.e-11])

    for n in range(1, n_timesteps):
        ts = n * dt
        trans_mat, trans_const, p_gates[n] = hhmodel.get_trans(X[n-1, 0], X[n-1, 1], X[n-1, 2], p_gates[n-1],
                                                               i_inj(ts), dt)
        X[n, :] = (np.dot(trans_mat, np.array([X[n-1, :]]).T) + trans_const).flatten()
        X[n, :] += Gamma * np.random.randn(3)

    # plot
    t = np.arange(0, tstop+dt, dt)
    pl.figure()
    pl.plot(t, X[:, 0], 'k')

    pl.figure()
    pl.plot(t, [p[1] for p in p_gates], 'orange')
    pl.plot(t, [p[2] for p in p_gates], 'g')

    pl.figure()
    pl.plot(t, X[:, 1], 'r', label='$g_e$')
    pl.plot(t, X[:, 2], 'b', label='$g_i$')
    pl.legend()
    pl.show()
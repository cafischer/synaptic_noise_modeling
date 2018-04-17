from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from hodgkin_huxley_model import *
pl.style.use('paper')


def create_simple_model(seed=0, plot=False):
    np.random.seed(seed)

    tstop = 500
    dt = 0.01
    n_timesteps = int(round(tstop / dt)) + 1
    n_vars = 2
    n_obs = 1

    Z = np.zeros((n_timesteps, n_vars, 1))
    X = np.zeros((n_timesteps, n_obs, 1))

    A = np.array([[-0.5, 1], [-1, 0.5]])
    b = np.array([[0.0], [0.0]])
    C = np.array([[1, 0]])
    mu0 = np.array([[-1.0], [1.0]])
    P0 = np.diag([0.5, 0.01])  # must be >0 for inverses etc.
    Lambda = np.diag([0.5, 0.01])  # must be >0 for inverses etc.
    Sigma = np.array([[1.0]])  # must be >0 for inverses etc.

    Z[0] = mu0 + np.array([np.diag(P0) * np.random.randn(n_vars)]).T
    X[0] = np.dot(C, Z[0]) + np.array([np.diag(Sigma) * np.random.randn(len(Sigma))]).T

    for n in range(1, n_timesteps):
        Z[n] = (np.dot(A, Z[n-1]) + b) + np.array([np.diag(Lambda) * np.random.randn(n_vars)]).T
        X[n] = np.dot(C, Z[n]) + np.array([np.diag(Sigma) * np.random.randn(len(Sigma))]).T

    if plot:
        t = np.arange(0, tstop + dt, dt)
        pl.figure()
        pl.plot(t, Z[:, 0], 'k', label='without obs. noise')
        pl.plot(t, X[:, 0], 'b', label='with obs. noise')
        pl.ylabel('Observed')
        pl.xlabel('Time (ms)')
        pl.legend(fontsize=16)
        pl.tight_layout()

        pl.figure()
        pl.plot(t, Z[:, 0], 'g', label='0')
        pl.plot(t, Z[:, 1], 'y', label='1')
        pl.ylabel('Hidden')
        pl.xlabel('Time (ms)')
        pl.legend(fontsize=16)
        pl.tight_layout()
        pl.show()

    return Z, X, mu0, P0, A, b, Lambda, C, Sigma, n_timesteps, n_vars, n_obs
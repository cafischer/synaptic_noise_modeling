from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import multivariate_normal
from hodgkin_huxley_model import *
from create_hh_model import create_hh_model
from LDS_simple_example import create_simple_model
pl.style.use('paper')


def kalman_filter(X, mu0, P0, A, b, Gamma, C, Sigma, n_timesteps, n_vars):
    # forward
    mus = np.zeros((n_timesteps, n_vars, 1))
    Vs = np.zeros((n_timesteps, n_vars, n_vars))
    cs = np.zeros(n_timesteps)
    P = np.zeros((n_timesteps, n_vars, n_vars))
    K = np.zeros((n_timesteps, n_vars, np.shape(Sigma)[0]))

    K[0] = np.dot(np.dot(P0, C.T), np.linalg.inv(np.dot(np.dot(C, P0), C.T) + Sigma))

    mus[0] = mu0 + np.dot(K[0], (X[0] - np.dot(C, mu0)))
    Vs[0] = np.dot((np.eye(n_vars) - np.dot(K[0], C)), P0)
    cs[0] = multivariate_normal.pdf(X[0].flatten(), np.dot(C, mu0).flatten(), np.dot(np.dot(C, P0), C.T) + Sigma)

    for n in range(1, n_timesteps):
        P[n-1] = np.dot(np.dot(A[n], Vs[n-1]), A[n].T) + Gamma
        K[n] = np.dot(np.dot(P[n-1], C.T), np.linalg.inv(np.dot(np.dot(C, P[n-1]), C.T) + Sigma))

        mus[n] = ((np.dot(A[n], mus[n-1]) + b[n])
                  + np.dot(K[n], (X[n] - np.dot(C, np.dot(A[n], mus[n-1]) + b[n]))))
        Vs[n] = np.dot((np.eye(n_vars) - np.dot(K[n], C)), P[n-1])
        cs[n] = multivariate_normal.pdf(X[n].flatten(), np.dot(np.dot(C, A[n]), mus[n-1]).flatten(),
                                        np.dot(np.dot(C, P[n-1]), C.T) + Sigma)

    # backward
    mu_gamma = np.zeros((n_timesteps, n_vars, 1))  # of gamma(z_n)
    V_gamma = np.zeros((n_timesteps, n_vars, n_vars))  # of gamma(z_n)
    J = np.zeros((n_timesteps, n_vars, n_vars))

    mu_gamma[n_timesteps-1] = mus[n_timesteps-1]
    V_gamma[n_timesteps-1] = Vs[n_timesteps-1]

    for n in range(n_timesteps-2, -1, -1):
        J[n] = np.dot(np.dot(Vs[n], A[n+1].T), np.linalg.inv(P[n]))

        mu_gamma[n] = mus[n] + np.dot(J[n], (mu_gamma[n+1] - (np.dot(A[n+1], mus[n]) + b[n+1])))
        V_gamma[n] = Vs[n] + np.dot(np.dot(J[n], (V_gamma[n+1] - P[n])), J[n].T)

    # Likelihood for monitoring
    p_x = np.prod(cs)
    log_p_x = np.log(p_x)

    return mu_gamma, V_gamma, J, log_p_x


def kalman_filter_transition_time_independent(X, params_fit, n_timesteps, n_vars):
    mu0, P0, A, b, Gamma, C, Sigma = params_fit['mu0'], params_fit['P0'], params_fit['A'], params_fit['b'], \
                                     params_fit['Gamma'], params_fit['C'], params_fit['Sigma']

    # forward
    mus = np.zeros((n_timesteps, n_vars, 1))
    Vs = np.zeros((n_timesteps, n_vars, n_vars))
    cs = np.zeros(n_timesteps)
    P = np.zeros((n_timesteps, n_vars, n_vars))
    K = np.zeros((n_timesteps, n_vars, np.shape(Sigma)[0]))

    K[0] = np.dot(np.dot(P0, C.T), np.linalg.inv(np.dot(np.dot(C, P0), C.T) + Sigma))

    mus[0] = mu0 + np.dot(K[0], (X[0] - np.dot(C, mu0)))
    Vs[0] = np.dot((np.eye(n_vars) - np.dot(K[0], C)), P0)
    cs[0] = multivariate_normal.pdf(X[0].flatten(), np.dot(C, mu0).flatten(), np.dot(np.dot(C, P0), C.T) + Sigma)

    for n in range(1, n_timesteps):
        P[n-1] = np.dot(np.dot(A, Vs[n-1]), A.T) + Gamma
        K[n] = np.dot(np.dot(P[n-1], C.T), np.linalg.inv(np.dot(np.dot(C, P[n-1]), C.T) + Sigma))

        mus[n] = ((np.dot(A, mus[n-1]) + b)
                  + np.dot(K[n], (X[n] - np.dot(C, np.dot(A, mus[n-1]) + b))))
        Vs[n] = np.dot((np.eye(n_vars) - np.dot(K[n], C)), P[n-1])
        cs[n] = multivariate_normal.pdf(X[n].flatten(), np.dot(np.dot(C, A), mus[n-1]).flatten(),
                                        np.dot(np.dot(C, P[n-1]), C.T) + Sigma)

    # backward
    mu_gamma = np.zeros((n_timesteps, n_vars, 1))  # of gamma(z_n)
    V_gamma = np.zeros((n_timesteps, n_vars, n_vars))  # of gamma(z_n)
    J = np.zeros((n_timesteps, n_vars, n_vars))

    mu_gamma[n_timesteps-1] = mus[n_timesteps-1]
    V_gamma[n_timesteps-1] = Vs[n_timesteps-1]

    for n in range(n_timesteps-2, -1, -1):
        J[n] = np.dot(np.dot(Vs[n], A.T), np.linalg.inv(P[n]))

        mu_gamma[n] = mus[n] + np.dot(J[n], (mu_gamma[n+1] - (np.dot(A, mus[n]) + b)))
        V_gamma[n] = Vs[n] + np.dot(np.dot(J[n], (V_gamma[n+1] - P[n])), J[n].T)

    # Likelihood for monitoring
    log_p_x = np.sum(np.log(cs))

    return mu_gamma, V_gamma, J, log_p_x


def do_expectation(mu_gamma, V_gamma, J, n_timesteps, n_vars):
    E_z_n_z_n_min_1 = np.zeros((n_timesteps-1, n_vars, n_vars))
    E_z_n_z_n = np.zeros((n_timesteps, n_vars, n_vars))

    E_z_n = copy.copy(mu_gamma)
    for n in range(1, n_timesteps):
        E_z_n_z_n_min_1[n-1] = np.dot(V_gamma[n], J[n-1].T) + np.dot(mu_gamma[n], mu_gamma[n-1].T)
    for n in range(n_timesteps):
        E_z_n_z_n[n] = V_gamma[n] + np.dot(mu_gamma[n], mu_gamma[n].T)
    return E_z_n, E_z_n_z_n_min_1, E_z_n_z_n


def do_maximization(X, params_fit, params_to_fit, E_z_n, E_z_n_z_n_min_1, E_z_n_z_n):

    if params_to_fit['mu0']:
        mu0_est = E_z_n[0]
        params_fit['mu0'] = mu0_est

    if params_to_fit['P0']:
        P0_est = E_z_n_z_n[0] - np.dot(E_z_n[0], E_z_n[0].T)
        params_fit['P0'] = P0_est

    if params_to_fit['b']:
        b_est = 1/(n_timesteps - 1) * np.sum([E_z_n[n] - np.dot(params_fit['A'], E_z_n[n-1])
                                              for n in range(1, n_timesteps)], 0)
        params_fit['b'] = b_est

    if params_to_fit['A']:
        b_est = params_fit['b']
        A_est = np.dot(np.sum([E_z_n_z_n_min_1[n] - np.dot(b_est, E_z_n[n].T) for n in range(0, n_timesteps-1)], 0),
                       np.linalg.inv(np.sum(E_z_n_z_n[:-1], 0)))
        params_fit['A'] = A_est

    if params_to_fit['Gamma']:
        A_est = params_fit['A']
        b_est = params_fit['b']
        Gamma_est = 1/(n_timesteps - 1) * np.sum(
            [E_z_n_z_n[n+1] - np.dot(E_z_n_z_n_min_1[n], A_est.T) - np.dot(E_z_n[n+1], b_est.T)
             - np.dot(A_est, E_z_n_z_n_min_1[n].T) - np.dot(b_est, E_z_n[n+1].T)
             + np.dot(np.dot(A_est, E_z_n_z_n[n]), A_est.T) + np.dot(np.dot(A_est, E_z_n[n]), b_est.T)
             + np.dot(np.dot(b_est, E_z_n[n].T), A_est.T) + np.dot(b_est, b_est.T)
             for n in range(n_timesteps-1)], 0
        )
        params_fit['Gamma'] = Gamma_est

    if params_to_fit['C']:
        C_est = np.dot(np.sum([np.dot(X[n], E_z_n[n].T) for n in range(n_timesteps)], 0),
                       np.linalg.inv(np.sum(E_z_n_z_n, 0)))
        params_fit['C'] = C_est

    if params_to_fit['Sigma']:
        C_est = params_fit['C']
        Sigma_est = 1/n_timesteps * np.sum(
            [np.dot(X[n], X[n].T) - np.dot(np.dot(C_est, E_z_n[n]), X[n].T)
            - np.dot(np.dot(X[n], E_z_n[n].T), C_est.T) + np.dot(np.dot(C_est, E_z_n_z_n[n]), C_est.T)
             for n in range(n_timesteps)], 0
        )
        params_fit['Sigma'] = Sigma_est


if __name__ == '__main__':
    # Z, X, mu0, P0, A, b, Lambda, C, Sigma, n_timesteps, n_vars = create_hh_model()  # dont change A, b or C they are given by the HHmodel + use time dependent variant (TODO update)
    Z, X, mu0, P0, A, b, Gamma, C, Sigma, n_timesteps, n_vars, n_obs = create_simple_model(seed=3)

    # try estimation of parameters with EM
    max_iter = 1
    tol = 1e-8
    printing = True

    convergence = False
    p_x_log_old = np.inf

    params_to_fit = {'mu0': True, 'P0':True, 'A': False, 'b': False, 'Gamma': True, 'C': False, 'Sigma': True}
    params_true = {'mu0': mu0, 'P0': P0, 'A': A, 'b': b, 'Gamma': Gamma, 'C': C, 'Sigma': Sigma}
    params_fit = {'mu0': None, 'P0': None, 'A': A, 'b': b, 'Gamma': None, 'C': C, 'Sigma': None}
    params_type = {'mu0': ('array', n_vars, 1), 'P0': ('mat', n_vars, n_vars), 'A': ('mat', n_vars, n_vars),
                   'b': ('array', n_vars, 1), 'Gamma': ('mat', n_vars, n_vars), 'C': ('mat', n_obs, n_vars),
                   'Sigma': ('mat', n_obs, n_obs)}

    # initialization of parameters (zeros for vectors, identity matrix for matrices)
    for p_name, p_val in params_fit.iteritems():
        if params_to_fit[p_name]:
            array_type, dim1, dim2 = params_type[p_name]
            if array_type == 'array':
                params_fit[p_name] = np.zeros((dim1, dim2))
            elif array_type == 'mat':
                params_fit[p_name] = np.eye(dim1, dim2)
        else:
            if params_fit[p_name] is None:
                raise ValueError('Parameter %s not given' % p_name)

    for iteration in range(max_iter):
        mu_gamma, V_gamma, J, log_p_x = kalman_filter_transition_time_independent(X, params_fit,
                                                                                  n_timesteps, n_vars)

        E_z_n, E_z_n_z_n_min_1, E_z_n_z_n = do_expectation(mu_gamma, V_gamma, J, n_timesteps, n_vars)
        do_maximization(X, params_fit, params_to_fit, E_z_n, E_z_n_z_n_min_1, E_z_n_z_n)

        # check convergence
        if np.abs(log_p_x - p_x_log_old) <= tol:
            convergence = True
            break
        p_x_log_old = log_p_x

        if printing:
            print('Estimated parameters iteration %i: ' % iteration)
            for p_name, p_val in params_fit.iteritems():
                if params_to_fit[p_name]:
                    if params_type[p_name][0] == 'array':
                        val = ['%.3f' % s for s in p_val]
                    elif params_type[p_name][0] == 'mat':
                        val = [['%.3f' % s for s in s_row] for s_row in p_val]
                    print(p_name + ': ' + str(val))
            print()

    if printing:
        print()
        print('converged: ', convergence)
        print('# iterations: ', iteration)
        print()
        print('Estimated parameters: ')
        for p_name, p_val in params_fit.iteritems():
            if params_to_fit[p_name]:
                if params_type[p_name][0] == 'array':
                    val = ['%.3f' % s for s in p_val]
                elif params_type[p_name][0] == 'mat':
                    val = [['%.3f' % s for s in s_row] for s_row in p_val]
                print(p_name + ': ' + str(val))
        print()
        print('True parameters: ')
        for p_name, p_val in params_true.iteritems():
            if params_to_fit[p_name]:
                if params_type[p_name][0] == 'array':
                    val = ['%.3f' % s for s in p_val]
                elif params_type[p_name][0] == 'mat':
                    val = [['%.3f' % s for s in s_row] for s_row in p_val]
                print(p_name + ': ' + str(val))
        print()
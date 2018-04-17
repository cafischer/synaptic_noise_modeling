import numpy as np
from pykalman import KalmanFilter
from LDS_simple_example import create_simple_model


if __name__ == '__main__':
    Z, X, mu0, P0, A, b, Gamma, C, Sigma, n_timesteps, n_vars, n_obs = create_simple_model(seed=3)
    b = np.repeat(b.T, n_timesteps, 0)  # b: n_timesteps, n_vars

    kf = KalmanFilter(transition_matrices=A, observation_matrices=C, transition_covariance=None,
                      observation_covariance=None, transition_offsets=b, observation_offsets=np.zeros(np.shape(C)[0]),
                      initial_state_mean=None, initial_state_covariance=None, random_state=1,
                      n_dim_state=2, n_dim_obs=1)
    kf.em(X[:, :, 0], n_iter=1, em_vars=['transition_covariance', 'observation_covariance',
                               'initial_state_mean', 'initial_state_covariance'])

    mu0_est = kf.initial_state_mean
    P0_est = kf.initial_state_covariance
    Gamma_est = kf.observation_covariance
    Sigma_est = kf.transition_covariance

    print('Estimated parameters: ')
    print('m0: ', ['%.3f' % s for s in mu0_est])
    print('P0: ', [['%.3f' % t for t in t_row] for t_row in P0_est])
    #print('A: ', [['%.3f' % t for t in t_row] for t_row in A_est])
    #print('b: ', ['%.3f' % m for m in b_est])
    print('Sigma: ', [['%.3f' % t for t in t_row] for t_row in Sigma_est])
    #print('C: ', [['%.3f' % t for t in t_row] for t_row in C_est])
    print('Gamma: ', [['%.3f' % t for t in t_row] for t_row in Gamma_est])
    #print('log(p(x)): ', log_p_x)
    print()

    print('True parameters: ')
    print('m0: ', ['%.3f' % s for s in mu0])
    print('P0: ', [['%.3f' % t for t in t_row] for t_row in P0])
    #print('A: ', [['%.3f' % t for t in t_row] for t_row in A])
    #print('b: ', ['%.3f' % m for m in b])
    print('Gamma: ', [['%.3f' % t for t in t_row] for t_row in Gamma])
    #print('C: ', [['%.3f' % t for t in t_row] for t_row in C])
    print('Sigma: ', [['%.3f' % t for t in t_row] for t_row in Sigma])
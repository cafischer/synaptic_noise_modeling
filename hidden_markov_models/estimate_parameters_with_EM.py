from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import sys


# create data set
np.random.seed(1)
time = np.arange(0, 100, 0.1)

# parameters (theta)
trans_mat = np.array([[0.95, 0.15], [0.15, 0.85]])  # of hidden states
start_prob = np.array([0.0, 1.0])  # of hidden states
means = np.array([0.0, 3.0])  # of emission probability
covars = np.array([[0.01, 0.0], [0.0, 1.0]])  # of emission probability

# hidden states (Z)
Z = np.zeros(len(time)) 
Z[0] = 0 if np.random.rand() <= start_prob[0] else 1
for i, ts in enumerate(time[1:], 1):
    if Z[i-1] == 0:
        Z[i] = 0 if np.random.rand() <= trans_mat[0, 0] else 1
    else:
        Z[i] = 1 if np.random.rand() <= trans_mat[1, 1] else 0

# observed data (X)
X = np.zeros(len(time)) 
X[Z == 0] = covars[0, 0] * np.random.randn(sum(Z == 0)) + means[0]
X[Z == 1] = covars[1, 1] * np.random.randn(sum(Z == 1)) + means[1]

# # plot
# plt.figure()
# plt.plot(time, X, 'k', label='X')
# plt.plot(time, Z, 'b', label='Z')
# plt.ylabel('Calcium signal')
# plt.xlabel('Time')
# plt.legend()
# plt.show()

# In[3]:

def init_start_prob(n_states):
    """Column vector with entries in [0, 1] that must sum to 1."""
    start_prob_est = np.random.rand(n_states, 1)
    start_prob_est /= np.sum(start_prob_est, 0)
    assert np.isclose(np.sum(start_prob_est, 0), 1.)
    return start_prob_est


def init_trans_mat(n_states):
    """Matrix with entries in [0, 1]. Rows must sum to 1."""
    trans_mat_est = np.zeros((n_states, n_states))
    for i in range(n_states):
        trans_mat_est[i, :] = np.random.rand(n_states)
        trans_mat_est[i, :] /= np.sum(trans_mat_est[i, :])
    assert np.allclose(np.sum(trans_mat_est, 1), np.ones(n_states))
    return trans_mat_est


def is_positive_definite(x):
    return np.all(np.linalg.eigvals(x) > 0)


def init_means_and_covars(X, n_states):
    """Use k-means to get an estimate for the means and covariance matrix."""
    means_est = np.zeros((len(states)))
    covars_est = np.zeros((len(states), len(states)))
    kmeans = KMeans(n_clusters=n_states).fit(np.array([X]).T)
    labels = kmeans.labels_
    
    for i in range(n_states):
        means_est[i] = np.mean(X[labels==i])
        covars_est[i, i] = np.var(X[labels==i])

    assert is_positive_definite(covars_est)
    return means_est, covars_est


# In[12]:


def forward_backward_algorithm(X, start_prob_est, trans_mat_est, means_est, covars_est, n_timesteps, n_states):
    alpha = np.zeros((n_timesteps, n_states))
    beta = np.zeros((n_timesteps, n_states))
    
    for k in range(n_states):
        alpha[0, k] = start_prob_est[k] * norm.pdf(X[0], means_est[k], np.sqrt(covars_est[k][k]))
        beta[n_timesteps-1, k] = 1.0
    
    for n in range(1, n_timesteps):
        for k in range(n_states):
            alpha[n, k] = (norm.pdf(X[n], means_est[k], np.sqrt(covars_est[k][k])) 
                           * np.sum(alpha[n-1, :] * trans_mat_est[:, k]))
        
    for n in range(n_timesteps-2, -1, -1):
        for k in range(n_states):
            beta[n, k] = np.sum([beta[n+1, j] 
                                 * norm.pdf(X[n+1], means_est[j], np.sqrt(covars_est[j][j]))
                                 * trans_mat_est[k, j]
                                 for j in range(n_states)])
    
    return alpha, beta


def forward_backward_algorithm_scaled(X, start_prob_est, trans_mat_est, means_est, covars_est, n_timesteps, n_states):
    alpha_tmp = np.zeros((n_timesteps, n_states))
    alpha_head = np.zeros((n_timesteps, n_states))
    beta_tmp = np.zeros((n_timesteps, n_states))
    beta_head = np.zeros((n_timesteps, n_states))
    cs = np.zeros(n_timesteps)
    
    for k in range(n_states):
        alpha_tmp[0, k] = start_prob_est[k] * norm.pdf(X[0], means_est[k], np.sqrt(covars_est[k][k]))
        beta_tmp[n_timesteps-1, k] = 1.0
    cs[0] = np.sum(alpha_tmp[0, :])
    alpha_head[0, :] = alpha_tmp[0, :] / cs[0]
    beta_head[n_timesteps-1, :] = beta_tmp[n_timesteps-1, :]   # = 1.0
    
    for n in range(1, n_timesteps):
        for k in range(n_states):
            alpha_tmp[n, k] = (norm.pdf(X[n], means_est[k], np.sqrt(covars_est[k][k]))
                               * np.sum(alpha_head[n-1, :] * trans_mat_est[:, k]))
        cs[n] = np.sum(alpha_tmp[n, :])
        alpha_head[n, :] = alpha_tmp[n, :] / cs[n]
        
    for n in range(n_timesteps-2, -1, -1):
        for k in range(n_states):
            beta_tmp[n, k] = np.sum([beta_head[n+1, j]
                                     * norm.pdf(X[n+1], means_est[j], np.sqrt(covars_est[j][j]))
                                     * trans_mat_est[k, j]
                                 for j in range(n_states)])
        beta_head[n, :] = beta_tmp[n, :] / cs[n+1]
    return alpha_head, beta_head, cs


def do_expectation(X, alpha, beta, n_timesteps, n_states):
    p_x = max(np.sum(alpha[n_timesteps-1, :]), sys.float_info.epsilon)  # p_x should not be 0
    gamma = np.zeros((n_timesteps, n_states))
    xi = np.zeros((n_timesteps-1, n_states, n_states))
    
    for n in range(n_timesteps):
        gamma[n, :] = (alpha[n, :] * beta[n, :]) / p_x
    
    for n in range(n_timesteps-1):
        for k in range(n_states):
            for j in range(n_states):
                xi[n, k, j] = (alpha[n, k] * norm.pdf(X[n+1], means_est[j], np.sqrt(covars_est[j][j]))
                               * trans_mat_est[k, j] * beta[n+1, j]) / p_x
    return gamma, xi, np.log(p_x)


def do_expectation_scaled(X, alpha_head, beta_head, cs, n_timesteps, n_states):
    log_p_x = np.sum(np.log(cs))
    gamma = np.zeros((n_timesteps, n_states))
    xi = np.zeros((n_timesteps-1, n_states, n_states))
    
    for n in range(n_timesteps):
        gamma[n, :] = alpha_head[n, :] * beta_head[n, :]
    
    for n in range(n_timesteps-1):
        for k in range(n_states):
            for j in range(n_states):
                xi[n, k, j] = 1./cs[n+1] * (alpha_head[n, k] * norm.pdf(X[n+1], means_est[j], np.sqrt(covars_est[j][j]))
                                         * trans_mat_est[k, j] * beta_head[n+1, j])
    return gamma, xi, log_p_x


def do_maximization(gamma, xi, n_timesteps, n_states):
    start_prob_est = gamma[0, :] / np.sum(gamma[0, :])
    
    trans_mat_est = np.zeros((n_states, n_states))
    for j in range(n_states):
        for k in range(n_states):
            trans_mat_est[j, k] = np.sum(xi[:, j, k]) / np.sum([np.sum(xi[:, j, l]) for l in range(n_states)])

    means_est = np.zeros((n_states))
    covars_est = np.zeros((n_states, n_states))
    for k in range(n_states):
        means_est[k] = np.sum(gamma[:, k] * X) / np.sum(gamma[:, k])
        covars_est[k, k] = (np.sum([gamma[n, k] * (X[n] - means_est[k])**2 for n in range(n_timesteps)])
                            / np.sum(gamma[:, k]))

    assert np.isclose(np.sum(start_prob_est, 0), 1.)
    assert np.allclose(np.sum(trans_mat_est, 1), np.ones(n_states))
    if not is_positive_definite(covars_est):
        print covars_est
    assert is_positive_definite(covars_est)
        
    return start_prob_est, trans_mat_est, means_est, covars_est


# In[13]:

# EM algorithm
max_iter = 200  # TODO: 200
convergence = False
p_x_log_old = np.inf
tol = 1e-8
scaled = True

states = np.array([0, 1])
n_states = len(states)
n_timesteps = len(time)

# initialization of parameters
start_prob_est = init_start_prob(n_states)
trans_mat_est = init_trans_mat(n_states)
means_est, covars_est = init_means_and_covars(X, n_states)

print('start_prob_est: ', start_prob_est)   
print('trans_mat_est: ', trans_mat_est)  
print('means_est: ', means_est)  
print('covars_est', covars_est)

for iter in range(max_iter):
    
    # E-step
    if scaled:
        alpha_head, beta_head, cs = forward_backward_algorithm_scaled(X, start_prob_est, trans_mat_est, means_est,
                                                                      covars_est, n_timesteps, n_states)
        gamma, xi, log_p_x = do_expectation_scaled(X, alpha_head, beta_head, cs, n_timesteps, n_states)
    else:
        alpha, beta = forward_backward_algorithm(X, start_prob_est, trans_mat_est, means_est,
                                                 covars_est, n_timesteps, n_states)
        gamma, xi, log_p_x = do_expectation(X, alpha, beta, n_timesteps, n_states)

    # M-step
    start_prob_est, trans_mat_est, means_est, covars_est = do_maximization(gamma, xi, n_timesteps, n_states)
        
    print()
    print('start_prob_est: ', start_prob_est)
    print('trans_mat_est: ', trans_mat_est)  
    print('means_est: ', means_est)  
    print('covars_est', covars_est)

    # check convergence
    print 'log(p(x)): ', log_p_x
    if np.abs(log_p_x - p_x_log_old) <= tol:
        convergence = True
        break
    p_x_log_old = log_p_x
     
print
print 'converged: ', convergence
print 'iterations: ', iter
print 'startprob: ', ['%.3f' % s for s in start_prob_est]
print 'transmat: ', [['%.3f' % t for t in t_row] for t_row in trans_mat_est]
print 'means: ', ['%.3f' % m for m in means_est]
print 'covars: ', ['%.3f' % c for c in np.diag(covars_est)]
print
print 'true: '
print 'startprob: ', ['%.3f' % s for s in start_prob]
print 'transmat: ', [['%.3f' % t for t in t_row] for t_row in trans_mat]
print 'means: ', ['%.3f' % m for m in means]
print 'covars: ', ['%.3f' % c for c in np.diag(covars)]
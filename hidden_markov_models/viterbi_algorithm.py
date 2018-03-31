from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# def _log_multivariate_normal_density_diag(X, means, covars):
#     """Compute Gaussian log-density at X for a diagonal model."""
#     X_ = np.array([X]).T
#     means_ = np.array([np.array([m]) for m in means])
#     covars_ = np.array([[v] for v in np.diag(covars)])
#
#     n_samples, n_dim = X_.shape
#     lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars_), 1)
#                   + np.sum((means_ ** 2) / covars_, 1)
#                   - 2 * np.dot(X_, (means_ / covars_).T)
#                   + np.dot(X_ ** 2, (1.0 / covars_).T))
#     return lpr
# for checking:
# print _log_multivariate_normal_density_diag(X, means, covars)[0]
# [np.log(norm.pdf(X[0], means[k], np.sqrt(covars[k, k]))) for k in range(n_states)]


# create data set
time = np.arange(0, 100, 0.1)

# parameters (theta)
trans_mat = np.array([[0.55, 0.15], [0.15, 0.85]])  # of hidden states
start_prob = np.array([0.0, 1.0])  # of hidden states
means = np.array([0.0, 5.0])  # of emission probability
covars = np.array([[0.1, 0.0], [0.0, 1.0]])  # of emission probability

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


def pass_messages(X, start_prob, trans_mat, n_timesteps, n_states):
    w = np.zeros((n_timesteps, n_states))
    psi = np.zeros(n_timesteps, dtype=int)

    for k in range(n_states):
        w[0, k] = np.log(start_prob[k]) + np.log(norm.pdf(X[0], means[k], np.sqrt(covars[k, k])))

    for n in range(n_timesteps-1):
        for k in range(n_states):
            w[n+1, k] = np.log(norm.pdf(X[n+1], means[k], np.sqrt(covars[k, k]))) \
                        + np.max([np.log(trans_mat[j, k]) + w[n, j]
                                 for j in range(n_states)])

    # proper backtracking
    psi[n_timesteps-1] = np.argmax(w[n_timesteps-1, :])
    for n in range(n_timesteps - 2, -1, -1):
        psi[n] = np.argmax(np.log(trans_mat[:, psi[n+1]]) + w[n, :])
    return w, psi

# Viterbi
n_timesteps = len(X)
n_states = len(np.unique(Z))
w, psi = pass_messages(X, start_prob, trans_mat, n_timesteps, n_states)
Z_est = psi

plt.figure()
plt.plot(time, Z, 'b', label='Z', linewidth=2.0)
plt.plot(time, Z_est, '--r', label='Z est.')
plt.legend()
plt.show()
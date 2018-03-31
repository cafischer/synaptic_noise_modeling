from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")


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
X[Z==0] = covars[0, 0] * np.random.randn(sum(Z==0)) + means[0]
X[Z==1] = covars[1, 1] * np.random.randn(sum(Z==1)) + means[1]
X = np.array([X]).T

# pl.figure()
# pl.plot(Y_true, 'r')
# pl.plot(hidden_states_true, 'g')

model = hmm.GaussianHMM(n_components=2, n_iter=200, covariance_type="diag", algorithm='viterbi')

model.fit(X)
hidden_states = model.predict(X)

print 'converged: ', model.monitor_.converged
print 'startprob: ', ['%.3f' % s for s in model.startprob_]
print 'transmat: ', [['%.3f' % t for t in t_row] for t_row in model.transmat_]
print 'means: ', ['%.3f' % m for m in model.means_]
print 'covars: ', ['%.3f' % c for c in model.covars_[:, :, 0]]

print 'true: '
print 'startprob: ', ['%.3f' % s for s in start_prob]
print 'transmat: ', [['%.3f' % t for t in t_row] for t_row in trans_mat]
print 'means: ', ['%.3f' % m for m in means]
print 'covars: ', ['%.3f' % c for c in np.diag(covars)]

pl.figure()
pl.title('Hidden States')
pl.plot(Z, '--g', label='true', linewidth=2.5)
pl.plot(hidden_states, '--r', label='fitted')

# sample from fitted model
Y_new, hidden_states_new = model.sample(len(time))
pl.figure()
pl.title('New observations and hidden states from fitted')
pl.plot(Y_new, 'b', label='true')
pl.plot(hidden_states_new, '--r', label='fitted')
pl.show()
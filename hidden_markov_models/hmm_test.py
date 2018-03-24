from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

# model 1 with multinomial
# ns = 1000
# hidden_states_true = (np.floor(np.arange(ns) / 100) % 2) == 1
# Y_true = np.zeros(len(hidden_states_true))
# Y_true[hidden_states_true==0] = np.random.rand(sum(hidden_states_true==0)) < .4
# Y_true[hidden_states_true==1] = np.random.rand(sum(hidden_states_true==1)) < .01
# Y_true = [[int(val)] for val in Y_true]
#
# pl.figure()
# pl.plot(Y_true, 'b')
# pl.plot(hidden_states_true, '--g')

# model 2 with gaussian
ns = 1000
model = hmm.GaussianHMM(n_components=2, n_iter=200, covariance_type="diag", init_params="mcs", algorithm='viterbi')
model.startprob_ = np.array([[1.0], [0.0]])  # of hidden state 0 and 1
model.transmat_ = np.array([[.99, .01], [.01, .99]])  # between hidden state 0 and 1
model.means_ =  np.array([[0.0], [1.0]])  # of emission probability
model.covars_ = np.array([[0.0], [0.1]]) # of emission probability

Y_true, hidden_states_true = model.sample(ns)
# pl.figure()
# pl.plot(Y_true, 'r')
# pl.plot(hidden_states_true, 'g')



model = hmm.GaussianHMM(n_components=2, n_iter=200, covariance_type="diag", algorithm='viterbi')
#model = hmm.MultinomialHMM(n_components=2, n_iter=200, algorithm='viterbi')

model.fit(Y_true)
print model.monitor_.converged
hidden_states = model.predict(Y_true)

print model.startprob_
print model.transmat_
print model.means_
print model.covars_
#print model.emissionprob_

pl.figure()
pl.title('Hidden States')
pl.plot(hidden_states_true, '--g', label='true', linewidth=2.5)
pl.plot(hidden_states, '--r', label='fitted')

# sample from fitted model
Y_new, hidden_states_new = model.sample(ns)
pl.figure()
pl.title('New observations and hidden states from fitted')
pl.plot(Y_new, 'b', label='true')
pl.plot(hidden_states_new, '--r', label='fitted')
pl.show()


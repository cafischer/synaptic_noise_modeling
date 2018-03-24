function [ll g] = loglike(theta,est_d,X)
% Log-Likelihood function assuming p(n) ~ exp(-n/E(n)), E(n)=exp(k.x);
% Outoputs:
%         ll - log-likelihood.
%          g - gradiant.
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
% Michael Vidne, 2011.
TX = theta*X;
Y = [est_d;0]';
ll = sum( -TX - (Y.*(1./exp(TX))) );
g = -sum(X,2)  + X*(Y./(exp(TX)+eps))'; 

ll = -ll;
g = -g;



function [theta] = optim_theta(Y,Xd,theta)
% optimize theta given the estimate of the conductances.
% assuming p(n) ~ exp(-n/E(n)), E(n)=exp(k.x); n is the conductance jump.
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
%         optim_theta - optimize theta in the M-step.
% Michael Vidne, 2011.

%% initialize
epsi = 1e-6;
max_itr = 1e2;
lt = length(Y);
lambda = exp(theta*Xd);
fval = (theta*Xd)*Y-sum(lambda);
fval_diff = 1;
%%
itr = 0;
while ((fval_diff>epsi)&(itr<max_itr))
    itr = itr+1;
    lin_est = [theta];
    XTheta = theta*Xd;
    lambda = exp(XTheta);
    sd_lambda = spdiags(lambda',0,lt,lt);
    dFdTheta = Xd*Y - Xd*lambda';
    AThTh =  - Xd*sd_lambda*Xd';
    % direction of Newtwon step
    step_dir = (AThTh\dFdTheta)';
    step_size = 1;
    fval_new = fval-1;
    while (fval_new<fval & step_size>epsi)
        est_tmp = lin_est - step_size*step_dir;
        theta_tmp = est_tmp;
        XTheta_tmp = theta_tmp*Xd;
        lambda_tmp = exp(XTheta_tmp);
        %write down full objective function
        fval_new = (XTheta_tmp)*Y-sum(lambda_tmp);
        step_size = step_size*0.5;
    end
    fval_diff = abs(fval - fval_new);
    fval = fval_new;
    theta = theta_tmp;
end

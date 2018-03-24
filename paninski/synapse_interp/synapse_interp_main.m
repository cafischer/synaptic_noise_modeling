% function [] = synapse_interp_main()
% This function calls all other functions and sets all values.
% We first generate the toy data, then this code estimates the synaptic
% input g_e(t) and g_i(t) via particle filtering. Also, fits a generalized
% linear model with exponential observations to the synaptic inputs
% p(n) ~ exp(-n/E(n)), E(n)=exp(k.x); n is the conductance jump.
% We allow to generate the toy data with different probability
% distributions (Exp, Poisson), but all estimations are done assuming Exp!
% Code implementing of: "Inferring synaptic inputs given a noisy voltage
%                        trace via sequential Monte Carlo methods"
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
%         optim_theta - optimize theta in the M-step.
% Michael Vidne, 2011.

clear all;clc;
addpath('/home/cf/programs/octave_packages/jsonlab');
global fit model run stim;

% -----read the toy data-----
% TODO make_toy_data();
[model, stim, run] = load_toy_data();

% -----flags-----
% parametric or nonparametric EM.
run.non_parametric_EM = 1;
run.extra_plots = 1; % generate more plots with convergance


%-----initialize for EM-----
if run.non_parametric_EM
    % build the matrix of basis functions
    b = [run.tt(1):run.numBin*run.dt:run.tt(end) ]; % knots
    myI = eye(length(b));
    % matrix of basis functions. each row is a basis func'
    model.X_e = spline(b,myI,run.tt)*run.dt;
    model.X_i = model.X_e;
    theta_e = ones(size(b));
    theta_i = ones(size(b));
    fit.mus_e = exp(theta_e*model.X_e);
    fit.mus_i = exp(theta_i*model.X_i);
else
    k_e = 0; % can take any value
    k_i = 0; % can take any value
    
    theta_e = k_e;
    theta_i = k_i;
    
    fit.mus_e = exp(model.X_e*k_e*run.dt); % vector of mean of exc inputs
    fit.mus_i = exp(model.X_i*k_i*run.dt); % vector of mean of inh inputs
    
end
THETA_e = theta_e; THETA_i = theta_i;
MSEe(1) = 1; MSEi(1) = 1;
% -----start the EM iterations-----
for iEM = 1:run.N_emsteps
    disp(['EM iter : ',num2str(iEM)])
    % particle filter forward
    [Q,V,SV] = Gaussian_particle_filtering();
    
    % backward sweep
    disp('in backward sweep')
    bWs = zeros(run.lt,run.N);
    fWs = zeros(run.lt,run.N)+1/run.N;
    bWs(end,:) = 1/run.N;
    
    for(k=run.lt-1:-1:1) %recurse backwards
        for(j=1:run.N)
            e_err = Q(j,1,k+1)-model.a_e*Q(:,1,k);
            i_err = Q(j,2,k+1)-model.a_i*Q(:,2,k);
            p = zeros(1,run.N);
            f = find(e_err>0 & i_err>0);
            p(f) = exp(-e_err(f)/(fit.mus_e(k))-i_err(f)/(fit.mus_i(k)));
            bWs(k,:) = bWs(k,:)+bWs(k+1,j)*p;
        end
        bWs(k,:)=bWs(k,:).*fWs(k,:);
        bWs(k,:)=bWs(k,:)/sum(bWs(k,:));
    end
    
    % compute sufficient statistics for EM:
    est_v = sum(bWs'.*V)'; %posterior mean voltage
    est_e = sum(bWs.*squeeze(Q(:,1,:))',2); %posterior mean exc
    est_i = sum(bWs.*squeeze(Q(:,2,:))',2); %posterior mean inh
    
    est_de = est_e(2:end)-model.a_e*est_e(1:end-1); %posterior mean exc jumps: used in M step
    est_di = est_i(2:end)-model.a_i*est_i(1:end-1); %posterior mean inh jumps: used in M step
    
    std_v = sqrt(sum(bWs.*(V'-repmat(est_v,1,run.N)).^2,2)); % posterior var voltage
    std_e = sqrt(sum(bWs.*(squeeze(Q(:,1,:))'-repmat(est_e,1,run.N)).^2,2));% posterior var exc
    std_i = sqrt(sum(bWs.*(squeeze(Q(:,2,:))'-repmat(est_i,1,run.N)).^2,2));% posterior var inh
    
    % make sure none of the jump are negative.
    est_de(est_de<0) = 0;
    est_di(est_di<0) = 0;
    
    % M-step. optimize over theta the vector of parameters (can change tolerance etc in the function)
    [theta_e_new] = optim_theta([est_de;0],model.X_e,theta_e);
    [theta_i_new] = optim_theta([est_di;0],model.X_i,theta_i);
    
    fit.mus_e = exp(theta_e_new*model.X_e); % new mean exc input
    fit.mus_i = exp(theta_i_new*model.X_i); % new mean inh input
    
    % record the new parametrs
    THETA_e = [THETA_e;theta_e_new];
    THETA_i = [THETA_i;theta_i_new];
    % record the distance from the true conductances
    MSEe(iEM+1) = sum((est_e(15:end-15) - model.gs_e(15:end-15)).^2)/sum(model.gs_e(15:end-15).^2);
    MSEi(iEM+1) = sum((est_i(15:end-15) - model.gs_i(15:end-15)).^2)/sum(model.gs_i(15:end-15).^2);
end

%
display_results;
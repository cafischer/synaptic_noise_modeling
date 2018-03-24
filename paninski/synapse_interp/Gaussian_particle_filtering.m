function [Q,V,StdW] = Gaussian_particle_filtering()
% This code estimates the synaptic input g_e(t) and g_i(t) via
% Gaussian particle filtering. As discribed in Sec 2.2 of:
% "Inferring synaptic inputs given a noisy voltage
% trace via sequential Monte Carlo methods"
% Outputs:
%        Q - position of N particles in g_e and g_i space over time.
%        V - Voltages.
%        StdW - std.
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
% Michael Vidne, 2011.
global fit model run;
disp('in Gaussian_particle_filtering')
obs = model.obsNoiseyV;
N = run.N;

dSigma(1) = model.sdtv; % voltage dynamic noise

mus_e = fit.mus_e; %vector of mean of exc inputs
mus_i = fit.mus_i ; %vector of mean of inh inputs

oSigma = sqrt(model.varObsNoise); % observation noise

lt = length(obs);
Q = zeros(N,2,lt);
V = zeros(N,lt);
Wi = zeros(N,lt);
V(:,1) = model.v_leak;
Wi(:,1) = 1;
Q(:,:,1) = rand(N,2) + ones(N,1)*[0  0];
model.prior_e = zeros(size(run.tt)); %the vector of inferred exc conductances
model.prior_i = zeros(size(run.tt)); %the vector of inferred inh conductances
for t = 2:lt
    dSigma(2) = mus_e(t);
    dSigma(3) = mus_i(t);
    
    [q1,v1,Wi1] = kernel(Q(:,:,t-1),V(:,t-1),Wi(:,t-1),obs(t),dSigma,oSigma,run,model);
    
    model.prior_e(t) = model.prior_e(t-1)*model.a_e +  dSigma(2);
    model.prior_i(t) = model.prior_i(t-1)*model.a_i +  dSigma(3);
    
    V(:,t) = v1;
    Wi(:,t) = Wi1;
    Q(:,:,t) = q1;
end

StdW = sqrt(Wi);


function [q1,V1,Wi1]  =  kernel(q0,v0,Wi0,obs,dSigma,oSigma,run,model)

N = length(q0);

% linear coefficients 
ai = 1-run.dt*(model.g_leak + q0(:,1) + q0(:,2) );
bi = run.dt*(model.g_leak*model.v_leak + q0(:,1)*model.v_e + q0(:,2)*model.v_i );

%  calculate analyticaly conditional distribution over the indices (Eq. 10)
mu = (ai.*v0+bi);
Sigma = sqrt(ai.^2.*Wi0+dSigma(1)^2+oSigma^2);
logSqrtDetSigma = log(Sigma);
xRinv = (obs - mu)./Sigma;
quadform = xRinv.^2;
lp = -0.5*quadform - logSqrtDetSigma - log(2*pi)/2;
lambda = exp(lp - max(lp)); % stabilize
lambda = lambda/sum(lambda);% normelize

% stratified sample the aux ver
r = rand(N,1)/N + (0:N-1)'/N;
[dummy,auxVer] = histc(r,[0; cumsum(lambda)]);
q0 = q0(auxVer,:);
v0 = v0(auxVer);
Wi0 = Wi0(auxVer,:);
ai = ai(auxVer);
bi = bi(auxVer);

% sample from the prior
N_e = -dSigma(2) .* log(rand(N,1));
N_i = -dSigma(3) .* log(rand(N,1));
% propegate the conductance forward
q1(:,1) = q0(:,1).*model.a_e + N_e;
q1(:,2) = q0(:,2).*model.a_i + N_i;

% obtain mean and variance of the voltage (Eq 12 & 13)
Wi1 = 1./(   1./(ai.^2.*Wi0+dSigma(1)^2)    +   1./(oSigma^2)  );
V1 = Wi1.*((ai.*v0+bi).*(1./(ai.^2.*Wi0+dSigma(1)^2)) + obs./(oSigma^2));
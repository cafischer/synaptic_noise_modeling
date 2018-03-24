function [] = make_toy_data()
% Generate the presynaptic mean stimulus. Either cos wave or random O-U
% process. Then, evolve Eqs 1-3 forward, and generate observations.
% The synaptic input can be out of an Exp or Poisson distributions.
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
% Michael Vidne, 2011.
%global run model fit stim

% -----simulation parameters-----
run.random_stim = 1;
run.dt = .002; % sampling rate (sec)
run.T = 1; % total simulation time (sec)
run.N = 100; % number of particles
run.N_emsteps = 10; %number of EM iterations
run.numBin = 10; % width of each basis function in number of dts

run.tt = (run.dt:run.dt:run.T)';
run.lt = length(run.tt);

% -----stimulation parameters-----
stim.fr = 5; % stimulation frequncy
stim.delay = 0.01; % delay b/w exc and inh (sec)
stim.excAmp = 100; % amplitude of exc stimulation
stim.bias = -40; % bias of stimulation

% -----model parameters-----
model.tranProbName = 'exp';  %'exp'; % could also use 'poiss', but the estimation is assuming Exp
model.tau_e = .003; %exc time constant (sec)
model.tau_i = .01; %inh time constant (sec)

model.v_e = 0; %exc reversal potential (mV)
model.v_i = -75; %inh reversal potential (mV)

model.g_leak = 80; % 1/membrane time constant (sec)
model.v_leak = -60; %  leak potential (mV)

model.v_noise = 1e-6; % voltage evolution variance
model.varObsNoise = 1e-6; % observation noise variance

model.k_e_true = 7; %exc weights; this parameter is estimated in the EM step
model.k_i_true = 7; %inh weights; this parameter is estimated in the EM step


% ----- initializations----- 
rand('state',100*pi^2); % seed the random number generator
model.vs = zeros(size(run.tt)); %the vector of observed voltages
model.gs_e = zeros(size(run.tt)); %the vector of inferred exc conductances
model.gs_i = zeros(size(run.tt)); %the vector of inferred inh conductances
model.vs(1) = model.v_leak; % Start the voltage at leak voltage
model.gs_e(1) = 0; %true excitatory conductance.
model.gs_i(1) = 0; %true inhibitory conductance.

% parameters
model.sdtv = sqrt(run.dt)*model.v_noise;
model.a_e = exp(-run.dt/model.tau_e); % decay factor in one time step. TODO: the exp does not make sense for me, also does not occur in Paninskis equations
model.a_i = exp(-run.dt/model.tau_i); % decay factor in one time step.


% ----- generate presynaptic inputs -----
X_e = (-stim.excAmp*cos(stim.fr*2*pi*run.tt)+stim.bias); % sin(2*2*pi*run.tt)];
if run.random_stim
    disp('in make_toy_data: using random presynaptic stimulus')
    % presynaptic stimulus corresponding to exc
    noise = randn(size(X_e));  % Normalized white Gaussian noise
    X_e = -30+20*filter(1,[1 -0.8 ],noise); % mean presynaptic input is an O-U process
    model.true_mu_e = abs(filter(1,[1 -0.9 ],noise));
    % presynaptic stimulus corresponding to inh
    noise = randn(size(X_e));  % Normalized white Gaussian noise
    X_i = -40+20*filter(1,[1 -0.95 ],noise);
    model.true_mu_i = abs(filter(1,[1 -0.9 ],noise));
else
    disp('in make_toy_data: using cos wave presynaptic stimulus')
    % presynaptic stimulus corresponding to exc
    X_e = (-stim.excAmp*cos(stim.fr*2*pi*run.tt)+stim.bias); % cos wave stimulus
    model.true_mu_e = exp(X_e*model.k_e_true*run.dt); % vector of mean of exc inputs

    delay = ceil(stim.delay/run.dt); % delay between exc and inh
    % presynaptic stimulus corresponding to inh
    X_i = [X_e(1,1)*ones(delay,1); X_e(1:end-delay,1)];
    model.true_mu_i = exp(X_i*model.k_i_true*run.dt); % vector of mean of inh inputs
end

for k = 2:run.lt
    oldV = model.vs(k-1);
    oldE = model.gs_e(k-1);
    oldI = model.gs_i(k-1);
    %     switch stim.noise
    switch model.tranProbName
        case 'exp'
            N_e = -log(rand)*model.true_mu_e(k);
            N_i = -log(rand)*model.true_mu_i(k);
        case 'poiss'
            N_e = random('poiss', model.true_mu_e(k));
            N_i = random('poiss', model.true_mu_i(k));
        otherwise
            disp('in make_toy_data: unrecognized probability.')
    end
    
    % propagate the values forward (Eqs 1-3 in the paper )
    model.gs_e(k) = oldE*model.a_e + N_e;
    model.gs_i(k) = oldI*model.a_i + N_i;
    model.vs(k) =  oldV + ...
        run.dt*(model.g_leak*(model.v_leak-oldV)+ oldE*(model.v_e-oldV)...
        + oldI.*(model.v_i-oldV));
    % noise to the expected voltage
    model.vs(k) = model.vs(k) + model.sdtv*randn;
end
model.X_e = X_e'*run.dt;
model.X_i = X_i'*run.dt;

% add noise to get the actual observed voltage
model.obsNoiseyV = model.vs + randn(size(model.vs))*sqrt(model.varObsNoise);

% save simulated traces
folder_parent = 'data';
folder_child = 'example_paninski';
save_dir = strcat('.', filesep, folder_parent, filesep, folder_child);
mkdir(folder_parent);
mkdir(folder_parent, folder_child);
cd(save_dir);

addpath('/home/cf/programs/octave_packages/jsonlab');
savejson('', model, 'model.json');
savejson('', stim, 'stim.json');
savejson('', run, 'run.json');

cd('../../')


%{
v = model.vs;
v_with_obs_noise = model.obsNoiseyV;
g_e = model.gs_e;
g_i = model.gs_i;
dlmwrite(strcat(save_dir, '/v.txt'), v, '\n'); 
dlmwrite(strcat(save_dir, '/v_with_obs_noise.txt'), v_with_obs_noise, '\n'); 
dlmwrite(strcat(save_dir, '/g_e.txt'), g_e, '\n');
dlmwrite(strcat(save_dir, '/g_i.txt'), g_i, '\n');
%}






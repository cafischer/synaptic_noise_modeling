function [model, stim, run] = load_toy_data()
% Load toy data that was previously simulated
% Caroline Fischer, 2018

addpath('/home/cf/programs/octave_packages/jsonlab');

folder_parent = 'data';
folder_child = 'model_pas';
save_dir = strcat('.', filesep, folder_parent, filesep, folder_child);

model = loadjson(strcat(save_dir, filesep, 'model.json'));
% should contain:
%{
model.tranProbName = 'exp'; % could also use 'poiss', but the estimation is assuming Exp
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

model.vs = dlmread(strcat(save_dir, '/v.txt'), '\n'); %the vector of observed voltages
model.obsNoiseyV = dlmread(strcat(save_dir, '/v_with_obs_noise.txt'), '\n'); 
model.gs_e = dlmread(strcat(save_dir, '/g_e.txt'), '\n'); %the vector of inferred exc conductances
model.gs_i = dlmread(strcat(save_dir, '/g_i.txt'), '\n'); %the vector of inferred inh conductances

model.sdtv = sqrt(run.dt)*model.v_noise;
model.a_e = exp(-run.dt/model.tau_e); % decay factor in one time step.
model.a_i = exp(-run.dt/model.tau_i); % decay factor in one time step.
%}


% -----stimulation parameters-----
stim = loadjson(strcat(save_dir, filesep, 'stim.json'));
%{
stim.fr = 5; % stimulation frequncy
stim.delay = 0.01; % delay b/w exc and inh (sec)
stim.excAmp = 100; % amplitude of exc stimulation
stim.bias = -40; % bias of stimulation
%}

% -----simulation parameters-----
run = loadjson(strcat(save_dir, filesep, 'run.json'));
%{
run.dt = .002; % sampling rate (sec)
run.T = 1; % total simulation time (sec)
run.N = 100; % number of particles
run.N_emsteps = 10; %number of EM iterations
run.numBin = 10; % width of each basis function in number of dts

run.tt = (run.dt:run.dt:run.T)';
run.lt = length(run.tt);
%}

%cd('-')

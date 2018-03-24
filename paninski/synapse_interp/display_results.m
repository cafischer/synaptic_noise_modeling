% generate the main figure of:
% "Inferring synaptic inputs given a noisy voltage
% trace via sequential Monte Carlo methods"
% See also:
%         make_toy_data - generate presynaptic inputs and simulate Eq1-3.
%         Gaussian_particle_filtering - forward estimation of the
%         state-space. as discrived in Sec 2.2
%         loglike - loglikelihood assuming exp distribution and exp
%         nonlinearity.
%         display_results - make the main figure from the paper.
% Michael Vidne, 2011.
rand('state',pi*rand);
display('in display_results: plotting...');

figNum = ceil(rem(now,1)*1e6);
fig=figure(figNum);
set(fig,'Position',[100 10 900 1000]);

subs=7; sub=0;

% plot voltage
sub=sub+1; subplot(subs,1,sub);
hold all
p_std_v = patch([run.tt; flipud(run.tt)], [est_v+std_v; flipud(est_v-std_v)]...
    ,'r','facealpha',0.5,'edgecolor','none');
p_true_v = plot(run.tt,model.vs,'k-','LineWidth',1.5);
p_obs_v = plot(run.tt,model.obsNoiseyV,'b.','LineWidth',1.5);
p_mean_v = plot(run.tt,est_v,'r--','LineWidth',1.5);
c={'True V' 'Observed V' 'Estimated V'};
order=[1 2 3];
legend([p_true_v,p_obs_v,p_mean_v,],c{order});legend boxoff
axis tight;
ylabel('V (mV)','fontsize',14);
set(gca,'XTickLabel',[]);

% plot exc conductance (g_e and est' g_e)
sub=sub+1; subplot(subs,1,sub); hold on;
p_std_e = patch([run.tt; flipud(run.tt)], [est_e+std_e; flipud(est_e-std_e)]...
    ,'r','facealpha',0.5,'edgecolor','none');
p_true_e = plot(run.tt,model.gs_e,'k-','LineWidth',1.5);
p_est_e = plot(run.tt,est_e,'r--','LineWidth',1.5);
c={'std ','True', 'Estimated'}; % legend list
order=[2 3];
legend([p_true_e,p_est_e],c{order}); legend boxoff
axis tight; ylabel('g_e','fontsize',14);
set(gca,'XTickLabel',[]);


% plot inh conductance (g_i and est' g_i)
sub=sub+1; subplot(subs,1,sub);
hold on;
p_std_i = patch([run.tt; flipud(run.tt)], [est_i+std_i; flipud(est_i-std_i)]...
    ,'r','facealpha',0.5,'edgecolor','none');
p_true_i = plot(run.tt,model.gs_i,'k-','LineWidth',1.5);
p_est_i = plot(run.tt,est_i,'r--','LineWidth',1.5);
order=[2 3];
axis tight; ax=axis; axis([ax(1:2) 0 ax(4)]);
ylabel('g_i','fontsize',14);
set(gca,'XTickLabel',[]);

% plot presynaptic inputs (N_e and N_i)
sub=sub+1; subplot(subs,1,sub);
tth=run.tt(1:end-1)+run.dt/2;
lin=plot(tth,model.gs_e(2:end)-model.a_e*model.gs_e(1:end-1),'k-','LineWidth',1.5);
hold on;
plot(tth,est_de,'r-','LineWidth',1.5);
axis tight; ax=axis; axis([ax(1:2) 0 ax(4)]);
ylabel('N_e','fontsize',14);
set(gca,'XTickLabel',[]);

sub=sub+1; subplot(subs,1,sub);
lin=plot(tth,model.gs_i(2:end)-model.a_i*model.gs_i(1:end-1),'k-','LineWidth',1.5); hold on;
plot(tth,est_di,'r-','LineWidth',1.5);
axis tight; ylabel('N_i','fontsize',14)
xlabel('time');

% plot presynaptic mean inputs (lambda_e and lambda_i)
sub=sub+1; subplot(subs,1,sub);
lin=plot(run.tt,fit.mus_e,'r-','LineWidth',1.5); hold on;
lin=plot(run.tt,model.true_mu_e,'k-','LineWidth',1.5); hold on;
axis tight; ylabel('\lambda_e','fontsize',14);
set(gca,'XTickLabel',[]);

sub=sub+1; subplot(subs,1,sub);
lin=plot(run.tt,fit.mus_i,'r-','LineWidth',1.5); hold on;
lin=plot(run.tt,model.true_mu_i,'k-','LineWidth',1.5); hold on;
axis tight; ylabel('\lambda_i','fontsize',14)
xlabel('time [sec]');

if run.extra_plots
    s = size(THETA_e);
    figNum = ceil(rem(now,1)*1e6);
    fig=figure(figNum);
    hold all
    for i = 1:s(1)
        plot(run.tt,exp(THETA_e(i,:)*model.X_e),'color',[0.8-i/(2*s(1)) 0.1 0.9])
    end
    xlabel('time [sec]');
    ylabel('\lambda_e','fontsize',14);
    
    figNum = ceil(rem(now,1)*1e6);
    fig=figure(figNum);
    hold all
    plot([0:run.N_emsteps],MSEe); hold all
    plot([0:run.N_emsteps],MSEi);
    xlabel('EM iteration');
    ylabel('MSE','fontsize',14);
end
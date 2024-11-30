#bgev_fit_demo

#setsid nohup octave --eval "bgev_fit_demo"  >& ~/outo.txt &

pkg load statistics

switch hostname
case {'NKL'}
  doc_dir = '/home/nir/Documents/';
  era_data_dir = [doc_dir 'data/era5/'];
  direc_out = [doc_dir 'data/misc/']; 
case {'crest-cluster-node-6'}
  doc_dir = '/crest/faculty/nkrakauer/documents/';
  era_data_dir = direc_out = '/crest/faculty/nkrakauer/other/misc/irrig/';
endswitch
station_data_dir = [doc_dir 'data/misc/'];
program_dir = [doc_dir 'programs/'];

load ([program_dir 'bgev_test_data.mat'])
t = t(:);

run_label = '_grid';

#with default settings, this script runs for the ERA5 sampled grid cells

#region means rather than single grid cells
region_use = false;
if region_use
  out_file = [era_data_dir 'era5_T_max_100_regions.mat'];
  load (out_file)
  data = region_vals(region_level==4, :)';
  run_label = '_region';
endif

#station (GHCN-D) rather than reanalysis data
#cf. ghcn_tmax.m
station_use = false;
if station_use
  out_file = [station_data_dir 'ghcn_tmax.mat'];
  load (out_file)
  data = double(subset_vals) / 10; #convert to degrees C
  T = load ([station_data_dir 'Land_and_Ocean_summary.txt']); #longer global T time series from Berkeley Earth (starts at 1850)
  t = T(:, 2);
  run_label = '_station';
endif

file_out = [direc_out 'bgev_fits' run_label '.mat'];


[nt, nl] = size (data);

#check ability to fit annual maximum temperature data and make year-ahead forecasts
#of GEV and the normal distribution
#(both with and without trend)
#plus optionally the bGEV with negative shape parameter 

nt_start = 30;
nt_fcst = nt - nt_start + 1;
as = []; #0.9;
bs = as - 0.01; #as - 0.01;
s = 5; #beta distribution shape parameters
nvals = numel(as); #number of different a, b values to try

ii_max = 15;

n_draws = 1000; #draws from GEV parameter likelihood for estimating posterior predictive distribution

fcst_q = fcst_nll = fcst_mean = nan (nt_fcst-1, nl, nvals+4);
shape_params = nan (nt_fcst-1, nl, nvals+2);

rand ("state", pi) #initialize random number generator for reproducibility

#warning('off', 'all', 'local')

for i = 1:(nt_fcst-1)
  disp(i)
  
  for j = 1:nl    
    y = data(1:(nt_start+i-1), j);
    yn = data(nt_start+i, j);
    if isfinite(yn) && (sum(isfinite(y)) >= nt_start) 
      x = t(1:(nt_start+i-1));
      xn = t(nt_start+i);
      ii = isfinite(y);
      y = y(ii);
      x = x(ii);
      nd = numel (x);
      xm = mean(x);
      x -= xm;
      xn -= xm;
  
      ##GEV (with trend)
      pos = 1;
      pos_shape = 1;
      [gev_params, FIM, warnflag] = gevt_fit_grad_regularized (y, x);
      ii = 0;
      while warnflag && ii < ii_max
      #try to get better parameter values by increasing regularization
        l = 3 ^ ii;
        ii++;
        [gev_params, FIM, warnflag] = gevt_fit_grad_regularized (y, x, [], l);
      endwhile
      params = gev_params;   
      #now include parameter uncertainty in forecast NLL and q
      Co = inv(FIM);
      #scale to express derivatives in terms of log(sigma) rather than sigma
      sigma = params(2);
      FIM(2, :) .*= sigma;
      FIM(:, 2) .*= sigma;
      C = inv(FIM);
      #draws from candidate distribution (Laplace approximation)
      rescale_params = params;
      rescale_params(2) = log(params(2));
      try
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
      catch
        disp(['i = ' num2str(i) ', j = ' num2str(j) ': GEV fit likely did not converge, continuing with generic sampling'])
        return
        C = [0.5, 0.5, 3, 1] .^ 2;
        rescale_params = gev_params_init;
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
        Co = C;
        Co(2) .*= (sigma.^2);
      end_try_catch
      #get their likelihoods under the candidate distribution and the actual GEV
      params_draw = rescale_params_draw;
      params_draw(:, 2) = exp(rescale_params_draw(:, 2));
      p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);
      params_nll = sum (gevnll (y, params(1), params(2), params(3) + x*params(4)));
      draws_nll = zeros (n_draws, 1);
      for d = 1:n_draws
        draws_nll(d) = sum (gevnll (y, params_draw(d, 1), params_draw(d, 2), params_draw(d, 3) + x*params_draw(d, 4)));
      endfor
      w = params_nll - draws_nll + p_proposal;
      w = exp(w - max(w));
      ii = (w > 0);
      w /= sum(w(ii));
      fcst_nll(i, j, pos) =  -log( sum (w(ii) .* gevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4))));
      if !isfinite(fcst_nll(i, j, pos))
        #increase variance of shape parameter to try and get some draws where the observation has positive probability      
        disp([num2str(i) ', ' num2str(j) ' GEV zero-likelihood observation'])
        C(1, 1) = (params(1)/2) .^ 2;
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
        params_draw = rescale_params_draw;
        params_draw(:, 2) = exp(rescale_params_draw(:, 2));
        p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);
        params_nll = sum (gevnll (y, params(1), params(2), params(3) + x*params(4)));
        draws_nll = zeros (n_draws, 1);
        for d = 1:n_draws
          draws_nll(d) = sum (gevnll (y, params_draw(d, 1), params_draw(d, 2), params_draw(d, 3) + x*params_draw(d, 4)));
        endfor
        w = params_nll - draws_nll + p_proposal;
        w = exp(w - max(w));
        ii = (w > 0);
        w /= sum(w(ii));
        fcst_nll(i, j, pos) =  -log( sum (w(ii) .* gevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4))));        
        if !isfinite(fcst_nll(i, j, pos))
          disp([num2str(i) ', ' num2str(j) ' GEV zero-likelihood observation even after resampling'])        
        endif
      endif
      tmp = gevcdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4));
      if min(tmp) == max(tmp)
        fcst_q(i, j, pos) = tmp(1);
      else  
        fcst_q(i, j, pos) = sum (w(ii) .* tmp);
      endif
      if fcst_q(i, j, pos) > 1
        disp([num2str(i) ', ' num2str(j) ' GEV CDF > 1 returned!'])
        return
      endif
      shape_params(i, j, pos_shape) = gev_params(1);
      mu = params_draw(ii, 3) + xn*params_draw(ii, 4);
      sigma = params_draw(ii, 2);
      k = params_draw(ii, 1);
      gev_means = mu + sigma .* (gamma(1 - k) - 1) ./ k; #mean (expected value) of each forecast GEV distribution drawn 
      fcst_mean(i, j, pos) = sum (w(ii) .* gev_means);

      ##GEV (without trend)
      pos = nvals+3;
      pos_shape = nvals+2;
      [gev_params, FIM, warnflag] = gev_fit_grad_regularized (y);
      ii = 0;
      while warnflag && ii < ii_max
      #try to get better parameter values by increasing regularization
        l = 3 ^ ii;
        ii++;
        [gev_params, FIM, warnflag] = gev_fit_grad_regularized (y, [], l);
      endwhile
      params = gev_params;   
      #now include parameter uncertainty in forecast NLL and q
      Co = inv(FIM);
      #scale to express derivatives in terms of log(sigma) rather than sigma
      sigma = params(2);
      FIM(2, :) .*= sigma;
      FIM(:, 2) .*= sigma;
      C = inv(FIM);
      #draws from candidate distribution (Laplace approximation)
      rescale_params = params;
      rescale_params(2) = log(params(2));
      try
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
      catch
        warning("bgev_fit_demo:bad_fit", ['i = ' num2str(i) ', j = ' num2str(j) ': GEV fit likely did not converge, continuing with generic sampling'])
        C = [0.5, 0.5, 3] .^ 2;
        rescale_params = gev_params;
        rescale_params(2) = log (rescale_params(2));        
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
        Co = C;
        Co(2) .*= (sigma.^2);
      end_try_catch
      #get their likelihoods under the candidate distribution and the actual GEV
      params_draw = rescale_params_draw;
      params_draw(:, 2) = exp(rescale_params_draw(:, 2));
      p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);
      params_nll = sum (gevnll (y, params(1), params(2), params(3)));
      draws_nll = zeros (n_draws, 1);
      for d = 1:n_draws
        draws_nll(d) = sum (gevnll (y, params_draw(d, 1), params_draw(d, 2), params_draw(d, 3)));
      endfor
      w = params_nll - draws_nll + p_proposal;
      w = exp(w - max(w));
      ii = (w > 0);
      w /= sum(w(ii));
      fcst_nll(i, j, pos) =  -log( sum (w(ii) .* gevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3))));
      if !isfinite(fcst_nll(i, j, pos))
        #increase variance of shape parameter to try and get some draws where the observation has positive probability      
        disp([num2str(i) ', ' num2str(j) ' GEV zero-likelihood observation'])
        C(1, 1) = (params(1)/2) .^ 2;
        rescale_params_draw = mvnrnd (rescale_params, C, n_draws);
        params_draw = rescale_params_draw;
        params_draw(:, 2) = exp(rescale_params_draw(:, 2));
        p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);
        params_nll = sum (gevnll (y, params(1), params(2), params(3)));
        draws_nll = zeros (n_draws, 1);
        for d = 1:n_draws
          draws_nll(d) = sum (gevnll (y, params_draw(d, 1), params_draw(d, 2), params_draw(d, 3)));
        endfor
        w = params_nll - draws_nll + p_proposal;
        w = exp(w - max(w));
        ii = (w > 0);
        w /= sum(w(ii));
        fcst_nll(i, j, pos) =  -log( sum (w(ii) .* gevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3))));        
        if !isfinite(fcst_nll(i, j, pos))
          disp([num2str(i) ', ' num2str(j) ' GEV zero-likelihood observation even after resampling'])        
        endif
      endif
      tmp = gevcdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3));
      if min(tmp) == max(tmp)
        fcst_q(i, j, pos) = tmp(1);
      else  
        fcst_q(i, j, pos) = sum (w(ii) .* tmp);
      endif
      if fcst_q(i, j, pos) > 1
        disp([num2str(i) ', ' num2str(j) ' GEV CDF > 1 returned!'])
        return
      endif
      shape_params(i, j, pos_shape) = gev_params(1);
      mu = params_draw(ii, 3);
      sigma = params_draw(ii, 2);
      k = params_draw(ii, 1);
      gev_means = mu + sigma .* (gamma(1 - k) - 1) ./ k; #mean (expected value) of each forecast GEV distribution drawn 
      fcst_mean(i, j, pos) = sum (w(ii) .* gev_means);
      
      #bGEV
      if nvals > 0
        if gev_params(1) == 0
          fcst_nll(i, j, 2:end) = fcst_nll(i, j, 1); 
          fcst_q(i, j, 2:end) = fcst_q(i, j, 1);
        elseif gev_params(1) < 0
          for k = 1:nvals
            p_a = as(k);
            p_b = bs(k);  
            bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4), p_a, p_b, s));
            bgev_params = fminunc (bgev_nll, gev_params);
            rescale_params = bgev_params;
            rescale_params_draw = mvnrnd (bgev_params, C, n_draws);
            params = rescale_params;
            params(2) = exp (rescale_params(2));
            params_draw = rescale_params_draw;
            params_draw(:, 2) = exp(rescale_params_draw(:, 2));            
            p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);
            params_nll = bgev_nll(rescale_params);
            draws_nll = zeros (n_draws, 1);
            for d = 1:n_draws
              draws_nll(d) = bgev_nll(params_draw(d, :));
            endfor
            w = params_nll - draws_nll + p_proposal;
            w = exp(w - max(w));
            ii = (w > 0);
            w /= sum(w(ii));
            fcst_nll(i, j, k+1) =  -log( sum (w(ii) .* bgevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4), p_a, p_b, s)));
            fcst_q(i, j, k+1) = sum (w(ii) .* bgevcdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4), p_a, p_b, s));            
            shape_params(i, j, k+1) = bgev_params(1);
            if !isfinite(fcst_nll(i, j, k+1))
              disp('surprise, stopping')
              return
            endif
          endfor 
          elseif gev_params(1) > 0 #use default hyperparameters from bgevnll
           bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
           bgev_params = fminunc (bgev_nll, gev_params);
            rescale_params = bgev_params;
            rescale_params_draw = mvnrnd (bgev_params, C, n_draws);
            params = rescale_params;
            params(2) = exp (rescale_params(2));
            params_draw = rescale_params_draw;
            params_draw(:, 2) = exp(rescale_params_draw(:, 2)); 
            p_proposal = mvnnll(rescale_params_draw, rescale_params, C) ./ params_draw(:, 2);           
            params_nll = bgev_nll(rescale_params);
            draws_nll = zeros (n_draws, 1);
            for d = 1:n_draws
              draws_nll(d) = bgev_nll(params_draw(d, :));
            endfor
            w = params_nll - draws_nll + p_proposal;
            w = exp(w - max(w));
            ii = (w > 0);
            w /= sum(w(ii));
            fcst_nll(i, j, 2:(nvals+1)) =  -log( sum (w(ii) .* bgevpdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4))));
            fcst_q(i, j, 2:(nvals+1)) = sum (w(ii) .* bgevcdf (yn, params_draw(ii, 1), params_draw(ii, 2), params_draw(ii, 3) + xn*params_draw(ii, 4)));          
            shape_params(i, j, 2:end) = bgev_params(1);
            if !isfinite(fcst_nll(i, j, 2))
             disp('surprise, stopping') #this should not happen with the bGEV
            return
           endif  
         endif
       endif
       
       #normal distribution (with trend) (t distribution posterior predictive)
       pos = nvals+2;
       [mu, sigma, nu] = regress_params ([ones(nd, 1) x], y, [1 xn]);
       fcst_nll(i, j, pos) = -tpdf_log((yn - mu)/sigma, nu) + log(sigma);
       fcst_q(i, j, pos) = tcdf ((yn - mu)/sigma, nu);
       fcst_mean(i, j, pos) = mu;
       
       #normal distribution (without trend)
       pos = nvals+4;       
       [mu, sigma, nu] = regress_params ([ones(nd, 1)], y, [1]);
       fcst_nll(i, j, pos) = -tpdf_log((yn - mu)/sigma, nu) + log(sigma);
       fcst_q(i, j, pos) = tcdf ((yn - mu)/sigma, nu);
       fcst_mean(i, j, pos) = mu;
       
     endif
  endfor
endfor

save (file_out, 'fcst_q', 'fcst_nll', 'fcst_mean', 'shape_params', 'nt_start', 'as', 'bs', 's', 'nt_start', 'nt_fcst', 'data')
disp ('Finished')
disp (file_out)

return

#forecast bias
#mean(data((nt_start+1):end, :)(:) - fcst_mean(:, :, 2)(:), "omitnan")

#forecast RMSE
#sqrt(mean((data((nt_start+1):end, :)(:) - fcst_mean(:, :, 2)(:)) .^ 2, "omitnan"))

#test posterior predictive PDF for difference from uniform distribution, in particular for the most dramatic exceedances
RP = 1 ./ (1 - fcst_q);
mm = sort (RP(:, :, nvals+2)(:), "descend");
nn = numel(mm);
rr = sort(1 ./ (1 - rand(nn, 100)), 1, "descend");
qs = quantile(rr', [0.025, 0.5, 0.975]);

ntop = 50;
semilogy(1:ntop, mm(1:ntop), 1:ntop, qs(:, 1:ntop))

semilogy((start_year+nt_start)+(2:nt_fcst), nanmax(RP(:, :, nvals+2), [], 2))


#check the likelihood of the Lytton record under a normal distribution with trend for extreme temp.
file = "/home/nir/data/processed/era5_T_max_100.mat"
load (file)
y = squeeze(v_by_year(959, 160, :)); #Lytton, BC grid cell
clear v_by_year
nd = 81;
[mu, sigma, nu] = regress_params ([ones(nd, 1) t(1:nd)], y(1:nd), [1 t(nd+1)])
-tpdf_log((y(nd+1) - mu)/sigma, nu) + log(sigma) #9.2425
tcdf ((y(nd+1) - mu)/sigma, nu, "upper") #5.0018e-05, close to the maximum for all 100 points and years [4.4202e-05; 1.2124e-04 for 2021]
y(nd+1) - max(y(1:nd)) #5.0 K above the previous record!



#save output, generate plots




plot_dir = [doc_dir 'ktbn/gev/plots/gev_cmp/'];
run_labels = {'_grid', '_region', '_station'};
fcsts = {'GEVD', 'normal'};
nl = numel (run_labels);
nf = numel (fcsts);
for l = 1:nl
  run_label = run_labels{l};
  file_out = ['bgev_fits' run_label '.mat'];
  load (file_out)
  for f = 1:nf
    fcst = fcsts{f};
    switch fcst
      case 'GEVD'
        i = 1;
      case 'normal'
        i = 3;
      otherwise
        error ("undefined fcst method")
    endswitch
    s = fcst_q(:, :, i)(:);
    hist(100*s(isfinite(s)), 100, 100);
    set(gca(),
      'linewidth',2,
      'tickdir','out',
      'ticklength',[0.005,0.005],
      'FontSize',17
      )
    ylabel('% of Observations')
    grid on
    print('-depsc2', '-F:Helvetica:16', '-tight', [plot_dir 'hist_' fcst run_label]);
  endfor
endfor

%xlabel('Forecast percentile')
#print('-depsc', '-tight', [plot_dir 'histu']);

plot(as, sum(sum(fcst_nll, 1), 2)(:)(2:end), '-s')
xlabel('Blending quantile a')
ylabel('Forecast NLL')
print('-deps', [plot_dir 'a_NLL'])

ii = 50;
jj = find(!isfinite(fcst_nll(50, :, 1)));
y = data(1:(nt_start+ii-1), jj);
yn = data(nt_start+ii, jj);
x = t(1:(nt_start+ii-1));
xm = mean(x);
x -= xm;
xn = t(nt_start+ii) - xm;
gev_const_params = gevfit_lmom (y);
gev_nll = @(params) sum (gevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
gev_params_init = [gev_const_params(1) log(gev_const_params(2)) gev_const_params(3) 0];
gev_params = fminunc (gev_nll, gev_params_init);
a = 0.9; b = 0.89; s = 5;
bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4), a, b, s));
bgev_params = fminunc (bgev_nll, gev_params);

yy = linspace(min(y) - 1, yn + 1, 100);
p_gev = gevpdf (yy, gev_params(1), exp(gev_params(2)), gev_params(3) + xn*gev_params(4));
p_bgev = bgevpdf (yy, bgev_params(1), exp(bgev_params(2)), bgev_params(3) + xn*bgev_params(4), a, b, s);

[nn, xx] = hist(y);
w = xx(2) - xx(1);

set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultAxesFontSize', 13);
set(0,'DefaultTextFontSize',13);
set(0,'DefaultAxesFontWeight','bold')
set(0,'DefaultTextFontWeight','bold')
hist(y, xx, 1/w)
hold on
plot(yy, p_gev, ':g', yy, p_bgev)
line ("xdata",[yn, yn], "ydata",[0, 0.5], "linewidth", 1, "color", "k")
xlim([min(yy) max(yy)])
legend('Past', 'GEV', 'bGEV', "location", "northwest")
ylabel("Probability density")
xlabel("Temperature (K)")
print('-depsc', [plot_dir 'example_fcst'])
hold off
#gev_params(3) + xn*gev_params(4) - exp(gev_params(2))/gev_params(1) # GEV upper limit

pos = 2
mean(fcst_nll(:, :, pos)(:), "omitnan")
[h, p, Dn, cv] = kstest (fcst_q(:, :, pos)(:), "CDF", @(x) x)
mean (1 ./ (1 - fcst_q(:, :, pos)(:)) .^ 0.1, "omitnan") - 10/9	
mean (1 ./ (1 - fcst_q(:, :, pos)(:)) .^ 0.5, "omitnan") - 2
mean (1 ./ (1 - fcst_q(:, :, pos)(:)) .^ 0.9, "omitnan") - 10
n = nansum(fcst_q(:, :, pos)(:) > 0.99), nx = sum(isfinite(fcst_q(:, :, pos)(:))), 100*n/nx
n = nansum(fcst_q(:, :, pos)(:) > 0.999), 100*n/nx

#bootstrap confidence interval for difference in NLL
fcst_nll_diff = fcst_nll(:, :, 1) - fcst_nll(:, :, 2);
nc = numel (fcst_nll_diff);
ni = 1000;
nll_diffs = nan (ni, 1);
for i = 1:ni
  nll_diffs(i) = mean (randsample (fcst_nll_diff(:), nc, replacement=true)); 
endfor

#{
mean(nll_diffs)
0.019577
min(nll_diffs)
4.9336e-03
max(nll_diffs)
0.03744
quantile(nll_diffs, [0.025, 0.975])
9.9338e-03   3.1052e-02

nx
11677
5400
12798

#}

